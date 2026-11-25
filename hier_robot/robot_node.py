"""The robot program: connects all levels.  LLM (level 1) -> D* Lite (level 2) -> LQR + EKF (level 3),
on top of RTAB-Map SLAM and the SigLIP2 semantic tensor.

It plans paths itself with D* Lite on RTAB-Map's occupancy grid (/map) and drives with
its own LQR + Kalman filter (no Nav2).

    mode:=orders   (default) you type tasks one by one ("Задача>"); the robot remembers
                   where the cubes are between tasks
    mode:=mission  one task given with -p task:="..." then exit
    mode:=selftest run config/selftest.yaml (tasks + expected cube rooms) -> ~/hier_results/selftest.md
    mode:=explore  drive a tour through all rooms (RTAB-Map builds the map; with mapper_node
                   running, SigLIP2 builds the semantic tensor)

Inputs   /map (RTAB-Map grid), TF map->base_footprint (RTAB-Map pose), /odom, /scan
Outputs  /cmd_vel, /hierarchy/path (current D* path, for RViz), /mission/plan, /mission/result

    ros2 run hier_robot robot_node --ros-args -p use_sim_time:=true -p use_prior:=true -p prior_cubes:=true
"""
from __future__ import annotations

import csv
import json
import math
import os
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .level1_llm import ActionExecutor, SemanticLlmPlanner, run_hierarchical, world_from_tensor
from .level2_dstar import GeometricLayer, GridFrame
from .level3_lqr import Ekf, LqrTracker
from .semantic_tensor import SemanticTensor
from .world import World

DEFAULT_TOUR = "0.0,0.0; 0.0,0.55; 2.5,-0.5; 2.5,-2.3; 2.5,-0.5; -3.5,-0.5; -3.5,-2.4; -3.5,-0.5; " \
               "-4.0,0.5; -4.0,2.4; -4.0,0.5; 4.0,0.5; 4.0,2.4; 4.0,0.5; 0.0,0.0"
LOOK_POINTS = {(0.0, 0.55), (2.5, -2.3), (-3.5, -2.4), (-4.0, 2.4), (4.0, 2.4)}


def quat_to_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class RobotNode(Node):
    def __init__(self):
        super().__init__("robot_node")
        share = get_package_share_directory("hier_robot")
        p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.mode = p("mode", "orders")
        self.task = p("task", "Отвези все кубики из комнаты, где стоит стол со стульями, в комнату со стеллажами")
        self.goal_str = p("goal", "")
        self.world_file = p("world_file", os.path.join(share, "config", "building.yaml"))
        self.tensor_file = os.path.expanduser(p("tensor_file", "~/hier_results/semantic/semantic_tensor.npz"))
        self.llm_context = p("context", "tensor")         # tensor | objects | none (Node.context is taken by rclpy)
        self.ollama_url = p("ollama_url", "http://localhost:11434")
        self.model = p("ollama_model", os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"))
        self.max_replans = int(p("max_replans", 3))
        self.map_topic = p("map_topic", "/map")
        self.map_frame, self.base_frame = p("map_frame", "map"), p("base_frame", "base_footprint")
        self.stamped = bool(p("cmd_vel_stamped", False))
        self.dt = 1.0 / float(p("rate_hz", 20.0))
        self.v_ref = float(p("v_ref", 0.18))
        self.stop_distance = float(p("stop_distance", 0.20))
        self.blocked_timeout = float(p("blocked_timeout", 15.0))
        self.goal_timeout = float(p("goal_timeout", 300.0))
        self.manip_time = float(p("manip_time", 8.0))
        self.tour = p("tour", DEFAULT_TOUR)
        self.look_seconds = float(p("look_seconds", 45.0))   # one slow 360° turn per room (~6 SigLIP2 frames)
        # prior knowledge: add the known furniture from world_file ("furniture") to the perceived tensor.
        # SigLIP2-base hardly recognises plain Gazebo furniture; cubes always come from perception.
        self.use_prior = bool(p("use_prior", False))
        # quick demo: also take the cube start positions from world_file ("known_cubes"), no SigLIP2 tour needed
        self.prior_cubes = bool(p("prior_cubes", False))
        self.results_csv = os.path.expanduser(p("results_csv", "~/hier_results/missions.csv"))
        self.selftest_file = p("selftest_file", "")

        with open(self.world_file, encoding="utf-8") as f:
            self.world_cfg = yaml.safe_load(f)
        self.no_go = [r["region"] for r in self.world_cfg["rooms"].values() if r.get("restricted") and r.get("region")]
        self.regions = {n: r.get("region") for n, r in self.world_cfg["rooms"].items()}
        self.building = [r for r in self.regions.values() if r]   # outside these polygons = no-go

        self.ekf, self.tracker = Ekf(), LqrTracker(dt=self.dt, v_ref=self.v_ref, goal_tol=0.15)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.grid_msg, self.geo, self.geo_stamp = None, None, None
        self.min_front = float("inf")
        self._last_odom_t, self._last_fix = None, 0.0
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))  # RTAB-Map /map is latched
        self.create_subscription(Odometry, "odom", self._on_odom, 20)
        self.create_subscription(LaserScan, "scan", self._on_scan, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(TwistStamped if self.stamped else Twist, "cmd_vel", 10)
        self.path_pub = self.create_publisher(PathMsg, "hierarchy/path", 1)
        self.plan_pub = self.create_publisher(String, "mission/plan", 10)
        self.result_pub = self.create_publisher(String, "mission/result", 10)
        self.track = []

    # ------------------------------------------------------------------ inputs
    def _on_map(self, msg: OccupancyGrid):
        self.grid_msg = msg

    def _on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        best = float("inf")
        for i in range(n):
            a = msg.angle_min + i * msg.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))
            r = msg.ranges[i]
            if abs(a) < math.radians(20) and msg.range_min < r < msg.range_max:
                best = min(best, r)
        self.min_front = best

    def _on_odom(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.ekf.initialised and self._last_odom_t is not None:
            dt = min(max(t - self._last_odom_t, 0.0), 0.2)
            self.ekf.predict(msg.twist.twist.linear.x, msg.twist.twist.angular.z, dt)
        self._last_odom_t = t

    def _slam_pose(self):
        """RTAB-Map pose = TF map -> base_footprint."""
        try:
            tr = self.tf.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        t, q = tr.transform.translation, tr.transform.rotation
        return t.x, t.y, quat_to_yaw(q)

    def _fuse_slam(self, force=False):
        """EKF update with the RTAB-Map pose at 5 Hz (fixed covariance: RTAB-Map gives none on TF)."""
        now = time.time()
        if not force and now - self._last_fix < 0.2:
            return
        z = self._slam_pose()
        if z is None:
            return
        self._last_fix = now
        R = np.diag([0.03 ** 2, 0.03 ** 2, 0.03 ** 2])
        if not self.ekf.initialised:
            self.ekf.init(z, R)
        else:
            self.ekf.update(z, R)

    def wait_ready(self, timeout=120.0) -> bool:
        log = self.get_logger()
        log.info(f"waiting for RTAB-Map ({self.map_topic} and TF {self.map_frame}->{self.base_frame}) ...")
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            self._fuse_slam(force=True)
            if self.grid_msg is not None and self.ekf.initialised:
                log.info("RTAB-Map ready")
                return True
        log.error("RTAB-Map not ready: is rtabmap running? (ros2 topic list | grep map)")
        return False

    # ------------------------------------------------------------------ middle level
    def _grid(self):
        m = self.grid_msg
        info = m.info
        occ = np.asarray(m.data, np.int8).reshape(info.height, info.width)
        frame = GridFrame(info.origin.position.x, info.origin.position.y, info.resolution, info.height, info.width)
        return occ, frame, (info.width, info.height, round(info.origin.position.x, 3),
                            round(info.origin.position.y, 3))

    def _refresh_geo(self, robot_xy):
        """Take the latest RTAB-Map grid. Same size/origin -> incremental D* repair;
        the map grew -> new costmap and a fresh D* search."""
        occ, frame, key = self._grid()
        if self.geo is None or key != self.geo_stamp:
            self.geo = GeometricLayer(occ, frame, self.no_go, allowed_polys=self.building)
            self.geo_stamp = key
            return None, "new"
        path, msg, n = self.geo.update_map(occ, robot_xy)
        return (path, msg) if n else (None, "same")

    # ------------------------------------------------------------------ bottom level
    def _publish(self, v, w):
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        else:
            msg = Twist()
            msg.linear.x, msg.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)

    def _publish_path(self, path):
        msg = PathMsg()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path.p[::5]:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def drive(self, goal_xy, arrive=0.20):
        """D* path on the RTAB-Map grid + LQR on the EKF pose. Returns (ok, msg, seconds, stats)."""
        log = self.get_logger()
        self._fuse_slam(force=True)
        x, y, _ = self.ekf.pose
        if math.hypot(goal_xy[0] - x, goal_xy[1] - y) < arrive:
            return True, "ok", 0.0, {}
        self._refresh_geo((x, y))
        path, msg = self.geo.plan((x, y), goal_xy)
        if path is None:
            log.warn(msg)
            return False, msg, 0.0, {}
        self._publish_path(path)
        log.info(f"[2 D*] путь {path.length:.1f} м до ({goal_xy[0]:.1f}, {goal_xy[1]:.1f}), "
                 f"раскрыто {self.geo.ds.expanded} клеток")
        self.tracker.reset()
        clock = self.get_clock()
        now = lambda: clock.now().nanoseconds * 1e-9  # noqa: E731
        t0, blocked_since, last_map_check, cte, replans, extensions = now(), None, time.time(), [], 0, 0
        map_seen = self.grid_msg
        while True:
            tick_wall, tick = time.time(), now()
            self._fuse_slam()
            x, y, th = self.ekf.pose
            if math.hypot(goal_xy[0] - x, goal_xy[1] - y) < arrive:
                break
            # RTAB-Map published a new grid (new walls / obstacles): repair the D* path
            if self.grid_msg is not map_seen and time.time() - last_map_check > 2.0:
                map_seen, last_map_check = self.grid_msg, time.time()
                self._publish(0.0, 0.0)        # stand still while D* works
                newp, m = self._refresh_geo((x, y))
                if m == "new":
                    newp, m = self.geo.plan((x, y), goal_xy)
                if newp is not None:
                    remaining = path.length - float(path.s[min(self.tracker.idx, len(path.s) - 1)])
                    changed = abs(newp.length - remaining) > 0.5
                    path = newp
                    replans += 1
                    self.tracker.reset()
                    self._publish_path(path)
                    if changed:   # the map grows every second; log only real route changes
                        log.info(f"[2 D*] карта RTAB-Map обновилась -> маршрут изменён, {path.length:.1f} м")
                elif m not in ("same",):
                    self._publish(0.0, 0.0)
                    return False, m, tick - t0, {"replans": replans}
            if self.min_front < self.stop_distance:
                self._publish(0.0, 0.0)
                if blocked_since is None:
                    blocked_since = tick
                    log.warn(f"[3 LQR] LiDAR: препятствие {self.min_front:.2f} м, стоп")
                elif tick - blocked_since > self.blocked_timeout:
                    return False, "путь заблокирован (LiDAR)", tick - t0, {"replans": replans}
            else:
                blocked_since = None
                v, w, done = self.tracker.command((x, y, th), path)
                cte.append(abs(path.cross_track(x, y, self.tracker.idx)))
                if done:
                    if math.hypot(goal_xy[0] - x, goal_xy[1] - y) < 2 * arrive or extensions >= 5:
                        break
                    # the goal was outside the map RTAB-Map had so far (the path ended at its edge):
                    # the map has grown meanwhile, plan the rest of the way
                    extensions += 1
                    self._publish(0.0, 0.0)
                    self._refresh_geo((x, y))
                    newp, m = self.geo.plan((x, y), goal_xy)
                    if newp is None:
                        return False, m, tick - t0, {"replans": replans}
                    path = newp
                    self.tracker.reset()
                    self._publish_path(path)
                    log.info("[2 D*] цель была за краем карты, путь продлён")
                    continue
                self._publish(v, w)
            if tick - t0 > self.goal_timeout:
                self._publish(0.0, 0.0)
                return False, "LQR: таймаут", tick - t0, {"replans": replans}
            deadline = tick_wall + self.dt
            while time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=max(0.0, deadline - time.time()))
        self._publish(0.0, 0.0)
        rms = math.sqrt(sum(c * c for c in cte) / max(len(cte), 1))
        st = {"path_m": round(path.length, 2), "time_s": round(now() - t0, 1), "rms_cte_m": round(rms, 3),
              "replans": replans}
        self.track.append(st)
        log.info(f"[3 LQR] прибыл, RMS отклонения {100 * rms:.1f} см, перепланирований D* {replans}")
        return True, "ok", now() - t0, st

    def look_around(self, seconds=None, w=None):
        seconds = seconds or self.look_seconds
        w = w or 2 * math.pi / seconds          # exactly one turn
        end = time.time() + seconds
        while time.time() < end:
            self._publish(0.0, w)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._publish(0.0, 0.0)

    # ------------------------------------------------------------------ modes
    def run_tour(self):
        pts = [tuple(float(v) for v in s.split(",")) for s in self.tour.split(";") if s.strip()]
        for k, (x, y) in enumerate(pts):
            ok, msg, _, _ = self.drive((x, y))
            self.get_logger().info(f"[тур {k + 1}/{len(pts)}] ({x}, {y}): {msg}")
            if ok and (x, y) in LOOK_POINTS:
                self.look_around()
        self.get_logger().info("тур завершён: карта RTAB-Map и семантический тензор построены")

    def setup_mission(self):
        """Tensor, world, LLM planner and executor (built once; reused by interactive mode)."""
        log = self.get_logger()
        tensor = None
        if os.path.exists(self.tensor_file) or self.use_prior or self.prior_cubes:
            doors = [tuple(d) for d in self.world_cfg.get("doors", [])] or None
            if os.path.exists(self.tensor_file):
                tensor = SemanticTensor.load(self.tensor_file)
            else:
                from .perception import SemanticConfig
                share = get_package_share_directory("hier_robot")
                cfg = SemanticConfig.from_yaml(os.path.join(share, "config", "siglip_labels.yaml"))
                tensor = SemanticTensor.empty(cfg.object_classes, self.world_cfg["rooms"], doors)
            n_seen = len(tensor.ids)
            prior = list(self.world_cfg.get("furniture", [])) if self.use_prior else []
            if self.prior_cubes:
                prior += self.world_cfg.get("known_cubes", [])
            if prior:
                tensor = tensor.with_prior(prior, self.world_cfg["rooms"], doors)
                log.info(f"априорные знания: + мебель из {os.path.basename(self.world_file)} "
                         f"(увидено SigLIP2: {n_seen}, всего в тензоре: {len(tensor.ids)})")
            world = world_from_tensor(tensor, self.world_cfg.get("start_room", "hall"))
            log.info(f"семантический тензор: {tensor.summary()}")
            log.info("тензор для LLM:\n" + tensor.to_prompt())
        else:
            log.warn(f"нет {self.tensor_file}: беру объекты из {self.world_file} (без тензора)")
            world = World.from_yaml(self.world_file)
            self.llm_context = "none"
        planner = SemanticLlmPlanner(tensor, self.model, self.ollama_url, context=self.llm_context)
        if self.geo is None and self.grid_msg is not None:
            self._refresh_geo(tuple(self.ekf.pose[:2]))
        ex = ActionExecutor(world, self.geo, self.drive, self.regions, manip_time=0.0, geo_getter=lambda: self.geo,
                          pose_of=(lambda n: tuple(tensor.P[tensor.index(n)]) if tensor and n in tensor.ids else None))
        orig_execute = ex.execute

        def execute(step, state):   # pick/place: wait for a person to put the cube on/off the robot
            ok, msg, viol, dt = orig_execute(step, state)
            log.info(f"[1 LLM] шаг {step}: {'выполнен' if ok else 'ОШИБКА: ' + msg}")
            if ok and step.action in ("pick", "place"):
                log.info(f"MANIP: {step.action} {step.target}, жду {self.manip_time:.0f} с")
                end = time.time() + self.manip_time
                while time.time() < end:
                    rclpy.spin_once(self, timeout_sec=0.1)
                dt += self.manip_time
            return ok, msg, viol, dt
        ex.execute = execute
        goal = {}
        for item in filter(None, self.goal_str.split(",")):
            o, r = item.split(":")
            goal[o.strip()] = [x.strip() for x in r.split("|")]

        self._m = dict(tensor=tensor, world=world, planner=planner, ex=ex, goal=goal)
        return world

    def run_task(self, task, state=None):
        log = self.get_logger()
        m = self._m

        def on_plan(plan):
            text = " -> ".join(map(str, plan))
            log.info(f"[1 LLM] План: {text}")
            self.plan_pub.publish(String(data=text))
        log.info(f"[1 LLM] Задача: {task}  (LLM {self.model}, контекст {self.llm_context})")
        self.track = []
        res = run_hierarchical(task, m["world"], m["planner"], m["ex"], m["goal"], m["tensor"], scenario="ros",
                               max_replans=self.max_replans, on_plan=on_plan, state=state)
        for line in res.log:
            log.info(line)
        row = res.row()
        row.update({"task": task, "model": self.model,
                    "mean_rms_cte_m": round(float(np.mean([t["rms_cte_m"] for t in self.track])), 3) if self.track else None,
                    "dstar_replans": sum(t.get("replans", 0) for t in self.track)})
        log.info("Итог: " + json.dumps(row, ensure_ascii=False))
        self.result_pub.publish(String(data=json.dumps(row, ensure_ascii=False)))
        os.makedirs(os.path.dirname(self.results_csv) or ".", exist_ok=True)
        new = not os.path.exists(self.results_csv)
        with open(self.results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        return res

    def run_mission(self):
        self.setup_mission()
        self.run_task(self.task)

    def run_interactive(self):
        """Type tasks one after another; the map, the tensor and where the cubes are stay between tasks."""
        world = self.setup_mission()
        state = world.initial_state()
        print("\n=== Робот готов. Пишите задачу (Enter без текста или 'q' — выход). ===", flush=True)
        while True:
            print(f"\nСейчас: {state.describe()}", flush=True)
            try:
                task = input("Задача> ").strip()
            except EOFError:
                break
            if task in ("", "q", "quit", "exit"):
                break
            self.run_task(task, state=state)

    def run_selftest(self):
        """Run the tasks of config/selftest.yaml one by one and check where the cubes end up."""
        log = self.get_logger()
        share = get_package_share_directory("hier_robot")
        with open(self.selftest_file or os.path.join(share, "config", "selftest.yaml"), encoding="utf-8") as f:
            tests = yaml.safe_load(f)["tests"]
        world = self.setup_mission()
        lines = ["# Self-test", "", f"LLM: {self.model}", "",
                 "| Test | Result | Plan executed | Problems | LLM calls | Robot time, s |", "|---|---|---|---|---|---|"]
        passed = 0
        start_objects = dict(world.objects)       # every test starts from the original cube positions
        for t in tests:
            world.objects.clear()
            world.objects.update(start_objects)
            state = world.initial_state()
            here = self._m["ex"]._room_of_xy(tuple(self.ekf.pose[:2]))
            state.robot_room = here
            log.info(f"===== SELFTEST {t['id']}: {t['task']}")
            res = self.run_task(t["task"], state=state)
            wrong = [f"{c} in {state.obj_room.get(c)} (expected {r})"
                     for c, r in t["expect"].items() if state.obj_room.get(c) != r]
            if res.safety_violations:
                wrong.append(f"{res.safety_violations} safety violation(s)")
            if state.holding != "none":
                wrong.append(f"still holding {state.holding}")
            ok = not wrong
            passed += ok
            log.info(f"===== SELFTEST {t['id']}: {'PASS' if ok else 'FAIL'} {'; '.join(wrong)}")
            lines.append(f"| {t['id']} | {'PASS' if ok else 'FAIL'} | {res.final_plan or '-'} | "
                         f"{'; '.join(wrong) or '-'} | {res.llm_calls} | {res.robot_time_s} |")
        lines += ["", f"**{passed} / {len(tests)} passed**"]
        out = os.path.join(os.path.dirname(self.results_csv) or ".", "selftest.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log.info(f"SELFTEST: {passed}/{len(tests)} passed -> {out}")


def main():
    rclpy.init()
    node = RobotNode()
    try:
        if node.wait_ready():
            if node.mode in ("explore", "tour"):
                node.run_tour()
            elif node.mode == "selftest":
                node.run_selftest()
            elif node.mode in ("orders", "interactive"):
                node.run_interactive()
            else:
                node.run_mission()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish(0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
