"""Offline robot model (no ROS, no Gazebo): unicycle with noise + odometry + SLAM fix.
Used by the tests to check D* + LQR + EKF together."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .level2_dstar import GeometricLayer, costmap_from_grid
from .level3_lqr import Ekf, LqrTracker, wrap


@dataclass
class Obstacle:
    """A box the map does not know about (a person, a moved chair). Seen by the LiDAR when close."""
    x: float
    y: float
    sx: float = 0.4
    sy: float = 0.4


class SimDriver:
    """Unicycle robot with actuator noise, wheel odometry and a SLAM pose fix (5 Hz),
    driven by LQR on the EKF estimate along D* paths. Same noise model as control_benchmark."""

    def __init__(self, geo: GeometricLayer, start_pose, seed: int = 0, dt: float = 0.05,
                 obstacles: Sequence[Obstacle] = (), sense_range: float = 1.2, fix_xy: float = 0.03,
                 fix_th: float = 0.03, use_dstar_repair: bool = True):
        self.geo, self.dt = geo, dt
        self.rng = np.random.default_rng(seed)
        self.true = np.array(start_pose, float)
        self.ekf = Ekf()
        self.R = np.diag([fix_xy ** 2, fix_xy ** 2, fix_th ** 2])
        self.fix_sd = (fix_xy, fix_xy, fix_th)
        self.ekf.init(self._fix(), self.R)
        self.tracker = LqrTracker(dt=dt, goal_tol=0.08)
        self.hidden = list(obstacles)
        self.sense_range = sense_range
        self.use_repair = use_dstar_repair
        self.trajectory: List[Tuple[float, float]] = []
        self.log: List[str] = []
        self.collisions = 0
        self.steps = 0

    def _fix(self):
        return self.true + self.rng.normal(0, self.fix_sd)

    def _sense(self) -> bool:
        """Reveal hidden obstacles within LiDAR range; returns True if the map changed."""
        seen = [o for o in self.hidden if math.hypot(o.x - self.true[0], o.y - self.true[1]) < self.sense_range]
        if not seen:
            return False
        occ = self.geo.occ.copy()
        f = self.geo.frame
        for o in seen:
            self.hidden.remove(o)
            r0, c0 = f.to_cell(o.x - o.sx / 2, o.y - o.sy / 2)
            r1, c1 = f.to_cell(o.x + o.sx / 2, o.y + o.sy / 2)
            occ[r0:r1 + 1, c0:c1 + 1] = 100
            self._seen_obstacles = getattr(self, "_seen_obstacles", []) + [o]
        self.geo.occ_pending = occ
        return True

    def _collides(self) -> bool:
        x, y = self.true[:2]
        for o in getattr(self, "_seen_obstacles", []) + self.hidden:
            if abs(x - o.x) < o.sx / 2 + 0.1 and abs(y - o.y) < o.sy / 2 + 0.1:
                return True
        return self.geo.occ[self.geo.frame.to_cell(x, y)] >= 50

    def drive(self, goal_xy, timeout: float = 400.0, arrive: float = 0.15) -> Tuple[bool, str, float, dict]:
        if math.hypot(goal_xy[0] - self.ekf.pose[0], goal_xy[1] - self.ekf.pose[1]) < arrive:
            return True, "ok", 0.0, {}
        path, msg = self.geo.plan(self.ekf.pose[:2], goal_xy)
        if path is None:
            return False, msg, 0.0, {}
        self.tracker.reset()
        t, cte, replans, in_collision = 0.0, [], 0, False
        fix_every = int(round(0.2 / self.dt))
        while t < timeout:
            est = self.ekf.pose
            if math.hypot(goal_xy[0] - est[0], goal_xy[1] - est[1]) < arrive:
                break
            if self._sense():
                if self.use_repair:
                    newp, m, n = self.geo.update_map(self.geo.occ_pending, est[:2])
                else:    # baseline: plan from scratch on the new map
                    self.geo.occ = self.geo.occ_pending
                    self.geo.cost = costmap_from_grid(self.geo.occ, self.geo.frame.res, self.geo.robot_radius,
                                                      self.geo.inflation, no_go=self.geo.no_go)
                    newp, m = self.geo.plan(est[:2], goal_xy)
                    self.geo.stats["replans"] += 1
                replans += 1
                self.log.append(f"t={t:.1f}s LiDAR: новое препятствие -> D* {m}")
                if newp is None:
                    return False, m, t, {"replans": replans}
                path = newp
                self.tracker.reset()
            v, w, done = self.tracker.command(est, path)
            if done:
                break
            va = v * (1 + self.rng.normal(0, 0.05)) + self.rng.normal(0, 0.005)
            wa = w * (1 + self.rng.normal(0, 0.05)) + self.rng.normal(0, 0.02)
            th = self.true[2]
            self.true += np.array([va * math.cos(th) * self.dt, va * math.sin(th) * self.dt, wa * self.dt])
            self.true[2] = wrap(self.true[2])
            self.ekf.predict(va + self.rng.normal(0, 0.01), wa + self.rng.normal(0, 0.03), self.dt)
            self.steps += 1
            if self.steps % fix_every == 0:
                self.ekf.update(self._fix(), self.R)
            i = path.project(self.true[0], self.true[1], 0, window=len(path.p))
            cte.append(abs(path.cross_track(self.true[0], self.true[1], i)))
            if self.steps % 4 == 0:
                self.trajectory.append((float(self.true[0]), float(self.true[1])))
            c = self._collides()
            self.collisions += int(c and not in_collision)
            in_collision = c
            t += self.dt
        else:
            return False, "LQR: таймаут", t, {"replans": replans}
        rms = math.sqrt(sum(c * c for c in cte) / max(len(cte), 1))
        return True, "ok", t, {"replans": replans, "rms_cte_m": rms, "max_cte_m": max(cte or [0.0])}
