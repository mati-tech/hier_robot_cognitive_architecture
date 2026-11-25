"""HW3 ROS 2 node: SigLIP2 semantic layer on top of SLAM.

Run together with geometric SLAM (slam_toolbox / Cartographer) or with AMCL on a
saved map. It needs the TF map -> base_footprint, a camera and the LiDAR.

    ros2 run hier_robot mapper_node --ros-args \
        -p image_topic:=/camera/image_raw -p hfov_deg:=62.2 -p rate_hz:=1.0

Subscribes   image_topic (sensor_msgs/Image), /scan (LaserScan), TF map->base_footprint
Publishes    /semantic_map/markers  visualization_msgs/MarkerArray  (RViz: spheres + names)
             /semantic_map/graph    std_msgs/String  (JSON graph: object-в-room, рядом с, doors)
Saves        <out_dir>/semantic_map.json, semantic_graph.dot, world_semantic.yaml,
             semantic_tensor.npz + tensor_prompt.txt (relation tensor for the hierarchical planner)
             (world_semantic.yaml can be passed to mission_node as world_file -> HW1 planner
             works with the objects the robot found itself)
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

import yaml

from .semantic_tensor import SemanticTensor
from .perception import (SemanticConfig, SemanticMap, SiglipScorer, lidar_range, pixel_bearing,
                           split_strips, strip_labels)

COLORS = {"cube": (1.0, 0.5, 0.0), "door": (0.2, 0.6, 1.0), "chair": (0.3, 0.8, 0.3),
          "table": (0.6, 0.4, 0.2), "shelf": (0.7, 0.3, 0.7), "trash_can": (0.5, 0.5, 0.5)}


def image_to_pil(msg: Image):
    from PIL import Image as PILImage
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
    elif enc in ("rgba8", "bgra8"):
        arr = buf.reshape(msg.height, msg.step // 4, 4)[:, :msg.width, :3]
        if enc == "bgra8":
            arr = arr[:, :, ::-1]
    elif enc == "mono8":
        arr = np.stack([buf.reshape(msg.height, msg.step)[:, :msg.width]] * 3, axis=-1)
    else:
        raise ValueError(f"unsupported encoding {msg.encoding}")
    return PILImage.fromarray(np.ascontiguousarray(arr))


class SemanticMapper(Node):
    def __init__(self):
        super().__init__("semantic_mapper")
        share = get_package_share_directory("hier_robot")
        p = self.declare_parameter
        labels_file = p("labels_file", os.path.join(share, "config", "siglip_labels.yaml")).value
        world_file = p("world_file", os.path.join(share, "config", "world.yaml")).value
        self.image_topic = p("image_topic", "/camera/image_raw").value
        self.hfov = math.radians(p("hfov_deg", 62.2).value)      # RPi camera v2: 62.2, Gazebo waffle: 62.2
        self.cam_yaw = math.radians(p("camera_yaw_deg", 0.0).value)
        self.period = 1.0 / p("rate_hz", 1.0).value
        self.map_frame = p("map_frame", "map").value
        self.base_frame = p("base_frame", "base_footprint").value
        self.out_dir = os.path.expanduser(p("out_dir", "~/semantic_map").value)
        self.device = p("device", "cpu").value
        os.makedirs(self.out_dir, exist_ok=True)

        self.cfg = SemanticConfig.from_yaml(labels_file)
        with open(world_file, encoding="utf-8") as f:
            world = yaml.safe_load(f)
        self.start_room = world.get("start_room", next(iter(world["rooms"])))
        self.doors = [tuple(d) for d in world["doors"]] if world.get("doors") else None
        self.map = SemanticMap(self.cfg, world["rooms"])
        self.get_logger().info(f"Loading {self.cfg.model} on {self.device} (first run downloads ~1.5 GB) ...")
        self.scorer = SiglipScorer(self.cfg, self.device)
        self.get_logger().info("SigLIP2 ready")

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.last_image, self.last_scan = None, None
        self.create_subscription(Image, self.image_topic, lambda m: setattr(self, "last_image", m),
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, "scan", lambda m: setattr(self, "last_scan", m),
                                 qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, "semantic_map/markers", 10)
        self.graph_pub = self.create_publisher(String, "semantic_map/graph", 10)
        self.tensor_pub = self.create_publisher(String, "semantic_map/tensor", 10)
        self.create_timer(self.period, self.step)
        self.create_timer(10.0, self.save)
        self.frames, self.infer_time = 0, 0.0

    def robot_pose(self):
        try:
            tr = self.tf.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        t, q = tr.transform.translation, tr.transform.rotation
        return t.x, t.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def step(self):
        img_msg, scan = self.last_image, self.last_scan
        pose = self.robot_pose()
        if img_msg is None or scan is None or pose is None:
            return
        image = image_to_pil(img_msg)
        crops = split_strips(image, self.cfg.strips)
        t0 = time.time()
        scores = self.scorer.score([c for c, _ in crops])
        self.infer_time += time.time() - t0
        self.frames += 1
        dets = strip_labels(scores, self.scorer.labels, self.cfg)
        bearings = [pixel_bearing(u, image.width, self.hfov, self.cam_yaw) for _, u in crops]
        added = self.map.integrate(
            pose, dets, bearings,
            lambda b: lidar_range(scan.ranges, scan.angle_min, scan.angle_increment, b,
                                  max_range=min(self.cfg.max_range, scan.range_max)),
            feats=list(scores))
        if added:
            seen = sorted({d[0] for d in dets if d})
            self.get_logger().info(f"кадр {self.frames}: {', '.join(seen)}; ориентиров {len(self.map.confirmed())}")
        self.publish()

    def publish(self):
        if not rclpy.ok():   # Ctrl+C / pkill arrived during a slow SigLIP2 frame: ROS is already shut down
            return
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for k, lm in enumerate(self.map.confirmed()):
            r, g, b = COLORS.get(lm.label, (1.0, 1.0, 1.0))
            for kind in ("sphere", "text"):
                m = Marker()
                m.header.frame_id, m.header.stamp = self.map_frame, now
                m.ns, m.id = f"semantic_{kind}", k
                m.type = Marker.SPHERE if kind == "sphere" else Marker.TEXT_VIEW_FACING
                m.pose.position.x, m.pose.position.y = lm.x, lm.y
                m.pose.position.z = 0.1 if kind == "sphere" else 0.4
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.2 if kind == "sphere" else 0.15
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
                if kind == "text":
                    m.text = f"{lm.id} ({self.map.room_of(lm.x, lm.y)})"
                arr.markers.append(m)
        self.marker_pub.publish(arr)
        self.graph_pub.publish(String(data=self.map.to_json()))

    def save(self):
        with open(os.path.join(self.out_dir, "semantic_map.json"), "w", encoding="utf-8") as f:
            f.write(self.map.to_json())
        with open(os.path.join(self.out_dir, "semantic_graph.dot"), "w", encoding="utf-8") as f:
            f.write(self.map.to_dot())
        with open(os.path.join(self.out_dir, "world_semantic.yaml"), "w", encoding="utf-8") as f:
            f.write(self.map.to_world_yaml(self.start_room))
        try:
            tensor = SemanticTensor.from_semantic_map(self.map, doors=self.doors)
            tensor.save(os.path.join(self.out_dir, "semantic_tensor.npz"))
            prompt = tensor.to_prompt()
            with open(os.path.join(self.out_dir, "tensor_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            if rclpy.ok():   # the final save() in main() runs after ROS shutdown: files only
                self.tensor_pub.publish(String(data=prompt))
        except Exception as e:  # noqa: BLE001  (the HW3 outputs above are already saved)
            self.get_logger().warn(f"tensor not saved: {e}")
        if self.frames:
            self.get_logger().info(f"saved to {self.out_dir}; SigLIP2 {self.infer_time / self.frames:.2f} s/кадр")


def main():
    rclpy.init()
    node = SemanticMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
