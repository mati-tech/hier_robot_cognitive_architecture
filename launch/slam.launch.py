"""RTAB-Map SLAM (+ RViz, + optionally the SigLIP2 mapper).

    ros2 launch hier_robot slam.launch.py                  # RTAB-Map + RViz
    ros2 launch hier_robot slam.launch.py semantic:=true   # + SigLIP2 mapper (builds the tensor)

RTAB-Map runs in 2D LiDAR mode (TurtleBot3 Waffle Pi has an RGB camera but no depth):
ICP scan matching on /scan on top of wheel odometry, proximity loop closures, 2D
occupancy grid on /map, TF map -> odom. The SigLIP2 mapper takes the robot pose from that
TF, so every recognised object and the relation tensor live in the RTAB-Map frame.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim = LaunchConfiguration("use_sim_time")
    cam = LaunchConfiguration("use_camera")
    share = FindPackageShare("hier_robot")
    rtab_params = {
        "use_sim_time": ParameterValue(sim, value_type=bool),
        "frame_id": "base_footprint",
        "odom_frame_id": "odom",            # wheel odometry from TF (Gazebo / TurtleBot3 bringup)
        "map_frame_id": "map",
        "subscribe_depth": False,
        "subscribe_rgbd": False,
        "subscribe_rgb": ParameterValue(cam, value_type=bool),
        "subscribe_scan": True,
        "approx_sync": True,
        "sync_queue_size": 30,
        "topic_queue_size": 30,
        "wait_for_transform": 0.3,
        # RTAB-Map parameters are strings
        "Reg/Strategy": "1",                 # ICP registration on the laser scan
        "Reg/Force3DoF": "true",             # planar robot: x, y, yaw
        "RGBD/NeighborLinkRefining": "true",  # correct odometry with ICP between nodes
        "RGBD/ProximityBySpace": "true",     # loop closures by scan matching to nearby nodes
        "RGBD/AngularUpdate": "0.05",
        "RGBD/LinearUpdate": "0.05",
        "RGBD/OptimizeFromGraphEnd": "false",
        "Mem/IncrementalMemory": "true",
        "Grid/Sensor": "0",                  # occupancy grid from the laser scan
        "Grid/RangeMax": "3.5",              # LDS-01/02 range
        "Grid/CellSize": "0.05",
        "Grid/RayTracing": "true",           # clear free space along each beam
        "Icp/VoxelSize": "0.05",
        "Icp/MaxCorrespondenceDistance": "0.1",
        "Icp/PointToPlane": "false",
        "Optimizer/GravitySigma": "0",
        "Vis/MinInliers": "12",
    }
    rtabmap = Node(
        package="rtabmap_slam", executable="rtabmap", name="rtabmap", output="screen",
        parameters=[rtab_params],
        remappings=[("scan", "/scan"), ("rgb/image", LaunchConfiguration("image_topic")),
                    ("rgb/camera_info", LaunchConfiguration("camera_info_topic"))],
        arguments=["-d"])                    # start with an empty database every time
    mapper = Node(
        package="hier_robot", executable="mapper_node", name="mapper_node", output="screen",
        condition=IfCondition(LaunchConfiguration("semantic")),
        parameters=[{"use_sim_time": sim,
                     "image_topic": LaunchConfiguration("image_topic"),
                     "rate_hz": LaunchConfiguration("rate_hz"),
                     "world_file": PathJoinSubstitution([share, "config", LaunchConfiguration("world")]),
                     "labels_file": PathJoinSubstitution([share, "config", LaunchConfiguration("labels")]),
                     "out_dir": LaunchConfiguration("out_dir")}])
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        condition=IfCondition(LaunchConfiguration("rviz")),
        arguments=["-d", PathJoinSubstitution([share, "config", "view.rviz"])],
        parameters=[{"use_sim_time": sim}])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_camera", default_value="false"),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
        DeclareLaunchArgument("semantic", default_value="false"),
        DeclareLaunchArgument("rate_hz", default_value="0.5"),
        DeclareLaunchArgument("world", default_value="building.yaml"),
        DeclareLaunchArgument("labels", default_value="siglip_labels.yaml"),
        DeclareLaunchArgument("out_dir", default_value="~/hier_results/semantic"),
        DeclareLaunchArgument("rviz", default_value="true"),
        rtabmap, mapper, rviz,
    ])
