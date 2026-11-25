"""Gazebo with the office world + TurtleBot3 waffle_pi spawned in the hall at (0, 0), facing +x.

    ros2 launch hier_robot sim.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory("hier_robot")
    gazebo_ros = get_package_share_directory("gazebo_ros")
    tb3_gazebo = os.path.join(get_package_share_directory("turtlebot3_gazebo"), "launch")
    world = os.path.join(share, "worlds", "office.world")

    def inc(path, args):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=args.items())

    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", os.environ.get("TURTLEBOT3_MODEL", "waffle_pi")),
        inc(os.path.join(gazebo_ros, "launch", "gzserver.launch.py"), {"world": world}),
        inc(os.path.join(gazebo_ros, "launch", "gzclient.launch.py"), {}),
        inc(os.path.join(tb3_gazebo, "robot_state_publisher.launch.py"), {"use_sim_time": "true"}),
        inc(os.path.join(tb3_gazebo, "spawn_turtlebot3.launch.py"), {"x_pose": "0.0", "y_pose": "0.0"}),
    ])
