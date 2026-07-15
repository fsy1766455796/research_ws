"""Launch the independent point-cloud baseline without changing BEV odometry."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(package="orbbec_bev_odometry",
        executable="pointcloud_odometry_node", name="orbbec_pointcloud_odometry",
        output="screen", parameters=[{"odom_interval_sec": 0.1, "voxel_size": 0.06,
            "min_height": -0.15, "max_correspondence": 0.35, "submap_scans": 12,
            "yaw_gain": 1.016}])])
