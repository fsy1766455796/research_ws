"""ROS 2 entry point for planar scan-to-submap point-cloud odometry."""

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from .cloud_preprocessor import PreprocessConfig
from .planar_icp import IcpConfig
from .pointcloud_odom_core import PointCloudOdomCore
from .se2 import components


def xyz_view(message):
    """Return zero-copy XYZ fields for the common float32 PointCloud2 layout."""
    offsets = {field.name: field.offset for field in message.fields}
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                      "offsets": [offsets[k] for k in ("x", "y", "z")],
                      "itemsize": message.point_step})
    records = np.frombuffer(message.data, dtype=dtype, count=message.width * message.height)
    return np.column_stack((records["x"], records["y"], records["z"]))


class PointCloudOdometryNode(Node):
    def __init__(self):
        super().__init__("orbbec_pointcloud_odometry")
        defaults = {
            "pointcloud_topic": "/camera/depth/points", "odom_topic": "/pointcloud_odom",
            "frame_id": "pointcloud_odom", "child_frame_id": "base_link",
            "odom_interval_sec": 0.1, "voxel_size": 0.06, "min_range": 0.35,
            "max_range": 6.0, "min_height": -0.15, "max_height": 1.2,
            "max_points": 9000, "submap_scans": 12, "max_correspondence": 0.35,
            "max_iterations": 18, "min_correspondences": 180,
            "yaw_gain": 1.016,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        p = lambda key: self.get_parameter(key).value
        preprocess = PreprocessConfig(voxel_size=p("voxel_size"), min_range=p("min_range"),
            max_range=p("max_range"), min_height=p("min_height"), max_height=p("max_height"),
            max_points=p("max_points"))
        icp = IcpConfig(max_iterations=p("max_iterations"),
            max_correspondence=p("max_correspondence"), min_correspondences=p("min_correspondences"))
        self.core = PointCloudOdomCore(preprocess, icp, p("submap_scans"), yaw_gain=p("yaw_gain"))
        self.frame_id, self.child_frame = p("frame_id"), p("child_frame_id")
        self.interval_ns = int(float(p("odom_interval_sec")) * 1e9)
        self.last_stamp_ns = None
        self.publisher = self.create_publisher(Odometry, p("odom_topic"), 10)
        self.debug = self.create_publisher(String, "/pointcloud_odometry/debug", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(PointCloud2, p("pointcloud_topic"), self.on_cloud, qos_profile_sensor_data)

    def on_cloud(self, message):
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if self.last_stamp_ns is not None and stamp_ns - self.last_stamp_ns < self.interval_ns:
            return
        self.last_stamp_ns = stamp_ns
        result = self.core.process(xyz_view(message))
        x, y, yaw = components(result.pose)
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        odom = Odometry(); odom.header.stamp = message.header.stamp
        odom.header.frame_id = self.frame_id; odom.child_frame_id = self.child_frame
        odom.pose.pose.position.x = x; odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = qz; odom.pose.pose.orientation.w = qw
        self.publisher.publish(odom)
        transform = TransformStamped(); transform.header = odom.header
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = x; transform.transform.translation.y = y
        transform.transform.rotation.z = qz; transform.transform.rotation.w = qw
        self.tf.sendTransform(transform)
        debug = String(); debug.data = json.dumps({"accepted": result.accepted, "x_m": x,
            "y_m": y, "yaw_deg": math.degrees(yaw), "rmse_m": result.rmse,
            "inlier_ratio": result.inlier_ratio, "correspondences": result.correspondences,
            "processing_ms": result.processing_ms, "points": result.filtered_points})
        self.debug.publish(debug)


def main(args=None):
    rclpy.init(args=args); node = PointCloudOdometryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()
