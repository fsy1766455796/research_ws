import math
import time

import cv2
import numpy as np

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64, String
from tf2_ros import TransformBroadcaster

from orbbec_bev_odometry.global_bev_map import GlobalBEVMap
from orbbec_bev_odometry.yaw_filter import gate_yaw_motion


# 将 ROS PointCloud2 转换为算法内部使用的坐标顺序：
# [横向 x, 竖直 y, 前向 z]，并统一换算为毫米。
def extract_xyz_from_pointcloud2(
    msg,
    scale_mm,
    lateral_axis_index,
    vertical_axis_index,
    forward_axis_index,
    lateral_sign,
    vertical_sign,
    forward_sign,
):
    # Orbbec ROS 驱动发布的点云字段单位为米，先取出所有有效 XYZ 点。
    points = point_cloud2.read_points_numpy(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    xyz = np.asarray(points)
    if xyz.dtype.fields is not None:
        xyz = np.column_stack((xyz["x"], xyz["y"], xyz["z"])).astype(np.float32)
    else:
        xyz = xyz.astype(np.float32)
    xyz = xyz.reshape(-1, 3)

    # 轴索引和符号通过参数配置，便于适配不同相机坐标系。
    mapped = np.column_stack(
        (
            xyz[:, int(lateral_axis_index)],
            xyz[:, int(vertical_axis_index)],
            xyz[:, int(forward_axis_index)],
        )
    ).astype(np.float32)
    mapped[:, 0] *= float(lateral_sign)
    mapped[:, 1] *= float(vertical_sign)
    mapped[:, 2] *= float(forward_sign)
    return mapped * float(scale_mm)


def ground_plane_mask(
    xyz,
    distance_threshold_mm,
    max_tilt_deg,
    ransac_iterations,
    sample_size,
    min_inliers,
    min_inlier_ratio,
    candidate_min_y_mm,
):
    """返回属于地面的点掩码；找不到可信地面时返回全 False。"""
    point_count = int(xyz.shape[0])
    no_ground = np.zeros(point_count, dtype=bool)
    if point_count < 3:
        return no_ground

    # Orbbec 光学坐标的 y 正方向朝下，地面通常位于图像下半部。
    # 只用偏下方的点拟合，可显著降低桌面、墙面主导 RANSAC 的概率。
    candidate_indices = np.flatnonzero(xyz[:, 1] >= float(candidate_min_y_mm))
    if candidate_indices.size < max(3, int(min_inliers)):
        return no_ground

    rng = np.random.default_rng(0)
    max_samples = max(3, int(sample_size))
    if candidate_indices.size > max_samples:
        candidate_indices = rng.choice(
            candidate_indices, size=max_samples, replace=False
        )
    candidates = xyz[candidate_indices].astype(np.float64, copy=False)

    cos_max_tilt = math.cos(math.radians(float(max_tilt_deg)))
    threshold = max(1.0, float(distance_threshold_mm))
    best_mask = None
    best_count = 0
    best_error = float("inf")

    for _ in range(max(1, int(ransac_iterations))):
        sample = candidates[rng.choice(candidates.shape[0], size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        normal_length = float(np.linalg.norm(normal))
        if normal_length < 1e-6:
            continue
        normal /= normal_length
        # 地面法向必须接近竖直轴；这会排除墙面等垂直平面。
        if abs(float(normal[1])) < cos_max_tilt:
            continue
        offset = -float(np.dot(normal, sample[0]))
        distances = np.abs(candidates @ normal + offset)
        inliers = distances <= threshold
        count = int(np.count_nonzero(inliers))
        error = float(np.mean(distances[inliers])) if count else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_mask = inliers
            best_count = count
            best_error = error

    required_count = max(
        int(min_inliers),
        int(math.ceil(float(min_inlier_ratio) * candidates.shape[0])),
    )
    if best_mask is None or best_count < required_count:
        return no_ground

    # 用全部 RANSAC 内点重新拟合，减少三点采样带来的平面抖动。
    inlier_points = candidates[best_mask]
    center = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    if abs(float(normal[1])) < cos_max_tilt:
        return no_ground

    distances = np.abs((xyz.astype(np.float64, copy=False) - center) @ normal)
    return distances <= threshold


# 将三维点云压缩为俯视密度图。图中横轴是相机左右方向，
# 纵轴是前后方向；越靠近图像上方，表示离相机越远。
def point_cloud_to_bev_density(
    xyz,
    x_min_mm,
    x_max_mm,
    z_min_mm,
    z_max_mm,
    y_min_mm,
    y_max_mm,
    resolution_mm,
    ground_filter_enabled=False,
    ground_distance_threshold_mm=40.0,
    ground_max_tilt_deg=25.0,
    ground_ransac_iterations=40,
    ground_ransac_sample_size=5000,
    ground_min_inliers=150,
    ground_min_inlier_ratio=0.08,
    ground_candidate_min_y_mm=100.0,
):
    # 删除 NaN、无穷值和相机驱动中表示无效深度的全零点。
    valid = np.isfinite(xyz).all(axis=1)
    valid &= ~np.all(xyz == 0, axis=1)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # 只保留机器人附近感兴趣的三维区域，减少地面、天花板和远处噪声。
    valid &= x >= x_min_mm
    valid &= x <= x_max_mm
    valid &= z >= z_min_mm
    valid &= z <= z_max_mm
    valid &= y >= y_min_mm
    valid &= y <= y_max_mm

    roi_xyz = xyz[valid]
    if bool(ground_filter_enabled) and roi_xyz.shape[0] > 0:
        is_ground = ground_plane_mask(
            roi_xyz,
            ground_distance_threshold_mm,
            ground_max_tilt_deg,
            ground_ransac_iterations,
            ground_ransac_sample_size,
            ground_min_inliers,
            ground_min_inlier_ratio,
            ground_candidate_min_y_mm,
        )
        roi_xyz = roi_xyz[~is_ground]

    x = roi_xyz[:, 0]
    z = roi_xyz[:, 2]

    width = int(np.ceil((x_max_mm - x_min_mm) / resolution_mm)) + 1
    height = int(np.ceil((z_max_mm - z_min_mm) / resolution_mm)) + 1
    density = np.zeros((height, width), dtype=np.float32)

    if x.size > 0:
        # 公制坐标投影到像素；v 轴取反，使前方显示在 BEV 上方。
        u = ((x - x_min_mm) / resolution_mm).astype(np.int32)
        v = (height - 1 - ((z - z_min_mm) / resolution_mm)).astype(np.int32)
        np.add.at(density, (v, u), 1.0)

    # 限制极密区域的影响，并用轻微模糊增强相邻帧配准的连续性。
    density = np.clip(density, 0, 30) / 30.0
    return cv2.GaussianBlur(density, (5, 5), 0), int(x.size)


# 相位相关只估计平移，用作后续 ECC 优化的快速初值。
def estimate_translation_mm(previous_bev, current_bev, resolution_mm):
    shift_px, response = cv2.phaseCorrelate(
        previous_bev.astype(np.float32),
        current_bev.astype(np.float32),
    )
    shift_u_px, shift_v_px = shift_px
    dx_mm = shift_u_px * resolution_mm
    dz_mm = -shift_v_px * resolution_mm
    return dx_mm, dz_mm, response

# 对比上一帧和当前帧
def estimate_motion_mm(
    previous_bev,
    current_bev,
    x_min_mm,
    z_min_mm,
    resolution_mm,
    ecc_iterations,
    ecc_epsilon,
):
    # 先用相位相关得到粗平移，避免 ECC 从单位矩阵开始时陷入错误局部最优。
    initial_dx_mm, initial_dz_mm, initial_response = estimate_translation_mm(
        previous_bev,
        current_bev,
        resolution_mm,
    )
    warp = np.array(
        [
            [1.0, 0.0, initial_dx_mm / resolution_mm],
            [0.0, 1.0, -initial_dz_mm / resolution_mm],
        ],
        dtype=np.float32,
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(ecc_iterations),
        float(ecc_epsilon),
    )

    previous_float = previous_bev.astype(np.float32)
    current_float = current_bev.astype(np.float32)
    # ECC 只在当前帧存在点云结构的区域内计算，降低大面积空白的干扰。
    current_mask = (current_bev > 1e-4).astype(np.uint8) * 255
    current_mask = cv2.dilate(current_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)

    try:
        # MOTION_EUCLIDEAN 同时估计二维平移和旋转，不允许缩放或形变。
        response, warp = cv2.findTransformECC(
            previous_float,
            current_float,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            current_mask,
            5,
        )
    except cv2.error:
        # ECC 失败时退回相位相关结果，保证节点继续输出而不是崩溃。
        return initial_dx_mm, initial_dz_mm, 0.0, initial_response, warp

    # BEV 并不包含相机原点（z_min 通常大于 0），因此先求出相机原点
    # 在图像中的虚拟像素位置，再通过 warp 计算真实的相机运动。
    height = previous_bev.shape[0]
    origin_u = -float(x_min_mm) / resolution_mm
    origin_v = (height - 1) + float(z_min_mm) / resolution_mm
    origin_u_warped = float(warp[0, 0]) * origin_u + float(warp[0, 1]) * origin_v + float(warp[0, 2])
    origin_v_warped = float(warp[1, 0]) * origin_u + float(warp[1, 1]) * origin_v + float(warp[1, 2])

    yaw_step_rad = math.atan2(float(warp[0, 1]), float(warp[0, 0]))
    dx_mm = (origin_u_warped - origin_u) * resolution_mm
    dz_mm = -(origin_v_warped - origin_v) * resolution_mm
    return dx_mm, dz_mm, yaw_step_rad, float(response), warp


def normalize_angle(angle_rad):
    # 将累计角度约束到 [-pi, pi]，避免数值无限增长。
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class BevOdometryNode(Node):
    """从深度点云估计平面里程计，并发布 Odometry、TF 和调试信息."""

    def __init__(self):
        super().__init__("orbbec_bev_odometry")

        # ROS 坐标系和输入点云配置。
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("pointcloud_topic", "/camera/depth/points")
        self.declare_parameter("pointcloud_scale_mm", 1000.0)
        self.declare_parameter("pointcloud_lateral_axis_index", 0)
        self.declare_parameter("pointcloud_vertical_axis_index", 1)
        self.declare_parameter("pointcloud_forward_axis_index", 2)
        self.declare_parameter("pointcloud_lateral_sign", 1.0)
        self.declare_parameter("pointcloud_vertical_sign", 1.0)
        self.declare_parameter("pointcloud_forward_sign", 1.0)
        # 点云裁剪范围和 BEV 分辨率。
        self.declare_parameter("x_min_mm", -2500.0)
        self.declare_parameter("x_max_mm", 2500.0)
        self.declare_parameter("z_min_mm", 200.0)
        self.declare_parameter("z_max_mm", 6000.0)
        self.declare_parameter("y_min_mm", -1200.0)
        self.declare_parameter("y_max_mm", 800.0)
        self.declare_parameter("resolution_mm", 25.0)
        # 先拟合并删除地面平面，避免地板在 OccupancyGrid 中显示为黑色障碍物。
        self.declare_parameter("ground_filter_enabled", False)
        self.declare_parameter("ground_distance_threshold_mm", 40.0)
        self.declare_parameter("ground_max_tilt_deg", 25.0)
        self.declare_parameter("ground_ransac_iterations", 40)
        self.declare_parameter("ground_ransac_sample_size", 5000)
        self.declare_parameter("ground_min_inliers", 150)
        self.declare_parameter("ground_min_inlier_ratio", 0.08)
        self.declare_parameter("ground_candidate_min_y_mm", 100.0)
        # 帧间配准的接受条件与处理频率。
        self.declare_parameter("min_confidence", 0.04)
        self.declare_parameter("max_step_mm", 500.0)
        self.declare_parameter("min_motion_mm", 2.0)
        self.declare_parameter("odom_interval_sec", 0.2)
        # 将图像配准方向转换为 ROS base_link 运动方向。
        self.declare_parameter("ros_forward_sign", -1.0)
        self.declare_parameter("ros_lateral_sign", -1.0)
        self.declare_parameter("ros_yaw_sign", -1.0)
        self.declare_parameter("yaw_gain", 1.165)
        self.declare_parameter("ecc_iterations", 50)
        self.declare_parameter("ecc_epsilon", 1e-5)
        # ``learned`` replaces ECC while leaving map fusion and ROS outputs intact.
        self.declare_parameter("odometry_backend", "ecc")
        self.declare_parameter("learned_checkpoint", "")
        self.declare_parameter("learned_device", "auto")
        self.declare_parameter("learned_min_confidence", 0.0)
        # xyz has already been remapped to [lateral, vertical, forward].
        self.declare_parameter("learned_lateral_to_left_sign", -1.0)
        self.declare_parameter("learned_vertical_to_up_sign", -1.0)
        self.declare_parameter("learned_forward_sign", 1.0)
        self.declare_parameter("learned_camera_forward_m", 0.0)
        self.declare_parameter("learned_camera_left_m", 0.0)
        self.declare_parameter("learned_camera_height_m", 0.0)
        self.declare_parameter("learned_camera_roll_rad", 0.0)
        self.declare_parameter("learned_camera_pitch_rad", 0.0)
        self.declare_parameter("learned_camera_yaw_rad", 0.0)
        # 移动时保留小转角；近静止时累积小 yaw，越过阈值后统一释放。
        self.declare_parameter("moving_yaw_deadband_deg", 0.25)
        self.declare_parameter("stationary_translation_deadband_mm", 20.0)
        self.declare_parameter("stationary_yaw_input_deadband_deg", 0.10)
        self.declare_parameter("stationary_yaw_release_deg", 1.0)
        self.declare_parameter("stationary_yaw_accumulation_enabled", True)
        self.declare_parameter("turn_translation_deadband_mm", 120.0)
        self.declare_parameter("turn_translation_min_yaw_deg", 2.0)
        # 当前简化全局地图只处理平移，偏航过大时暂停地图匹配与融合。
        self.declare_parameter("frame_to_map_max_abs_yaw_deg", 5.0)
        # 差速车不能侧向平移，因此可将车体横向速度约束为 0。
        self.declare_parameter("nonholonomic_constraint", True)
        self.declare_parameter("global_map_enabled", False)
        self.declare_parameter("frame_to_map_correction_enabled", False)
        # 固定全局 BEV 地图和局部模板搜索参数。
        self.declare_parameter("global_map_size_m", 20.0)
        self.declare_parameter("frame_to_map_search_radius_m", 0.5)
        self.declare_parameter("frame_to_map_min_score", 0.4)
        self.declare_parameter("frame_to_map_warmup_frames", 10)

        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        # 对外发布最终里程计、BEV 地图、累计路程和实验调试信息。
        self.odom_pub = self.create_publisher(Odometry, "bev_odom", 10)
        self.current_bev_pub = self.create_publisher(
            OccupancyGrid, "bev_odometry/current_bev", 1
        )
        self.global_map_pub = self.create_publisher(
            OccupancyGrid, "bev_odometry/global_map", 1
        )
        self.distance_pub = self.create_publisher(Float64, "bev_odometry/distance_m", 10)
        self.debug_pub = self.create_publisher(String, "bev_odometry/debug", 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # previous_bev 用于 frame-to-frame 粗估计；global_bev_map 用于漂移修正。
        self.previous_bev = None
        self.previous_learned_bev = None
        self.learned_runtime = None
        self.global_bev_map = None
        self.frame_to_map_score = 0.0
        self.frame_to_map_corrected = False
        self.coarse_forward_m = 0.0
        self.coarse_lateral_m = 0.0
        self.map_correction_forward_m = 0.0
        self.map_correction_lateral_m = 0.0
        self.total_forward_m = 0.0
        self.total_lateral_m = 0.0
        self.total_yaw_rad = 0.0
        self.pending_stationary_yaw_rad = 0.0
        self.total_distance_m = 0.0
        self.last_odom_time = None

        self.odometry_backend = str(
            self.get_parameter("odometry_backend").value
        ).lower()
        if self.odometry_backend not in ("ecc", "learned"):
            raise ValueError("odometry_backend must be 'ecc' or 'learned'")
        if self.odometry_backend == "learned":
            from orbbec_bev_odometry.learned_bev_runtime import (
                LearnedBEVRuntime,
            )

            checkpoint = str(self.get_parameter("learned_checkpoint").value)
            if not checkpoint:
                raise ValueError(
                    "learned_checkpoint is required for odometry_backend=learned"
                )
            self.learned_runtime = LearnedBEVRuntime(
                checkpoint,
                device=str(self.get_parameter("learned_device").value),
            )

        topic = str(self.get_parameter("pointcloud_topic").value)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            topic,
            self.process_pointcloud2,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Orbbec BEV odometry ({self.odometry_backend}) subscribed to "
            f"{topic}. Publishing /bev_odom."
        )

    def process_pointcloud2(self, msg):
        # ROS 回调只负责解码点云和时间戳，核心算法统一放在 process_xyz 中。
        xyz = extract_xyz_from_pointcloud2(
            msg,
            float(self.get_parameter("pointcloud_scale_mm").value),
            int(self.get_parameter("pointcloud_lateral_axis_index").value),
            int(self.get_parameter("pointcloud_vertical_axis_index").value),
            int(self.get_parameter("pointcloud_forward_axis_index").value),
            float(self.get_parameter("pointcloud_lateral_sign").value),
            float(self.get_parameter("pointcloud_vertical_sign").value),
            float(self.get_parameter("pointcloud_forward_sign").value),
        )
        stamp = msg.header.stamp
        now = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if now <= 0.0:
            now = time.time()
        self.process_xyz(xyz, now)

    def process_xyz(self, xyz, now):
        # 点云通常约 30 Hz；按 odom_interval 降采样，控制 ECC 的计算负载。
        odom_interval = float(self.get_parameter("odom_interval_sec").value)
        if self.last_odom_time is None:
            dt = odom_interval
        else:
            if now - self.last_odom_time < odom_interval:
                return
            dt = max(now - self.last_odom_time, 1e-6)
        self.last_odom_time = now

        # 第一步：把当前三维点云转换成当前相机坐标下的 BEV。
        bev, point_count = point_cloud_to_bev_density(
            xyz,
            float(self.get_parameter("x_min_mm").value),
            float(self.get_parameter("x_max_mm").value),
            float(self.get_parameter("z_min_mm").value),
            float(self.get_parameter("z_max_mm").value),
            float(self.get_parameter("y_min_mm").value),
            float(self.get_parameter("y_max_mm").value),
            float(self.get_parameter("resolution_mm").value),
            bool(self.get_parameter("ground_filter_enabled").value),
            float(self.get_parameter("ground_distance_threshold_mm").value),
            float(self.get_parameter("ground_max_tilt_deg").value),
            int(self.get_parameter("ground_ransac_iterations").value),
            int(self.get_parameter("ground_ransac_sample_size").value),
            int(self.get_parameter("ground_min_inliers").value),
            float(self.get_parameter("ground_min_inlier_ratio").value),
            float(self.get_parameter("ground_candidate_min_y_mm").value),
        )
        learned_bev = None
        if self.odometry_backend == "learned":
            robot_points_m = np.column_stack(
                (
                    xyz[:, 2]
                    * float(self.get_parameter("learned_forward_sign").value),
                    xyz[:, 0]
                    * float(
                        self.get_parameter(
                            "learned_lateral_to_left_sign"
                        ).value
                    ),
                    xyz[:, 1]
                    * float(
                        self.get_parameter(
                            "learned_vertical_to_up_sign"
                        ).value
                    ),
                )
            ).astype(np.float32) * 0.001
            roll = float(self.get_parameter("learned_camera_roll_rad").value)
            pitch = float(
                self.get_parameter("learned_camera_pitch_rad").value
            )
            yaw = float(self.get_parameter("learned_camera_yaw_rad").value)
            cos_r, sin_r = math.cos(roll), math.sin(roll)
            cos_p, sin_p = math.cos(pitch), math.sin(pitch)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            rotation_x = np.array(
                [[1.0, 0.0, 0.0], [0.0, cos_r, -sin_r],
                 [0.0, sin_r, cos_r]],
                dtype=np.float32,
            )
            rotation_y = np.array(
                [[cos_p, 0.0, sin_p], [0.0, 1.0, 0.0],
                 [-sin_p, 0.0, cos_p]],
                dtype=np.float32,
            )
            rotation_z = np.array(
                [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            robot_points_m = robot_points_m @ (
                rotation_z @ rotation_y @ rotation_x
            ).T
            robot_points_m += np.array(
                [
                    float(
                        self.get_parameter("learned_camera_forward_m").value
                    ),
                    float(
                        self.get_parameter("learned_camera_left_m").value
                    ),
                    float(
                        self.get_parameter("learned_camera_height_m").value
                    ),
                ],
                dtype=np.float32,
            )
            learned_bev = self.learned_runtime.make_bev(robot_points_m)

        dx_mm = 0.0
        dz_mm = 0.0
        yaw_step_rad = 0.0
        raw_ros_yaw_step_rad = 0.0
        near_stationary = False
        response = 0.0
        accepted = False

        # 第二步：current BEV vs previous BEV，得到增量运动的粗估计。
        if self.previous_bev is not None:
            if self.odometry_backend == "learned":
                learned_pose, response = self.learned_runtime.estimate(
                    self.previous_learned_bev, learned_bev
                )
                ros_forward_sign = float(
                    self.get_parameter("ros_forward_sign").value
                )
                ros_lateral_sign = float(
                    self.get_parameter("ros_lateral_sign").value
                )
                dz_mm = learned_pose[0] * 1000.0 / ros_forward_sign
                dx_mm = learned_pose[1] * 1000.0 / ros_lateral_sign
                yaw_step_rad = float(learned_pose[2])
                raw_ros_yaw_step_rad = yaw_step_rad
            else:
                dx_mm, dz_mm, raw_yaw_step_rad, response, _ = estimate_motion_mm(
                    self.previous_bev,
                    bev,
                    float(self.get_parameter("x_min_mm").value),
                    float(self.get_parameter("z_min_mm").value),
                    float(self.get_parameter("resolution_mm").value),
                    int(self.get_parameter("ecc_iterations").value),
                    float(self.get_parameter("ecc_epsilon").value),
                )
                yaw_step_rad = (
                    float(self.get_parameter("ros_yaw_sign").value)
                    * raw_yaw_step_rad
                )
                raw_ros_yaw_step_rad = yaw_step_rad
                yaw_step_rad *= float(self.get_parameter("yaw_gain").value)

            # 消除毫米级抖动，并根据运动类型选择不同的 yaw 死区。
            min_motion_mm = float(self.get_parameter("min_motion_mm").value)
            if abs(dx_mm) < min_motion_mm:
                dx_mm = 0.0
            if abs(dz_mm) < min_motion_mm:
                dz_mm = 0.0
            step_mm = float(np.hypot(dx_mm, dz_mm))
            stationary_translation_deadband_mm = float(
                self.get_parameter("stationary_translation_deadband_mm").value
            )
            near_stationary = step_mm < stationary_translation_deadband_mm
            # 近静止时平移视为配准抖动；yaw 交给有符号累积器判断。
            if near_stationary:
                dx_mm = 0.0
                dz_mm = 0.0
                step_mm = 0.0

            confidence_parameter = (
                "learned_min_confidence"
                if self.odometry_backend == "learned"
                else "min_confidence"
            )
            motion_is_credible = (
                response >= float(self.get_parameter(confidence_parameter).value)
                and step_mm <= float(self.get_parameter("max_step_mm").value)
            )
            if motion_is_credible and (
                not near_stationary
                or bool(
                    self.get_parameter(
                        "stationary_yaw_accumulation_enabled"
                    ).value
                )
            ):
                yaw_step_rad, self.pending_stationary_yaw_rad = gate_yaw_motion(
                    yaw_step_rad,
                    self.pending_stationary_yaw_rad,
                    step_mm,
                    stationary_translation_deadband_mm,
                    float(self.get_parameter("moving_yaw_deadband_deg").value),
                    float(
                        self.get_parameter(
                            "stationary_yaw_input_deadband_deg"
                        ).value
                    ),
                    float(self.get_parameter("stationary_yaw_release_deg").value),
                )
            else:
                yaw_step_rad = 0.0
                if near_stationary:
                    self.pending_stationary_yaw_rad = 0.0
            turn_yaw_deg = float(self.get_parameter("turn_translation_min_yaw_deg").value)
            # 原地转弯时 ECC 可能制造小平移；显著转角下将其清零。
            if (
                abs(math.degrees(yaw_step_rad)) >= turn_yaw_deg
                and step_mm < float(self.get_parameter("turn_translation_deadband_mm").value)
            ):
                dx_mm = 0.0
                dz_mm = 0.0
                step_mm = 0.0
            # 置信度不足或单帧跳变过大时，本帧运动不进入累计位姿。
            if (
                motion_is_credible
            ):
                forward_m = float(self.get_parameter("ros_forward_sign").value) * dz_mm / 1000.0
                lateral_m = float(self.get_parameter("ros_lateral_sign").value) * dx_mm / 1000.0
                if bool(self.get_parameter("nonholonomic_constraint").value):
                    # 非完整约束只禁止车体侧滑；转弯仍会通过 yaw 产生世界坐标 y。
                    lateral_m = 0.0
                # 用半步 yaw 积分平移，比直接使用帧首或帧尾朝向更接近圆弧运动。
                mid_yaw_rad = self.total_yaw_rad + yaw_step_rad * 0.5
                self.total_forward_m += forward_m * math.cos(mid_yaw_rad) - lateral_m * math.sin(mid_yaw_rad)
                self.total_lateral_m += forward_m * math.sin(mid_yaw_rad) + lateral_m * math.cos(mid_yaw_rad)
                self.total_yaw_rad = normalize_angle(self.total_yaw_rad + yaw_step_rad)
                self.total_distance_m += float(np.hypot(forward_m, lateral_m))
                accepted = True

        # 第三步：将浮点密度图转为全局地图使用的 uint8 图像。
        bev_uint8 = np.clip(bev * 255.0, 0.0, 255.0).astype(np.uint8)
        global_map_enabled = bool(self.get_parameter("global_map_enabled").value)
        if global_map_enabled and self.global_bev_map is None:
            # 全局地图只创建一次，分辨率必须与当前 BEV 完全一致。
            resolution_m = float(self.get_parameter("resolution_mm").value) / 1000.0
            self.global_bev_map = GlobalBEVMap(
                float(self.get_parameter("global_map_size_m").value),
                resolution_m,
                float(self.get_parameter("x_min_mm").value) / 1000.0,
                float(self.get_parameter("z_min_mm").value) / 1000.0,
                float(self.get_parameter("frame_to_map_search_radius_m").value),
                float(self.get_parameter("frame_to_map_min_score").value),
                int(self.get_parameter("frame_to_map_warmup_frames").value),
            )

        # 保存地图修正前的粗位姿，便于 rosbag 中区分误差来自哪一阶段。
        self.coarse_forward_m = self.total_forward_m
        self.coarse_lateral_m = self.total_lateral_m
        max_map_yaw_deg = float(self.get_parameter("frame_to_map_max_abs_yaw_deg").value)
        map_yaw_compatible = abs(math.degrees(self.total_yaw_rad)) <= max_map_yaw_deg
        correction_enabled = bool(
            self.get_parameter("frame_to_map_correction_enabled").value
        )
        corrected_lateral = self.total_lateral_m
        corrected_forward = self.total_forward_m
        self.frame_to_map_score = 0.0
        # 地图可独立绘制；只有显式开启时才允许它反向修改里程计。
        if global_map_enabled and self.global_bev_map is not None:
            corrected_lateral, corrected_forward, self.frame_to_map_score = (
                self.global_bev_map.update(
                    bev_uint8,
                    self.total_lateral_m,
                    self.total_forward_m,
                    yaw_rad=self.total_yaw_rad,
                    enable_correction=correction_enabled and map_yaw_compatible,
                    lateral_sign=float(
                        self.get_parameter("ros_lateral_sign").value
                    ),
                )
            )
        min_map_score = float(self.get_parameter("frame_to_map_min_score").value)
        self.frame_to_map_corrected = (
            correction_enabled
            and map_yaw_compatible
            and global_map_enabled
            and self.global_bev_map is not None
            and self.global_bev_map.frame_count > self.global_bev_map.warmup_frames
            and self.frame_to_map_score >= min_map_score
        )
        self.map_correction_forward_m = corrected_forward - self.coarse_forward_m
        self.map_correction_lateral_m = corrected_lateral - self.coarse_lateral_m
        self.total_lateral_m = corrected_lateral
        self.total_forward_m = corrected_forward
        # 当前帧始终成为下一帧粗匹配的参考图。
        self.previous_bev = bev.copy()
        if learned_bev is not None:
            self.previous_learned_bev = learned_bev.copy()
        self.publish_odometry(
            dx_mm,
            dz_mm,
            yaw_step_rad,
            response,
            point_count,
            accepted,
            dt,
            raw_ros_yaw_step_rad,
            near_stationary,
        )
        self.publish_bev_maps(bev_uint8)

    @staticmethod
    def density_to_occupancy(density):
        """将 0..255 的 BEV 密度转换为 OccupancyGrid 的 0..100."""
        scaled = np.rint(density.astype(np.float32) * (100.0 / 255.0))
        return np.clip(scaled, 0, 100).astype(np.int8)

    def publish_bev_maps(self, current_bev):
        """发布当前局部 BEV 和累计全局 BEV，供 RViz 的 Map 显示."""
        stamp = self.get_clock().now().to_msg()
        resolution_m = float(self.get_parameter("resolution_mm").value) / 1000.0

        # 原始 BEV 的列为相机右向、行为后向。转置并翻转两个轴后，
        # OccupancyGrid 的列对应 base_link 前向 x，行对应左向 y。
        current_grid = np.flip(current_bev.T, axis=(0, 1))
        current_msg = OccupancyGrid()
        current_msg.header.stamp = stamp
        current_msg.header.frame_id = self.child_frame_id
        current_msg.info.map_load_time = stamp
        current_msg.info.resolution = resolution_m
        current_msg.info.width = int(current_grid.shape[1])
        current_msg.info.height = int(current_grid.shape[0])
        current_msg.info.origin.position.x = (
            float(self.get_parameter("z_min_mm").value) / 1000.0
        )
        # 使用实际网格最右列坐标，兼容范围不能被分辨率整除的情况。
        x_min_m = float(self.get_parameter("x_min_mm").value) / 1000.0
        grid_x_max_m = x_min_m + (current_bev.shape[1] - 1) * resolution_m
        current_msg.info.origin.position.y = -grid_x_max_m
        current_msg.info.origin.orientation.w = 1.0
        current_msg.data = self.density_to_occupancy(current_grid).ravel().tolist()
        self.current_bev_pub.publish(current_msg)

        if self.global_bev_map is None:
            return

        global_msg = self.global_bev_map.to_occupancy_grid(stamp, self.frame_id)
        self.global_map_pub.publish(global_msg)

    def publish_odometry(
        self,
        dx_mm,
        dz_mm,
        yaw_step_rad,
        response,
        point_count,
        accepted,
        dt,
        raw_ros_yaw_step_rad,
        near_stationary,
    ):
        # Pose 表示 odom -> base_link 的累计二维位姿。
        stamp = self.get_clock().now().to_msg()
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.pose.pose.position.x = self.total_forward_m
        msg.pose.pose.position.y = self.total_lateral_m
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(self.total_yaw_rad * 0.5)
        msg.pose.pose.orientation.w = math.cos(self.total_yaw_rad * 0.5)

        forward_step_m = float(self.get_parameter("ros_forward_sign").value) * dz_mm / 1000.0
        lateral_step_m = float(self.get_parameter("ros_lateral_sign").value) * dx_mm / 1000.0
        if bool(self.get_parameter("nonholonomic_constraint").value):
            lateral_step_m = 0.0
        msg.twist.twist.linear.x = forward_step_m / dt if accepted else 0.0
        msg.twist.twist.linear.y = lateral_step_m / dt if accepted else 0.0
        msg.twist.twist.angular.z = yaw_step_rad / dt if accepted else 0.0

        # 被拒绝的帧使用更大的协方差，方便后续 EKF 降低其权重。
        covariance = 0.02 if accepted else 0.5
        msg.pose.covariance[0] = covariance
        msg.pose.covariance[7] = covariance
        msg.pose.covariance[35] = math.radians(30.0) ** 2
        msg.twist.covariance[0] = covariance
        msg.twist.covariance[7] = covariance
        msg.twist.covariance[35] = math.radians(30.0) ** 2

        self.odom_pub.publish(msg)

        distance_msg = Float64()
        distance_msg.data = self.total_distance_m
        self.distance_pub.publish(distance_msg)

        # 发布与 Odometry Pose 一致的 TF，供 RViz、Nav2 等组件使用。
        if self.tf_broadcaster is not None:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = self.frame_id
            tf_msg.child_frame_id = self.child_frame_id
            tf_msg.transform.translation.x = msg.pose.pose.position.x
            tf_msg.transform.translation.y = msg.pose.pose.position.y
            tf_msg.transform.rotation.z = msg.pose.pose.orientation.z
            tf_msg.transform.rotation.w = msg.pose.pose.orientation.w
            self.tf_broadcaster.sendTransform(tf_msg)

        # 文本调试话题同时记录粗估计、地图修正量和最终位姿，便于 rosbag 分析。
        debug = String()
        debug.data = (
            f"backend={self.odometry_backend} accepted={accepted} "
            f"dx_mm={dx_mm:+.1f} dz_mm={dz_mm:+.1f} "
            f"x_m={self.total_forward_m:+.3f} y_m={self.total_lateral_m:+.3f} "
            f"yaw_deg={math.degrees(self.total_yaw_rad):+.1f} "
            f"yaw_step_deg={math.degrees(yaw_step_rad):+.1f} "
            f"raw_yaw_step_deg={math.degrees(raw_ros_yaw_step_rad):+.3f} "
            f"pending_yaw_deg={math.degrees(self.pending_stationary_yaw_rad):+.3f} "
            f"near_stationary={near_stationary} "
            f"distance_m={self.total_distance_m:.3f} "
            f"confidence={response:.3f} points={point_count} "
            f"coarse_x_m={self.coarse_forward_m:+.3f} "
            f"coarse_y_m={self.coarse_lateral_m:+.3f} "
            f"map_dx_m={self.map_correction_forward_m:+.3f} "
            f"map_dy_m={self.map_correction_lateral_m:+.3f} "
            f"map_score={self.frame_to_map_score:.3f} "
            f"map_corrected={self.frame_to_map_corrected} "
            f"map_frames={self.global_bev_map.frame_count if self.global_bev_map else 0}"
        )
        self.debug_pub.publish(debug)


def main(args=None):
    # 标准 ROS 2 节点生命周期：初始化、阻塞运行、退出时释放资源。
    rclpy.init(args=args)
    node = None
    try:
        node = BevOdometryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
