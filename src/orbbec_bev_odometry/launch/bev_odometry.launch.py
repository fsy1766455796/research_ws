from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            # start_camera=true 时由本 launch 一并启动 Femto Bolt；若相机已启动则传 false。
            DeclareLaunchArgument("start_camera", default_value="true"),
            # 为 RViz 同时提供 RGB 着色点云；里程计算法仍使用更轻量的深度点云。
            DeclareLaunchArgument("enable_colored_point_cloud", default_value="true"),
            DeclareLaunchArgument("pointcloud_topic", default_value="/camera/depth/points"),
            DeclareLaunchArgument("pointcloud_scale_mm", default_value="1000.0"),
            # ecc 保留原始基线；learned 使用训练后的多通道 BEV 模型。
            DeclareLaunchArgument("odometry_backend", default_value="ecc"),
            DeclareLaunchArgument("learned_checkpoint", default_value=""),
            DeclareLaunchArgument("learned_device", default_value="auto"),
            # 连接里程计与相机 TF 树；若机器人 URDF 已发布该变换，可传 false 关闭。
            DeclareLaunchArgument("publish_camera_tf", default_value="true"),
            DeclareLaunchArgument("camera_parent_frame", default_value="base_link"),
            DeclareLaunchArgument("camera_frame", default_value="camera_link"),
            DeclareLaunchArgument("camera_x", default_value="0.0"),
            DeclareLaunchArgument("camera_y", default_value="0.0"),
            DeclareLaunchArgument("camera_z", default_value="0.0"),
            DeclareLaunchArgument("camera_roll", default_value="0.0"),
            DeclareLaunchArgument("camera_pitch", default_value="0.0"),
            DeclareLaunchArgument("camera_yaw", default_value="0.0"),
            # Orbbec 光学点云坐标为 x=横向、y=竖直、z=前向。
            DeclareLaunchArgument("pointcloud_lateral_axis_index", default_value="0"),
            DeclareLaunchArgument("pointcloud_vertical_axis_index", default_value="1"),
            DeclareLaunchArgument("pointcloud_forward_axis_index", default_value="2"),
            # 离线里程计调参阶段默认完全关闭全局地图及其位姿修正。
            DeclareLaunchArgument("global_map_enabled", default_value="false"),
            DeclareLaunchArgument(
                "frame_to_map_correction_enabled", default_value="false"
            ),
            DeclareLaunchArgument("moving_yaw_deadband_deg", default_value="0.25"),
            DeclareLaunchArgument("yaw_gain", default_value="1.165"),
            DeclareLaunchArgument(
                "stationary_translation_deadband_mm", default_value="20.0"
            ),
            DeclareLaunchArgument(
                "stationary_yaw_release_deg", default_value="1.0"
            ),
            DeclareLaunchArgument(
                "stationary_yaw_input_deadband_deg", default_value="0.10"
            ),
            DeclareLaunchArgument(
                "stationary_yaw_accumulation_enabled", default_value="true"
            ),
            # 相机驱动必须开启点云，否则里程计节点收不到 /camera/depth/points。
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("orbbec_camera"), "launch", "femto_bolt.launch.py"]
                    )
                ),
                condition=IfCondition(LaunchConfiguration("start_camera")),
                launch_arguments={
                    "enable_point_cloud": "true",
                    "enable_colored_point_cloud": LaunchConfiguration(
                        "enable_colored_point_cloud"
                    ),
                }.items(),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_link_to_camera_link",
                output="screen",
                condition=IfCondition(LaunchConfiguration("publish_camera_tf")),
                arguments=[
                    "--x",
                    LaunchConfiguration("camera_x"),
                    "--y",
                    LaunchConfiguration("camera_y"),
                    "--z",
                    LaunchConfiguration("camera_z"),
                    "--roll",
                    LaunchConfiguration("camera_roll"),
                    "--pitch",
                    LaunchConfiguration("camera_pitch"),
                    "--yaw",
                    LaunchConfiguration("camera_yaw"),
                    "--frame-id",
                    LaunchConfiguration("camera_parent_frame"),
                    "--child-frame-id",
                    LaunchConfiguration("camera_frame"),
                ],
            ),
            Node(
                package="orbbec_bev_odometry",
                executable="bev_odometry_node",
                name="orbbec_bev_odometry",
                output="screen",
                parameters=[
                    {
                        # 坐标系、输入话题和点云单位/轴映射。
                        "frame_id": "odom",
                        "child_frame_id": "base_link",
                        "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
                        "pointcloud_scale_mm": LaunchConfiguration("pointcloud_scale_mm"),
                        "pointcloud_lateral_axis_index": LaunchConfiguration("pointcloud_lateral_axis_index"),
                        "pointcloud_vertical_axis_index": LaunchConfiguration("pointcloud_vertical_axis_index"),
                        "pointcloud_forward_axis_index": LaunchConfiguration("pointcloud_forward_axis_index"),
                        "odometry_backend": LaunchConfiguration(
                            "odometry_backend"
                        ),
                        "learned_checkpoint": LaunchConfiguration(
                            "learned_checkpoint"
                        ),
                        "learned_device": LaunchConfiguration("learned_device"),
                        "learned_camera_forward_m": LaunchConfiguration(
                            "camera_x"
                        ),
                        "learned_camera_left_m": LaunchConfiguration("camera_y"),
                        "learned_camera_height_m": LaunchConfiguration(
                            "camera_z"
                        ),
                        "learned_camera_roll_rad": LaunchConfiguration(
                            "camera_roll"
                        ),
                        "learned_camera_pitch_rad": LaunchConfiguration(
                            "camera_pitch"
                        ),
                        "learned_camera_yaw_rad": LaunchConfiguration(
                            "camera_yaw"
                        ),
                        # frame-to-frame ECC 配准频率、置信度和异常步长门槛。
                        "odom_interval_sec": 0.2,
                        "min_confidence": 0.04,
                        "max_step_mm": 500.0,
                        "ecc_iterations": 50,
                        "ecc_epsilon": 1e-5,
                        # 去除地板平面，防止近处地面被当成黑色障碍物。
                        "ground_filter_enabled": True,
                        "ground_distance_threshold_mm": 40.0,
                        "ground_max_tilt_deg": 25.0,
                        "ground_ransac_iterations": 40,
                        "ground_ransac_sample_size": 5000,
                        "ground_min_inliers": 150,
                        "ground_min_inlier_ratio": 0.08,
                        "ground_candidate_min_y_mm": 100.0,
                        # 移动时保留小 yaw；近静止时有符号累积，达到阈值后释放。
                        "moving_yaw_deadband_deg": LaunchConfiguration(
                            "moving_yaw_deadband_deg"
                        ),
                        "yaw_gain": LaunchConfiguration("yaw_gain"),
                        "stationary_translation_deadband_mm": LaunchConfiguration(
                            "stationary_translation_deadband_mm"
                        ),
                        "stationary_yaw_release_deg": LaunchConfiguration(
                            "stationary_yaw_release_deg"
                        ),
                        "stationary_yaw_input_deadband_deg": LaunchConfiguration(
                            "stationary_yaw_input_deadband_deg"
                        ),
                        "stationary_yaw_accumulation_enabled": LaunchConfiguration(
                            "stationary_yaw_accumulation_enabled"
                        ),
                        "turn_translation_deadband_mm": 120.0,
                        "turn_translation_min_yaw_deg": 2.0,
                        # 地图独立绘制；反向校准默认关闭，避免污染里程计。
                        "global_map_enabled": LaunchConfiguration(
                            "global_map_enabled"
                        ),
                        "frame_to_map_correction_enabled": LaunchConfiguration(
                            "frame_to_map_correction_enabled"
                        ),
                        # 当前地图只修正平移；转角过大时暂停地图匹配。
                        "frame_to_map_max_abs_yaw_deg": 5.0,
                        # 差速车不允许车体侧滑。
                        "nonholonomic_constraint": True,
                        # 固定全局地图尺寸、局部搜索半径、接受分数和预热帧数。
                        "global_map_size_m": 20.0,
                        "frame_to_map_search_radius_m": 0.5,
                        "frame_to_map_min_score": 0.4,
                        "frame_to_map_warmup_frames": 10,
                    }
                ],
            )
        ]
    )
