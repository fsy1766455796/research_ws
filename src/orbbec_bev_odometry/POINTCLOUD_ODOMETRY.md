# 平面约束点云里程计基线

这个实现与现有 `bev_odometry_node` 完全独立，旧节点及其参数未修改。

处理链分为：

- `cloud_preprocessor.py`：范围裁剪、地面高度剔除、体素降采样、可选统计离群过滤；
- `planar_icp.py`：鲁棒 point-to-plane ICP，优化变量严格限制为 x、y、yaw；
- `pcl_gicp.cpp` / `gicp_initializer.py`：PCL-GICP 相邻帧连续初值；
- `se2.py`：SE(2) 变换；
- `local_submap.py`：滚动局部子图；
- `pointcloud_odom_core.py`：ROS 无关的里程计状态机；
- `pointcloud_odometry_node.py`：ROS 2 输入、odom/TF/debug 输出；
- `offline_ab.py`：MCAP 离线 A/B 评测。

## 运行

```bash
colcon build --packages-select orbbec_bev_odometry --symlink-install
source install/setup.bash
ros2 launch orbbec_bev_odometry pointcloud_odometry.launch.py
```

离线评测：

```bash
ros2 run orbbec_bev_odometry pointcloud_offline_ab odom_loop_test \
  --output offline_results/pointcloud_icp_ab.json
```

输出话题为 `/pointcloud_odom` 和 `/pointcloud_odometry/debug`，不会覆盖 `/bev_odom`。

## 当前实测状态

`odom_loop_test` 没有外部 ground truth，只能假定录制结束时回到起点计算闭环残差。
2026-07-14 最终参数在完整 `odom_loop_test` 上的结果如下：

| 方法 | yaw 闭环残差 | 位置闭环残差 | 开头静止漂移 | 平均耗时 |
|---|---:|---:|---:|---:|
| 录制的 BEV-ECC | 56.10° | 1.00 m | 0 | bag 中无耗时 |
| scan-to-local-submap GICP/ICP | 0.0026° | 0.299 m | 0 | 222.6 ms |

yaw 闭环误差已经低于 5°，且未使用回环或全局地图修正。位置闭环残差仍为 0.299 m，
应继续作为里程计基线优化项；当前阶段不接 OccupancyGrid。
