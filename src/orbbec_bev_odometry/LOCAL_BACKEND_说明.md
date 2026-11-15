# 局部地图后端第一版

本程序冻结已有神经网络，读取前端输出的相对位姿和对应 BEV 缓存，进行局部地图匹配与滑动窗口位姿优化。不重新训练模型，也不人为要求首尾位姿相同。

## 算法

1. 按序列和时间戳连接帧对。重叠时间区间不重复积分，无法连接的片段分开运行并重置起点。
2. 网络预测提供初始位姿，最近 8 个已处理观测帧组成滚动局部地图。第一版使用观测帧窗口，尚未实现距离触发的关键帧选择。
3. 从 BEV 占据通道取格点，使用每个序列 `metadata.json` 中的分辨率和范围转换为米制坐标。用项目已有的二维点到点 ICP 对当前观测与历史局部地图配准。
4. 检查重叠率、配准残差、空间分布退化和最大修正量。匹配失败时保留该帧的网络预测。当前帧不会出现在与自己匹配的地图中。
5. 将通过检查的匹配作为局部地图锚点到当前帧的相对位姿边，与原网络增量一起优化窗口内位姿。固定窗口首节点，使用稳健非线性最小二乘。随后按更新位姿重新构建地图。

后端核心函数 `run_backend(increments, point_loader, cfg)` 不接收真值。命令行评估器仅在优化结束后读取真值标签计算指标。

## 在 Windows 上运行

环境依赖：NumPy、SciPy；使用 `--plot` 时还需要 Matplotlib。当前电脑虚拟环境已经具备这些依赖。

先使用验证集检查，输出目录必须是新目录：

```powershell
cd C:\Users\17664\research_ws
$env:PYTHONPATH="$PWD\src\orbbec_bev_odometry"
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe -m orbbec_bev_odometry.local_map_backend `
  --predictions D:\dataset\openloris_bev_results\aligned_depth_yaw4_validation\predictions.csv `
  --manifest D:\dataset\openloris_bev_cache\aligned_depth\manifests\validation.jsonl `
  --output-dir D:\dataset\openloris_bev_results\local_backend_validation_my_run `
  --window 8 --plot
```

不依赖激活虚拟环境或 ROS。改用 `--mode frontend` 可以验证关闭后端后的基准；前端与后端必须使用同一套时间链。DA3 输入需要同时替换预测文件和清单，不能将实测深度缓存用于 DA3 实验。

## 结果文件

- `报告.md`：中文结果对照表。
- `summary.json`：配置、输入路径、各片段统计和误差。
- `trajectories.csv`：时间戳、原始轨迹、窗口修正后的轨迹、真值增量积分参考轨迹。
- `matches_*.json`：逐帧匹配是否接受、残差及修正量，便于排查失败。
- `trajectory_*.png`：使用 `--plot` 时输出中文轨迹与误差曲线。

## 评估边界

这是实验原型，不保证优于前端，也不是自动闭环 SLAM。地图建立在估计位姿上，仍会漂移；重复结构、地面占据和动态物体可能导致错误匹配。空间分布检查只能筛除明显共线退化，不能证明匹配唯一。

当前输出的位置均方根误差采用起点对齐，参考为平面真值增量积分；不是对原始三维真值做全轨迹刚体对齐后的标准 ATE。终点误差是相对参考轨迹终点的误差，不是到起点的闭合距离。角度误差折返到 ±180°，不能表达多圈累计转角。

滑动窗口会修正过去的位姿，导出的后端轨迹是固定窗口延迟后的结果，不是零延迟输出。窗口首节点固定，旧地图约束离开窗口后删除，当前版本未做边缘化先验。地图边与里程计边相关，权重是实验设置，不能解释为严格独立观测的统计协方差。

先在验证集确定参数，测试集只作冻结设置后的对照，不根据测试结果反复调参。自采 rosbag 尚需导出相同格式的前端预测及 BEV，当前命令行不直接读取 rosbag。

## 首次运行结果（2026-10-09）

使用现有 A1 预测及默认 8 帧窗口。先在验证集检查实现，再以相同配置运行两个测试序列，未依据测试结果调参。下表对应按时间连接选出的连续片段，office1-6 为 179 个帧对，office1-7 为 190 个帧对。

| 深度输入 | 序列 | 位置均方根误差：前端 → 后端（米） | 终点角度误差：前端 → 后端（度） |
| --- | --- | --- | --- |
| 实测深度 | office1-6 | 0.912 → 0.859 | 67.21 → 67.23 |
| 实测深度 | office1-7 | 1.314 → 1.277 | 78.75 → 84.18 |
| DA3 深度 | office1-6 | 1.033 → 1.031 | 62.45 → 66.14 |
| DA3 深度 | office1-7 | 1.182 → 1.144 | 78.81 → 78.20 |

位置指标有小幅改善，但角度漂移没有稳定降低。尤其 DA3 的 office1-6 终点位置误差从 1.667 米增加到 1.740 米。因此当前仅作为可运行的研究基线，不能宣称已经解决累计误差。

本机结果目录：

- 验证集：`D:\dataset\openloris_bev_results\local_backend_v1_validation_final`
- 实测深度测试：`D:\dataset\openloris_bev_results\local_backend_v1_aligned_test`
- DA3 深度测试：`D:\dataset\openloris_bev_results\local_backend_v1_da3_test`

测试验证了变换方向、共线退化拒绝、重叠时间处理、匹配拒绝回退、窗口锚点和当前帧不参与自匹配。连同相关原有测试共 11 项通过；这些检查不代表算法精度已达标。
