# 学习型多通道 BEV 里程计

这条前端用学习型 SE(2) 配准替换 `cv2.findTransformECC`，现有 ECC 和
3D GICP/ICP 继续作为实验基线。全局地图、Odometry、TF 和调试话题保持不变。

## 方法边界

运行时坐标统一为 ROS `base_link`：

- x：前向，单位 m；
- y：左向，单位 m；
- z：上向，单位 m；
- yaw：逆时针为正，单位 rad。

每帧深度先生成九通道 2.5D BEV：占据率、对数点密度、平均高度、最大高度、
高度方差和四个高度切片。共享 CNN 分别编码前后两帧；SE(2) 匹配器围绕机器人
原点旋转当前特征，并使用线性 FFT 相关显式搜索平移。细化头只预测一个相关网格
以内的残差。

默认 BEV 范围为前向 `[-1, 6] m`、左向 `[-3.5, 3.5] m`，分辨率 `0.05 m`。
机器人原点不在图像中心，模型的几何 warp 已显式处理这一点。

## 目录约定

OpenLORIS 解压目录中每条序列需要包含：

```text
office1-1/
  color.txt
  aligned_depth.txt
  groundtruth.txt
  color/*.png
  aligned_depth/*.png
```

DA3 预测必须保存为米制浮点 `.npy` 或 `.npz`，不能使用彩色深度图或 8-bit PNG。
预处理器依次搜索以下布局：

```text
DA3_ROOT/office1-1/color/<timestamp>.npy
DA3_ROOT/office1-1/<timestamp>.npy
```

`.npz` 优先读取名为 `depth` 的数组。输入形状必须能压缩成 `[H, W]`。

## 1. 预计算 BEV 和训练对

先用 RealSense 对齐深度验证模型结构的上限：

```bash
python -m orbbec_bev_odometry.prepare_learned_bev \
  --dataset-root /data/openloris/extracted/office1 \
  --cache-root /data/openloris/bev_cache/aligned_depth \
  --depth-source aligned_depth
```

正式训练使用冻结 DA3 的输出，保证训练输入与部署一致：

```bash
python -m orbbec_bev_odometry.prepare_learned_bev \
  --dataset-root /data/openloris/extracted/office1 \
  --cache-root /data/openloris/bev_cache/da3 \
  --depth-source da3 \
  --da3-root /data/openloris/da3_metric
```

命令可断点续跑，默认不覆盖已经生成的 `.npz`。默认按序列划分：

- train：office1-1 到 office1-4，帧间隔 3/6/9/12；
- validation：office1-5，帧间隔 6；
- test：office1-6 和 office1-7，帧间隔 6。

输出的 `manifests/*.jsonl` 保存配对路径和由 `groundtruth.txt` 插值得到的
`[forward, left, yaw]` 标签。没有使用 `odom.txt` 作为监督。

## 2. 训练

Linux/NVIDIA 环境安装 PyTorch 和 Pillow 后执行：

```bash
python -m orbbec_bev_odometry.train_learned_bev \
  --train-manifest /data/openloris/bev_cache/da3/manifests/train.jsonl \
  --validation-manifest /data/openloris/bev_cache/da3/manifests/validation.jsonl \
  --output-dir /home/oit/research/checkpoints/learned_bev \
  --device cuda \
  --batch-size 2 \
  --epochs 40
```

`best.pt` 按验证集平移 RMSE 保存，`last.pt` 用于断点续训，`history.jsonl`
记录每轮损失、平移误差和旋转误差。显存不足时先减小 batch size，不要改变
训练/测试序列划分。静止样本默认按 2 倍权重采样，同时加入静止假运动、预测
步长膨胀和置信度校准损失，专门约束 DA3 帧间尺度闪烁造成的累计路程虚增。

## 3. 独立测试

```bash
python -m orbbec_bev_odometry.evaluate_learned_bev \
  --checkpoint /home/oit/research/checkpoints/learned_bev/best.pt \
  --manifest /data/openloris/bev_cache/da3/manifests/test.jsonl \
  --output-dir /home/oit/research/results/learned_bev_test \
  --device cuda
```

结果包含 `predictions.csv` 和 `summary.json`。报告 pairwise RPE、按固定步长链接
后的 ATE、终点位置残差、终点 yaw 残差、累计路程比例和超过 10 cm 的假步数。
必须与 BEV-ECC、真实深度 GICP、
DA3 点云 GICP 使用相同序列和帧间隔比较。

## 4. 替换 ROS 2 中的 ECC

构建工作区后加载训练检查点：

```bash
colcon build --packages-select orbbec_bev_odometry --symlink-install
source install/setup.bash

ros2 launch orbbec_bev_odometry bev_odometry.launch.py \
  odometry_backend:=learned \
  learned_checkpoint:=/home/oit/research/checkpoints/learned_bev/best.pt \
  learned_device:=cuda \
  camera_x:=0.0 camera_y:=0.0 camera_z:=0.80
```

`camera_x/y/z` 和 `camera_roll/pitch/yaw` 必须填写相机光心在 `base_link` 中的
实际安装位姿，它们同时用于静态 TF 和学习型 BEV 点云变换。
上面的 `0.80 m` 只是命令示例，不能代替实际测量值。
当前 PointCloud2 轴默认按光学坐标 `右/下/前` 转为 `前/左/上`。如果驱动发布
的轴不同，需要使用已有 pointcloud 轴参数及 learned 符号参数校正。

运行节点订阅的是 PointCloud2。在线 DA3 前端需要把米制预测深度结合对应
CameraInfo 反投影并发布到 `pointcloud_topic`；离线训练不依赖这个 ROS 桥接层。

## 实验停止条件

先在真实对齐深度上训练。如果学习模型不能显著超过 BEV-ECC，应先检查标签、
坐标和相关搜索，而不是加入 DA3 噪声。真实深度实验通过后再使用 DA3 缓存微调。
GICP 是强基线；学习方法应同时报告精度、失败率、耗时和内存，而不能只比较
单次闭环终点残差。
