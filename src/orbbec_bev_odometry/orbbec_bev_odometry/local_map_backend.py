"""离线局部地图匹配与滑动窗口位姿优化，不使用真值生成约束。"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .planar_icp import IcpConfig, PlanarPointToPlaneICP
from .se2 import components, inverse, matrix, transform_points, wrap_angle


@dataclass
class BackendConfig:
    window: int = 8
    max_points: int = 1500
    voxel_m: float = 0.06
    min_points: int = 40
    correspondence_m: float = 0.20
    min_overlap: float = 0.5
    max_rmse_m: float = 0.08
    max_condition: float = 1e5
    max_correction_m: float = 0.12
    max_correction_deg: float = 4.0
    odom_sigma_m: float = 0.03
    odom_sigma_deg: float = 2.0
    map_sigma_m: float = 0.05
    map_sigma_deg: float = 2.0
    strategy: str = 'fixed_history'

    def __post_init__(self):
        if self.strategy not in ('fixed_history', 'legacy'):
            raise ValueError('strategy must be fixed_history or legacy')


def pair_key(row):
    return (row['sequence'], round(float(row['timestamp_0']), 6),
            round(float(row['timestamp_1']), 6))


def select_chains(rows):
    """选取确定性的非重叠时间链，缺失连接时另起一段。"""
    grouped = {}
    for row in rows:
        key = pair_key(row)
        if key[2] <= key[1]:
            raise ValueError('帧对结束时间必须晚于开始时间')
        grouped.setdefault(key[0], []).append(row)
    chains = []
    for sequence, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: pair_key(r)[1:])
        chain, endpoint = [], None
        for row in ordered:
            _, start, end = pair_key(row)
            if endpoint is not None and start < endpoint:
                continue
            if endpoint is not None and start != endpoint:
                chains.append((sequence, chain))
                chain = []
            chain.append(row)
            endpoint = end
        if chain:
            chains.append((sequence, chain))
    return chains


def thin_points(points, voxel, limit):
    if not len(points):
        return points
    _, indices = np.unique(np.floor(points / voxel).astype(np.int64),
                           axis=0, return_index=True)
    points = points[np.sort(indices)]
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
    return points


def load_points(filename, cfg):
    path = Path(filename)
    metadata = json.loads((path.parent / 'metadata.json').read_text(encoding='utf-8'))
    geometry = metadata['bev_config']
    with np.load(path, allow_pickle=False) as archive:
        bev = archive['bev'].astype(np.float32)
    if bev.ndim != 3 or not np.isfinite(bev).all():
        raise ValueError(f'无效 BEV: {path}')
    # 使用占据格中心，不将归一化后的高度或密度误当作米制坐标。
    rows, cols = np.nonzero(bev[0] > 0.5)
    resolution = float(geometry['resolution_m'])
    expected = (int(np.ceil((geometry['forward_max_m'] - geometry['forward_min_m']) / resolution)),
                int(np.ceil((geometry['left_max_m'] - geometry['left_min_m']) / resolution)))
    if bev.shape[1:] != expected:
        raise ValueError(f'BEV 尺寸与元数据不符: {path}')
    points = np.column_stack((geometry['forward_min_m'] + (rows + 0.5) * resolution,
                              geometry['left_min_m'] + (cols + 0.5) * resolution))
    return thin_points(points, cfg.voxel_m, cfg.max_points)


def relative(a, b):
    return np.asarray(components(inverse(matrix(*a)) @ matrix(*b)))


def compose(a, b):
    return np.asarray(components(matrix(*a) @ matrix(*b)))


def match_map(source, target, initial, cfg):
    """源点已被初始位姿变换到地图坐标；这里只估计小范围修正。"""
    diagnostic = {'accepted': False, 'reason': 'too_few_points'}
    if min(len(source), len(target)) < cfg.min_points:
        return initial, diagnostic
    moved = transform_points(matrix(*initial), source)
    solver = PlanarPointToPlaneICP(IcpConfig(
        min_correspondences=cfg.min_points, max_correspondence=cfg.correspondence_m,
        # A left correction translates when rotating about a displaced robot.
        # Gate robot-origin displacement below, not this coordinate-dependent term.
        max_step_translation=(cfg.max_correction_m if cfg.strategy == 'legacy'
                              else np.inf),
        max_step_yaw=math.radians(cfg.max_correction_deg)))
    # BEV 占据格并非连续墙面采样，采用已有点到点 ICP，避免依赖不稳定的法线。
    result = solver.register_point_to_point(moved, target, initial=np.eye(3))
    candidate = result.transform @ matrix(*initial)
    distances, indices = cKDTree(target).query(transform_points(candidate, source))
    mask = distances < cfg.correspondence_m
    overlap = float(np.mean(mask))
    rmse = float(np.sqrt(np.mean(distances[mask] ** 2))) if np.any(mask) else None
    # 空间分布检查只能拒绝明显的共线退化，不能证明匹配在感知上唯一。
    condition = np.inf
    if np.count_nonzero(mask) >= cfg.min_points:
        eigenvalues = np.linalg.eigvalsh(np.cov(target[indices[mask]].T))
        if eigenvalues[0] > 1e-10:
            condition = eigenvalues[-1] / eigenvalues[0]
    delta = relative(initial, components(candidate))
    correction_m = float(np.linalg.norm(delta[:2]))
    correction_deg = abs(math.degrees(delta[2]))
    accepted = (result.accepted and np.isfinite(condition)
                and condition <= cfg.max_condition and overlap >= cfg.min_overlap
                and rmse is not None and rmse <= cfg.max_rmse_m
                and correction_m <= cfg.max_correction_m
                and correction_deg <= cfg.max_correction_deg)
    diagnostic = dict(accepted=bool(accepted), reason='accepted' if accepted else 'quality_gate',
                      overlap=overlap, rmse_m=rmse,
                      condition=float(condition) if np.isfinite(condition) else None,
                      correction_m=correction_m, correction_deg=correction_deg)
    return np.asarray(components(candidate)) if accepted else initial, diagnostic


def optimize_window(poses, increments, edges, start, cfg):
    """固定窗口首节点；地图边和网络边采用独立尺度及稳健损失。"""
    initial = np.asarray(poses[start:]).copy()
    odom_scale = np.array([cfg.odom_sigma_m, cfg.odom_sigma_m,
                           math.radians(cfg.odom_sigma_deg)])
    map_scale = np.array([cfg.map_sigma_m, cfg.map_sigma_m,
                          math.radians(cfg.map_sigma_deg)])
    active = [(i, j, value) for i, j, value in edges
              if start <= j < len(poses) and (cfg.strategy != 'legacy' or i >= start)]
    # Older endpoints are fixed at their last optimized poses, not marginalized.
    fixed = {i: np.asarray(poses[i]).copy() for i, _, _ in active if i < start}
    if not active or len(initial) < 2:
        return initial, True

    def residual(values):
        current = np.vstack((initial[0], values.reshape(-1, 3)))
        errors = []
        for k in range(len(current) - 1):
            error = relative(current[k], current[k + 1]) - increments[start + k]
            error[2] = wrap_angle(error[2])
            errors.extend(error / odom_scale)
        for i, j, measured in active:
            anchor = fixed[i] if i < start else current[i - start]
            error = relative(anchor, current[j - start]) - measured
            error[2] = wrap_angle(error[2])
            errors.extend(error / map_scale)
        return np.asarray(errors)

    result = least_squares(residual, initial[1:].ravel(), loss='soft_l1', max_nfev=60)
    if not result.success or not np.isfinite(result.x).all():
        return initial, False
    optimized = np.vstack((initial[0], result.x.reshape(-1, 3)))
    optimized[:, 2] = wrap_angle(optimized[:, 2])
    return optimized, True


def run_backend(increments, point_loader, cfg):
    """输入只含前端增量和观测，无真值或首尾相同假设。"""
    poses, raw = [np.zeros(3)], [np.zeros(3)]
    frames = {0: point_loader(0)}
    edges, diagnostics = [], []
    for index, increment in enumerate(increments, start=1):
        raw.append(compose(raw[-1], increment))
        poses.append(compose(poses[-1], increment))
        current = point_loader(index)
        start = max(0, index - cfg.window)
        anchor = matrix(*poses[start])
        maps = [transform_points(inverse(anchor) @ matrix(*poses[k]), frames[k])
                for k in range(start, index)]
        target = thin_points(np.concatenate(maps), cfg.voxel_m, cfg.max_points * cfg.window)
        estimate = relative(poses[start], poses[index])
        if index < 2:
            diagnostic = {'accepted': False, 'reason': 'warmup'}
        else:
            measured, diagnostic = match_map(current, target, estimate, cfg)
            if diagnostic['accepted']:
                edges.append((start, index, measured))
                optimized, success = optimize_window(poses, increments, edges, start, cfg)
                if success:
                    poses[start:] = list(optimized)
                else:
                    edges.pop()
                    diagnostic.update(accepted=False, reason='optimizer_failed')
        diagnostics.append(dict(index=index, **diagnostic))
        # 当前帧在匹配和优化之后才进入地图；旧帧随修正位姿重新投影。
        frames[index] = current
        frames = {k: value for k, value in frames.items() if k >= max(0, index + 1 - cfg.window)}
        # Forward-time factors expire only when both endpoints precede the next window.
        endpoint = 0 if cfg.strategy == 'legacy' else 1
        edges = [edge for edge in edges if edge[endpoint] >= max(0, index + 1 - cfg.window)]
        if index % 50 == 0:
            print(f'已处理 {index}/{len(increments)} 个连续帧对', flush=True)
    return np.asarray(raw), np.asarray(poses), diagnostics


def metrics(poses, target):
    errors = np.linalg.norm(poses[:, :2] - target[:, :2], axis=1)
    yaw = wrap_angle(poses[:, 2] - target[:, 2])
    return dict(position_rmse_start_aligned_m=float(np.sqrt(np.mean(errors ** 2))),
                endpoint_translation_error_m=float(errors[-1]),
                endpoint_yaw_error_deg=abs(math.degrees(yaw[-1])),
                yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaw ** 2)))))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(output, segment, sequence, times, raw, corrected, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False
    elapsed = np.asarray(times, dtype=float) - float(times[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(target[:, 0], target[:, 1], color='black', label='真值增量积分参考')
    for name, poses, color in [('原始前端', raw, '#c94d40'), ('局部地图后端', corrected, '#176fbb')]:
        axes[0].plot(poses[:, 0], poses[:, 1], color=color, label=name)
        axes[1].plot(elapsed, np.linalg.norm(poses[:, :2] - target[:, :2], axis=1), color=color)
        axes[2].plot(elapsed, np.degrees(wrap_angle(poses[:, 2] - target[:, 2])), color=color)
    axes[0].set(xlabel='前向坐标 / 米', ylabel='左向坐标 / 米', title='轨迹（仅起点对齐）')
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='时间 / 秒', ylabel='位置误差 / 米', title='累计位置误差')
    axes[2].set(xlabel='时间 / 秒', ylabel='角度误差 / 度', title='累计角度误差（折返至 ±180°）')
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f'{sequence}：连续片段 {segment}，无首尾闭环约束')
    fig.savefig(output / f'trajectory_{segment}.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, help='已有前端预测 CSV')
    parser.add_argument('--manifest', required=True, help='与预测对应的 BEV 帧对清单')
    parser.add_argument('--output-dir', required=True, help='新的结果目录，不覆盖已有结果')
    parser.add_argument('--mode', choices=('frontend', 'backend'), default='backend')
    parser.add_argument('--window', type=int, default=8, help='最近观测帧数')
    parser.add_argument('--strategy', choices=('fixed_history', 'legacy'), default='fixed_history',
                        help='Fixed historical endpoints, or original filtering and ICP gate')
    parser.add_argument('--plot', action='store_true', help='输出中文轨迹和误差曲线（需要 matplotlib）')
    args = parser.parse_args(argv)
    cfg = BackendConfig(window=args.window, strategy=args.strategy)
    if cfg.window < 2:
        parser.error('--window 必须至少为 2')
    output = Path(args.output_dir)
    if output.exists():
        parser.error('输出目录已存在，请选择新目录')
    with Path(args.predictions).open(encoding='utf-8-sig', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    with Path(args.manifest).open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not predictions or not records:
        parser.error('输入为空')
    manifests = {pair_key(row): row for row in records}
    if len(manifests) != len(records) or len({pair_key(row) for row in predictions}) != len(predictions):
        parser.error('输入包含重复的帧对')
    for row in predictions:
        if pair_key(row) not in manifests:
            parser.error(f'预测与清单无法对应: {pair_key(row)}')
    output.mkdir(parents=True)
    summary = dict(config=asdict(cfg), mode=args.mode, predictions=str(Path(args.predictions).resolve()),
                   manifest=str(Path(args.manifest).resolve()), input_pairs=len(predictions), segments=[])
    trajectories = []
    for segment, (sequence, chain) in enumerate(select_chains(predictions)):
        print(f'序列 {sequence}，片段 {segment}，{len(chain)} 个连续帧对', flush=True)
        increments = np.array([[float(row[key]) for key in
                               ('pred_forward_m', 'pred_left_m', 'pred_yaw_rad')] for row in chain])
        if not np.isfinite(increments).all():
            raise ValueError('前端预测含非有限值')
        matching = [manifests[pair_key(row)] for row in chain]
        for previous, current in zip(matching, matching[1:]):
            if Path(previous['bev_1']).resolve() != Path(current['bev_0']).resolve():
                raise ValueError('连续时间戳对应了不同的 BEV 文件')
        filenames = [matching[0]['bev_0']] + [row['bev_1'] for row in matching]
        if args.mode == 'backend':
            raw, corrected, diagnostics = run_backend(increments, lambda i: load_points(filenames[i], cfg), cfg)
        else:
            raw = [np.zeros(3)]
            for increment in increments:
                raw.append(compose(raw[-1], increment))
            raw = np.asarray(raw)
            corrected, diagnostics = raw.copy(), []
        # 真值只在后端完成后用于评分，绝不进入匹配和优化。
        target = [np.zeros(3)]
        for row in matching:
            target.append(compose(target[-1], [row['forward_m'], row['left_m'], row['yaw_rad']]))
        target = np.asarray(target)
        times = [chain[0]['timestamp_0']] + [row['timestamp_1'] for row in chain]
        if args.plot:
            save_plot(output, segment, sequence, times, raw, corrected, target)
        for i, timestamp in enumerate(times):
            entry = dict(sequence=sequence, segment=segment, timestamp=float(timestamp))
            for name, values in [('raw', raw), ('optimized', corrected), ('target', target)]:
                entry.update({f'{name}_{axis}': float(value) for axis, value in
                              zip(('x_m', 'y_m', 'yaw_rad'), values[i])})
            trajectories.append(entry)
        summary['segments'].append(dict(sequence=sequence, segment=segment, pairs=len(chain),
                                        accepted_matches=sum(d['accepted'] for d in diagnostics),
                                        raw=metrics(raw, target), optimized=metrics(corrected, target)))
        (output / f'matches_{segment}.json').write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    summary['selected_pairs'] = sum(s['pairs'] for s in summary['segments'])
    summary['omitted_overlapping_pairs'] = len(predictions) - summary['selected_pairs']
    report = ['# 局部地图后端评估', '',
              '参考轨迹由清单中的真值平面增量积分得到；仅起点对齐，不进行尺度拟合或首尾强制闭合。', '',
              '| 序列 / 片段 | 连续帧对 | 接受匹配 | 位置均方根误差：前端 → 后端（米） | 终点位置误差：前端 → 后端（米） | 终点角度误差：前端 → 后端（度） |',
              '| --- | --- | --- | --- | --- | --- |']
    for item in summary['segments']:
        before, after = item['raw'], item['optimized']
        fields = ('position_rmse_start_aligned_m', 'endpoint_translation_error_m', 'endpoint_yaw_error_deg')
        cells = ' | '.join(f'{before[key]:.3f} → {after[key]:.3f}' for key in fields)
        report.append(f"| {item['sequence']} / {item['segment']} | {item['pairs']} | {item['accepted_matches']} | {cells} |")
    report.extend(['', f"输入 {len(predictions)} 个帧对，选取 {summary['selected_pairs']} 个非重叠连续帧对；其余重叠帧对不累加。",
                   '时间断点处分别重置起点，不能将这些片段称为完整序列轨迹。匹配接受率高不代表匹配一定正确。'])
    (output / '报告.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    write_csv(output / 'trajectories.csv', trajectories)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
