"""Planar yaw increments from the OpenLORIS D400 gyroscope.

The angular velocity (d400_gyroscope frame, 400 Hz) is rotated into
base_link with the calibration chain base_link <- d400_color_optical_frame
<- d400_gyroscope from ``trans_matrix.yaml``; the z component is the yaw
rate.  ``GyroYaw.increment(t0, t1)`` integrates it (trapezoid rule) after
subtracting a constant bias.

Bias: estimated without ground truth as the median yaw rate over windows
where the visual frontend says the robot is not turning (|yaw rate| below a
threshold), or set explicitly.
"""

import math
from pathlib import Path

import numpy as np


def _matrices(yaml_path):
    import cv2
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    node = fs.getNode('trans_matrix')
    out = {}
    for i in range(node.size()):
        item = node.at(i)
        out[(item.getNode('parent_frame').string(), item.getNode('child_frame').string())] = item.getNode('matrix').mat()
    fs.release()
    return out


class GyroYaw:
    def __init__(self, sequence_dir, bias=0.0):
        sequence_dir = Path(sequence_dir)
        m = _matrices(sequence_dir / 'trans_matrix.yaml')
        rotation = (m[('base_link', 'd400_color_optical_frame')][:3, :3]
                    @ m[('d400_color_optical_frame', 'd400_gyroscope')][:3, :3])
        data = np.loadtxt(sequence_dir / 'd400_gyroscope.txt', comments='#')
        data = data[np.argsort(data[:, 0])]
        _, unique = np.unique(data[:, 0], return_index=True)
        data = data[unique]
        self.t = data[:, 0]
        self.rate = (data[:, 1:4] @ rotation.T)[:, 2]
        self.bias = bias
        self.cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (self.rate[1:] + self.rate[:-1]) * np.diff(self.t))))

    def _angle(self, t):
        return np.interp(t, self.t, self.cumulative)

    def increment(self, t0, t1):
        return float(self._angle(t1) - self._angle(t0) - self.bias * (t1 - t0))

    def estimate_bias(self, intervals, visual_yaw, still_rate=math.radians(2.0)):
        """Median gyro rate over intervals the visual frontend calls non-turning."""
        rates = []
        for (t0, t1), yaw in zip(intervals, visual_yaw):
            dt = t1 - t0
            if dt > 0 and abs(yaw) / dt < still_rate:
                rates.append((self._angle(t1) - self._angle(t0)) / dt)
        self.bias = float(np.median(rates)) if rates else 0.0
        return self.bias
