"""Absolute heading cue from wall directions (Manhattan-world assumption).

Office walls are mostly mutually perpendicular.  For every BEV frame the
dominant wall direction (mod 90 deg) is estimated from the occupied cells:
local PCA on small neighbourhoods gives a line direction for points that lie
on straight structure; directions are averaged on the 4x angle circle.

``frame_manhattan(points)`` returns (phi, strength, count): phi in
[-45, 45) deg is the wall angle in the robot frame; strength in [0, 1] is the
resultant length of the 4x-angle vectors (1 = all lines agree).
"""

import math

import numpy as np
from scipy.spatial import cKDTree


def wrap90(angle):
    """Wrap radians to [-pi/4, pi/4)."""
    return (angle + math.pi / 4) % (math.pi / 2) - math.pi / 4


def frame_manhattan(points, radius=0.15, min_neighbors=5, linearity=0.85, max_range=4.5):
    points = points[np.hypot(points[:, 0], points[:, 1]) <= max_range]
    if len(points) < 20:
        return 0.0, 0.0, 0
    tree = cKDTree(points)
    groups = tree.query_ball_point(points, radius)
    angles, weights = [], []
    for members in groups:
        if len(members) < min_neighbors:
            continue
        local = points[members] - points[members].mean(0)
        cov = local.T @ local / len(members)
        values, vectors = np.linalg.eigh(cov)
        if values[1] <= 1e-9:
            continue
        line = 1.0 - values[0] / values[1]
        if line < linearity:
            continue
        direction = vectors[:, 1]
        angles.append(math.atan2(direction[1], direction[0]))
        weights.append(line)
    if len(angles) < 10:
        return 0.0, 0.0, len(angles)
    angles, weights = np.asarray(angles), np.asarray(weights)
    c, s = (weights * np.cos(4 * angles)).sum(), (weights * np.sin(4 * angles)).sum()
    strength = math.hypot(c, s) / weights.sum()
    return wrap90(math.atan2(s, c) / 4), float(strength), len(angles)
