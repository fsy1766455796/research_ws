"""PyTorch dataset for precomputed multi-channel OpenLORIS BEV pairs."""

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .openloris_data import load_pair_manifest


def invert_planar_target(target):
    """Invert ``[forward, left, yaw]`` while preserving tensor/array dtype."""
    forward, left, yaw = target
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    return np.asarray(
        [
            -cosine * forward - sine * left,
            sine * forward - cosine * left,
            -yaw,
        ],
        dtype=np.float32,
    )


class BEVPairDataset(Dataset):
    """Read compressed BEVs without loading the whole dataset into RAM."""

    def __init__(self, manifest, reverse_probability=0.0):
        """Load pair metadata and configure optional direction reversal."""
        self.records = load_pair_manifest(manifest)
        if not self.records:
            raise ValueError(f"Pair manifest is empty: {manifest}")
        self.reverse_probability = float(reverse_probability)

    def __len__(self):
        """Return the number of supervised pairs."""
        return len(self.records)

    @staticmethod
    def _load_bev(filename):
        with np.load(filename, allow_pickle=False) as archive:
            bev = np.asarray(archive["bev"], dtype=np.float32)
        return bev

    def __getitem__(self, index):
        """Load and return one BEV pair and its metric pose target."""
        record = self.records[index]
        previous = self._load_bev(record.bev_0)
        current = self._load_bev(record.bev_1)
        target = record.target
        if self.reverse_probability > 0.0 and torch.rand(()) < self.reverse_probability:
            previous, current = current, previous
            target = invert_planar_target(target)
        return {
            "previous": torch.from_numpy(previous),
            "current": torch.from_numpy(current),
            "target": torch.from_numpy(target),
            "sequence": record.sequence,
            "timestamp_0": record.timestamp_0,
            "timestamp_1": record.timestamp_1,
        }
