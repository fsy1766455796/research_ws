# BEV odometry checkpoints

Snapshots of the locally trained OpenLORIS models. Files are copied without
modifying checkpoint contents. Each directory includes `best.pt`, `last.pt`,
and the recorded `history.jsonl`.

| Version | Directory | Best epoch (zero-based) | Last epoch (zero-based) |
| --- | --- | --- | --- |
| A0 | `aligned_depth` | 27 | 39 |
| A1 | `aligned_depth_yaw4` | 20 | 39 |
| A2 | `aligned_depth_localmap_w5_a2` | 1 | 8 |

- A0 uses measured aligned depth and the original training setup.
- A1 uses measured aligned depth, yaw loss weight 4, and combined validation
  scoring. `aligned_depth_yaw4/best.pt` is the A1 checkpoint used in the
  preceding measured-depth and DA3-input evaluations.
- A2 is a local-map training snapshot initialized from A1, with yaw loss
  weight 6 and combined validation scoring. Only nine epochs are present in
  this snapshot; it is not a completed 40-epoch run.

Use `best.pt` for evaluation of the validation-selected model. `last.pt`
preserves the latest saved training state for resuming. These checkpoints
are not DA3-fine-tuned models.

The checkpoints contain model configuration and training state in addition
to model weights. Recorded manifest and pretraining paths refer to the
original Windows machine. Supply paths to your own data when running on
another machine; the dataset and BEV caches are not included here.

For A1 evaluation, pass `checkpoints/aligned_depth_yaw4/best.pt` as the
checkpoint path. Local smoke-test checkpoints are intentionally omitted.
