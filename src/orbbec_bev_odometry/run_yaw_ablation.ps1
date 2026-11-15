param(
    [ValidateSet('control', 'range', 'cosine')][string]$Variant = 'control',
    [int]$Epochs = 5,
    [string]$Checkpoint = 'D:\dataset\openloris_bev_checkpoints\aligned_depth_yaw4\best.pt',
    [string]$Cache = 'D:\dataset\openloris_bev_cache\aligned_depth',
    [string]$OutputRoot = 'D:\dataset\openloris_bev_checkpoints\yaw_fix_20261013'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$env:PYTHONPATH = $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$arguments = @(
    '-m', 'orbbec_bev_odometry.train_learned_bev',
    '--train-manifest', (Join-Path $Cache 'manifests\train.jsonl'),
    '--validation-manifest', (Join-Path $Cache 'manifests\validation.jsonl'),
    '--pretrained-checkpoint', $Checkpoint,
    '--output-dir', (Join-Path $OutputRoot $Variant),
    '--epochs', $Epochs, '--batch-size', '2', '--workers', '0',
    '--device', 'cuda', '--learning-rate', '0.00002',
    '--yaw-weight', '4', '--best-score', 'combined', '--seed', '7'
)
if ($Variant -ne 'control') { $arguments += @('--yaw-residual-deg', '8') }
if ($Variant -eq 'cosine') { $arguments += @('--correlation-mode', 'cosine') }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Training failed (exit $LASTEXITCODE)" }
