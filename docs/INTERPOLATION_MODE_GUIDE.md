# Temporal Interpolation Mode for Pheno4D

## Overview

This feature enables temporal interpolation experiments where the model is trained on a subset of timepoints and tested on the remaining timepoints. This is useful for evaluating the model's ability to interpolate between observed timesteps.

## Key Concept

- **Normal mode**: Load every Nth timepoint (e.g., with `train_step=2`, only load days 0, 2, 4, ...)
- **Interpolation mode**: Load ALL timepoints, but only train on every Nth position (e.g., with `interp_stride=2`, train on positions 0, 2, 4, ... and test on positions 1, 3, 5, ...)

## Changes Made

### 1. Dataset Loader (`datasets/data_pheno4d.py`)

Added new parameter `interp_stride`:
- When `None` (default): behaves as before
- When set to integer N: loads all timepoints but marks every Nth position as training

New features:
- `is_training_timestep(idx)`: method to check if a timestep is for training
- `is_train` flag in `__getitem__`: indicates whether this timestep should be used for training

### 2. Training Scripts (`multi_tools.py`)

- Added `--interp_stride` argument
- Both `Trainer` and `Trainer_woadjust` classes now:
  - Build a list of training timestep indices
  - Only sample from training timesteps during training
  - Print which timesteps are used for training

### 3. Experiment Naming

When `interp_stride` is set, the experiment name includes `_interp_stride_N` instead of `_stride_N`.

## Usage Example

To run an interpolation experiment with stride 2:

```bash
python train_debugV2.py \\
  --dataset_name pheno4d \\
  --object_name datasets/Pheno4D/processed/v2/organ_sequences/sorghum/sorghum_control_plant1_leaf1 \\
  --methods OF4D \\
  --adjust_use False \\
  --adjust_weights 0.0,0.0,0.0 \\
  --epoch_velo 10000 \\
  --clip 0,7,1 \\
  --interp_stride 2 \\
  --multi_ode True \\
  --pts_num 1
```

### Parameter Explanation

- `--clip 0,7,1`: start at index 0, load 7 timepoints, stride 1 (ignored when interp_stride is set)
- `--interp_stride 2`: Train on positions 0,2,4,6 and test on positions 1,3,5

## Expected Behavior

For a sequence with 7 timepoints (D02, D04, D06, D08, D10, D12, D14):

**Normal mode** (`train_step=2`, no `interp_stride`):
```
Position 0 (D02): TRAIN
Position 1 (D06): TRAIN
Position 2 (D10): TRAIN
Position 3 (D14): TRAIN
```
Only 4 timepoints loaded.

**Interpolation mode** (`interp_stride=2`):
```
Position 0 (D02): TRAIN
Position 1 (D04): TEST
Position 2 (D06): TRAIN
Position 3 (D08): TEST
Position 4 (D10): TRAIN
Position 5 (D12): TEST
Position 6 (D14): TRAIN
```
All 7 timepoints loaded, alternating between train and test.

## Testing

Run the test script to verify the dataloader works correctly:

```bash
python test_interp_dataloader.py
```

## Evaluation

Both `eval_sphere.py` and `eval_metrics.py` automatically detect interpolation mode and adapt accordingly.

### Automatic Detection

The scripts detect interpolation mode by parsing the experiment name:
- Normal: `..._start_id_0_gap_7_stride_1`
- Interpolation: `..._start_id_0_gap_7_interp_stride_2`

### Evaluation Behavior

**In interpolation mode:**
1. Only **TEST timesteps** are evaluated (positions 1, 3, 5, ...)
2. Uses **TEST points** from point splits (for consistency with other experiments)
3. Point split indices are correctly mapped from filtered to original indices

**In normal mode:**
- All timesteps are evaluated
- Behaves exactly as before

### Example

For a sequence with 7 timesteps in interpolation mode (stride=2):

**Training:**
- Positions 0, 2, 4, 6 (TRAIN)

**Evaluation:**
- Positions 1, 3, 5 (TEST)
- Uses test points from those timesteps
- Skips train timesteps during evaluation

### Running Evaluation

```bash
# Generate meshes (automatically filters to TEST timesteps)
python eval_sphere.py \
  -d pheno4d \
  -o datasets/Pheno4D/.../sequence \
  -en experiment_name \
  -e 15000 \
  -me OF4D \
  -m 2

# Compute metrics (automatically filters to TEST timesteps)
python eval_metrics.py \
  -d pheno4d \
  -o datasets/Pheno4D/.../sequence \
  -en experiment_name \
  -e 15000 \
  -me OF4D
```

## Notes

- The `interp_stride` parameter **overrides** `train_step` when set
- The stride is based on **position in the sequence**, not the actual day number
- This handles sequences with irregular intervals (e.g., D02, D04, D06, ...) correctly
- During training, only training timesteps are sampled for loss computation
- During evaluation, only TEST timesteps are evaluated with TEST points
- **All changes are backward compatible** - existing experiments work without modification
