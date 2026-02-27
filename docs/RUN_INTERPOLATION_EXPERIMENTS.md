#!/bin/bash# Running Interpolation Experiments

This guide shows how to run temporal interpolation experiments on your HPC cluster.

## Quick Start

### Single Sequence (Local Testing)

```bash
# Run interpolation experiment for one sequence
bash run_train_eval_interp.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1 \
  OF4D \
  2
```

### Multiple Sequences (Local)

```bash
# Run all test leaf sequences for maize with interpolation stride=2
bash run_crop_sequences_interp.sh maize OF4D 2 leaf test

# Run specific sequences
bash run_crop_sequences_interp.sh maize OF4D 2 leaf test \
  maize_control_plant1_leaf1 \
  maize_control_plant1_leaf2
```

### HPC Cluster (SLURM)

```bash
# Make sure sbatch_log directory exists
mkdir -p sbatch_log

# Submit job for all test leaf sequences
sbatch run_euler_interp.sh maize OF4D 2 leaf test

# Submit job for specific sequences
sbatch run_euler_interp.sh maize OF4D 2 leaf test \
  maize_control_plant1_leaf1 \
  maize_control_plant1_leaf2

# Submit job for all crops
sbatch run_euler_interp.sh all OF4D 2 leaf test
```

## Script Arguments

### `run_train_eval_interp.sh`
```
Arguments:
  $1: sequence_path (required)
  $2: method (default: OF4D)
  $3: interp_stride (default: 2)

Example:
  bash run_train_eval_interp.sh datasets/Pheno4D/.../sequence OF4D 2
```

### `run_crop_sequences_interp.sh`
```
Arguments:
  $1: crop_name (required) - maize, sorghum, tobacco, tomato, all
  $2: method (default: OF4D) - ode_solve, 4DSDF, TPSR, NVFI, OF4D
  $3: interp_stride (default: 2) - 2, 3, 4, ...
  $4: type (default: leaf) - plant, leaf
  $5: split (default: test) - test, val, all
  $6+: specific sequences (optional)

Example:
  bash run_crop_sequences_interp.sh maize OF4D 2 leaf test
```

### `run_euler_interp.sh`
Same arguments as `run_crop_sequences_interp.sh`, but submits to SLURM.

## What Happens

For each sequence with interpolation stride=2:

### Training
- **Loads**: ALL timesteps (e.g., 0, 1, 2, 3, 4, 5, 6)
- **Trains on**: Positions 0, 2, 4, 6 (TRAIN timesteps)
- **Holds out**: Positions 1, 3, 5 (TEST timesteps for interpolation)

Example with 7 timesteps:
```
Position 0 (Day 0):  TRAIN
Position 1 (Day 1):  TEST  <-- interpolation target
Position 2 (Day 2):  TRAIN
Position 3 (Day 3):  TEST  <-- interpolation target
Position 4 (Day 4):  TRAIN
Position 5 (Day 5):  TEST  <-- interpolation target
Position 6 (Day 6):  TRAIN
```

### Evaluation
Automatically detects interpolation mode from experiment name:
- **Evaluates only**: TEST timesteps (positions 1, 3, 5)
- **Uses**: TEST points from point splits
- **Skips**: TRAIN timesteps (already seen during training)

### Experiment Naming
- **Normal**: `sequence_P4D_full_l2_woAdjust_start_id_0_gap_7_stride_1`
- **Interpolation**: `sequence_P4D_full_l2_woAdjust_start_id_0_gap_7_interp_stride_2`

## Use Cases

### Experiment 1: Quick test on one sequence
```bash
bash run_train_eval_interp.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/sorghum/sorghum_control_plant1_leaf1 \
  OF4D \
  2
```

### Experiment 2: All maize test leaves
```bash
sbatch run_euler_interp.sh maize OF4D 2 leaf test
```

### Experiment 3: Compare different strides
```bash
# Stride 2 (interpolate half the timesteps)
sbatch run_euler_interp.sh maize OF4D 2 leaf test

# Stride 3 (interpolate 2/3 of the timesteps)
sbatch run_euler_interp.sh maize OF4D 3 leaf test

# Stride 4 (interpolate 3/4 of the timesteps)
sbatch run_euler_interp.sh maize OF4D 4 leaf test
```

### Experiment 4: Different methods
```bash
# OF4D (fastest, no velocity regularization)
sbatch run_euler_interp.sh maize OF4D 2 leaf test

# ode_solve (CanFields with adaptor)
sbatch run_euler_interp.sh maize ode_solve 2 leaf test

# Other baselines
sbatch run_euler_interp.sh maize 4DSDF 2 leaf test
sbatch run_euler_interp.sh maize NVFI 2 leaf test
```

### Experiment 5: All crops, all test leaves
```bash
sbatch run_euler_interp.sh all OF4D 2 leaf test
```

## Monitoring

### Check SLURM job status
```bash
squeue -u $USER
```

### Check logs
```bash
# Real-time monitoring
tail -f sbatch_log/<job_id>_interp.out

# Check for errors
grep -i error sbatch_log/<job_id>_interp.out

# Check training progress
grep "Completed:" sbatch_log/<job_id>_interp.out
```

### Check results
```bash
# List experiments
ls -lht results/pheno4d/OF4D/maize/

# Check if interpolation mode was used (should see "interp_stride")
ls results/pheno4d/OF4D/maize/maize_control_plant1_leaf1/
```

## Comparison with Normal Experiments

### Normal Experiments (existing workflow)
```bash
# Train on all timesteps
sbatch run_euler.sh maize OF4D leaf test
```
- Trains on ALL timesteps
- Evaluates on ALL timesteps
- Uses `_stride_1` in experiment name

### Interpolation Experiments (new workflow)
```bash
# Train on every 2nd timestep, interpolate the rest
sbatch run_euler_interp.sh maize OF4D 2 leaf test
```
- Trains on TRAIN timesteps only
- Evaluates on TEST timesteps only
- Uses `_interp_stride_2` in experiment name

## Expected Output

For each sequence, you'll get:
1. **Training results**: `results/pheno4d/{method}/{crop}/{sequence}/YYYYMMDD_HHMMSS_{sequence}_{params}_interp_stride_2/`
2. **Evaluation meshes**: `{exp_dir}/manifold/epoch_{epoch}/` (only TEST timesteps)
3. **Metrics JSON**: `{exp_dir}/metrics_epoch_{epoch}.json` (only TEST timesteps)

## Troubleshooting

### Job fails immediately
```bash
# Check if conda environment is active
conda activate canfields
sbatch run_euler_interp.sh ...
```

### No point split file warning
```
No point split file found, will use all points
```
This is fine - the code will still work, just won't split points within each timestep.

### Experiment name doesn't contain "interp_stride"
Check that you're using `run_train_eval_interp.sh` and not the regular `run_train_eval.sh`.

### Evaluation finds 0 timesteps
The evaluation scripts automatically filter to TEST timesteps. If you see 0 timesteps, check:
1. Experiment name contains `interp_stride`
2. Sequence has enough timesteps (at least `interp_stride + 1`)

## Tips

1. **Start small**: Test on 1-2 sequences locally before submitting large jobs
2. **Monitor first job**: Watch the first job complete to ensure everything works
3. **Check experiment names**: Ensure they contain `_interp_stride_N`
4. **Filesystem sync**: The scripts include 10s wait after training for NFS sync

## Backward Compatibility

✅ All existing scripts still work!
- `run_euler.sh` - Normal experiments (unchanged)
- `run_crop_sequences.sh` - Normal experiments (unchanged)
- `run_train_eval.sh` - Normal experiments (unchanged)

New scripts are separate and don't affect existing workflows.
