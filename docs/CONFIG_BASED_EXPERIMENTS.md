# Running Experiments with Config Files

This guide shows how to run experiments using JSON config files instead of command-line arguments.

## Why Use Configs?

✅ **Benefits:**
- **Reproducible**: Config files document exact experiment settings
- **Shareable**: Easy to share experiment configs with colleagues
- **Organized**: Keep all related experiments in one place
- **Version Control**: Track experiment configs with git
- **Less Error-Prone**: No typos in long command lines

## Available Configs

Check available experiment configs:
```bash
ls configs/experiments/
```

Current configs:
```
normal_of4d.json          - Standard OF4D (all timesteps)
interp_stride2.json       - Interpolation stride=2 with OF4D
interp_stride3.json       - Interpolation stride=3 with OF4D
interp_stride2_ode.json   - Interpolation stride=2 with ode_solve
```

## Quick Start

### 1. Single Sequence (Local)

```bash
# Run one sequence with a config
bash run_experiment_from_config.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1 \
  configs/experiments/interp_stride2.json
```

### 2. Multiple Sequences (Local)

```bash
# Run all test leaf sequences for maize
bash run_batch_experiments.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test

# Run specific sequences only
bash run_batch_experiments.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test \
  maize_control_plant1_leaf1 \
  maize_control_plant1_leaf2
```

### 3. HPC Cluster (SLURM)

```bash
# Make sure log directory exists
mkdir -p sbatch_log

# Submit batch job
sbatch run_euler_from_config.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test

# Submit for specific sequences
sbatch run_euler_from_config.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test \
  maize_control_plant1_leaf1 \
  maize_control_plant1_leaf2
```

## Creating Custom Configs

### Config Structure

```json
{
  "_comment": "Your comment here",
  "description": "Human-readable description",
  "experiment_type": "interpolation or normal",

  "data_config": {
    "interp_stride": 2,
    "clip": [0, null, 1]
  },

  "training_config": {
    "epochs": 15000,
    "adjust_use": false,
    "adjust_weights": [0.05, 0.05, 0.01],
    "method": "OF4D",
    "multi_ode": true,
    "epoch_velo": 10000
  },

  "evaluation_config": {
    "eval_epochs": "all"
  }
}
```

### Config Parameters

#### `data_config`
- **`interp_stride`**: `null` (normal) or integer (interpolation mode)
- **`clip`**: `[start_idx, gap, train_step]`
  - `start_idx`: Starting frame (usually 0)
  - `gap`: Number of frames (null = all available)
  - `train_step`: Ignored if `interp_stride` is set

#### `training_config`
- **`epochs`**: Number of training epochs (default: 15000)
- **`method`**: `"OF4D"`, `"ode_solve"`, `"4DSDF"`, `"TPSR"`, `"NVFI"`
- **`adjust_use`**: `true` for ode_solve, `false` for others
- **`adjust_weights`**: `[weight1, weight2, weight3]` (default: `[0.05, 0.05, 0.01]`)
- **`multi_ode`**: `true` for OF4D, `false` for ode_solve
- **`epoch_velo`**: Epoch to start velocity regularization (default: 10000)

### Example: Custom Interpolation Config

Create `configs/experiments/my_experiment.json`:
```json
{
  "description": "Custom interpolation experiment",
  "experiment_type": "interpolation",

  "data_config": {
    "interp_stride": 4,
    "clip": [0, null, 1]
  },

  "training_config": {
    "epochs": 20000,
    "adjust_use": false,
    "adjust_weights": [0.05, 0.05, 0.01],
    "method": "OF4D",
    "multi_ode": true,
    "epoch_velo": 12000
  },

  "evaluation_config": {
    "eval_epochs": "all"
  }
}
```

Then run it:
```bash
sbatch run_euler_from_config.sh \
  configs/experiments/my_experiment.json \
  maize \
  leaf \
  test
```

## Common Experiment Patterns

### Pattern 1: Compare Different Strides

```bash
# Create configs for stride 2, 3, 4
# Then run all of them:
sbatch run_euler_from_config.sh configs/experiments/interp_stride2.json maize leaf test
sbatch run_euler_from_config.sh configs/experiments/interp_stride3.json maize leaf test
sbatch run_euler_from_config.sh configs/experiments/interp_stride4.json maize leaf test
```

### Pattern 2: Compare Different Methods

```bash
# Run same stride with different methods
sbatch run_euler_from_config.sh configs/experiments/interp_stride2.json maize leaf test
sbatch run_euler_from_config.sh configs/experiments/interp_stride2_ode.json maize leaf test
sbatch run_euler_from_config.sh configs/experiments/interp_stride2_4dsdf.json maize leaf test
```

### Pattern 3: Quick Test Run

```bash
# Test on 2 sequences first
bash run_batch_experiments.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test \
  maize_control_plant1_leaf1 \
  maize_control_plant1_leaf2

# If successful, run all
sbatch run_euler_from_config.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test
```

### Pattern 4: All Crops

```bash
# Run across all crops (maize, sorghum, tobacco, tomato)
for crop in maize sorghum tobacco tomato; do
  sbatch run_euler_from_config.sh \
    configs/experiments/interp_stride2.json \
    $crop \
    leaf \
    test
done
```

## Monitoring

### Check Running Jobs
```bash
squeue -u $USER
```

### Monitor Logs
```bash
# Real-time monitoring
tail -f sbatch_log/<job_id>_config.out

# Check progress
grep "Processing" sbatch_log/<job_id>_config.out

# Check summary
grep "BATCH EXPERIMENT SUMMARY" -A 5 sbatch_log/<job_id>_config.out
```

### Check Results
```bash
# List completed experiments
ls -lht results/pheno4d/OF4D/maize/

# Verify interpolation mode (should contain "interp_stride")
ls results/pheno4d/OF4D/maize/maize_control_plant1_leaf1/
```

## Comparison: Config vs Command-Line

### Old Way (Command-Line)
```bash
sbatch run_euler_interp.sh maize OF4D 2 leaf test
```
- Quick for simple cases
- Hard to reproduce exact settings
- Easy to make typos

### New Way (Config)
```bash
sbatch run_euler_from_config.sh configs/experiments/interp_stride2.json maize leaf test
```
- Config file documents all settings
- Easy to share and reproduce
- Version control friendly
- Less error-prone

## Tips

1. **Start Local**: Test configs locally before submitting to cluster
2. **Name Configs Clearly**: Use descriptive names like `interp_stride2_of4d_long.json`
3. **Add Comments**: Use `"_comment"` fields to document choices
4. **Version Control**: Commit configs to git along with results
5. **Share Configs**: Share `configs/experiments/` with colleagues for reproducibility

## Troubleshooting

### Config file not found
```bash
ls configs/experiments/  # List available configs
```

### Invalid JSON
```bash
# Validate your config
python3 -c "import json; json.load(open('configs/experiments/my_config.json'))"
```

### Wrong parameters loaded
```bash
# Check what parameters are being read
python3 << 'EOF'
import json
config = json.load(open('configs/experiments/interp_stride2.json'))
print(json.dumps(config, indent=2))
EOF
```

## Backward Compatibility

✅ **All old scripts still work!**

You can use:
- Config-based approach (new, recommended)
- Command-line approach (existing scripts)

Both approaches are fully supported and can coexist.
