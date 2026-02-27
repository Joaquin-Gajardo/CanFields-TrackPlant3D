# Running Experiments - Complete Guide

This guide covers all ways to run experiments (normal and interpolation).

## Quick Reference

| Task | Command |
|------|---------|
| **Single sequence (local)** | `bash run_train_eval.sh <path> <method>` |
| **Multiple sequences (local)** | `bash run_crop_sequences.sh <crop> <method> <type> <split>` |
| **SLURM cluster** | `sbatch run_euler.sh <crop> <method> <type> <split>` |
| **Interpolation (local)** | `bash run_train_eval_interp.sh <path> <method> <stride>` |
| **Interpolation (SLURM)** | `sbatch run_euler_interp.sh <crop> <method> <stride> <type> <split>` |
| **Config-based (local)** | `bash run_batch_experiments.sh <config> <crop> <type> <split>` |
| **Config-based (SLURM)** | `sbatch run_euler_from_config.sh <config> <crop> <type> <split>` |

## Approach 1: Command-Line (Existing Workflow)

### Normal Experiments

```bash
# Single sequence
bash run_train_eval.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1 \
  OF4D

# Multiple sequences (SLURM)
sbatch run_euler.sh maize OF4D leaf test
```

**Features:**
- ✅ Simple and familiar
- ✅ Quick for single experiments
- ❌ Hard to reproduce exact settings
- ❌ Many command-line arguments

### Interpolation Experiments (New)

```bash
# Single sequence
bash run_train_eval_interp.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1 \
  OF4D \
  2

# Multiple sequences (SLURM)
sbatch run_euler_interp.sh maize OF4D 2 leaf test
```

**Features:**
- ✅ Easy interpolation experiments
- ✅ Automatic evaluation on TEST timesteps
- ❌ Still command-line based

## Approach 2: Config-Based (Recommended for Reproducibility)

### Using Configs

```bash
# Single sequence
bash run_experiment_from_config.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1 \
  configs/experiments/interp_stride2.json

# Multiple sequences (SLURM)
sbatch run_euler_from_config.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test
```

**Features:**
- ✅ Fully reproducible
- ✅ Easy to share
- ✅ Version control friendly
- ✅ Less error-prone
- ✅ Works for both normal and interpolation

## Which Approach to Use?

### Use **Command-Line** if:
- Quick one-off experiment
- Already familiar with the arguments
- Don't need to share exact settings

### Use **Config-Based** if:
- Running systematic experiments
- Need to reproduce results
- Sharing with colleagues
- Publishing results

## Common Workflows

### Workflow 1: Quick Local Test

```bash
# Test one sequence locally first
bash run_experiment_from_config.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1 \
  configs/experiments/interp_stride2.json

# If successful, run all on cluster
sbatch run_euler_from_config.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test
```

### Workflow 2: Compare Methods

```bash
# Create configs for each method
# Then submit all at once:
for config in configs/experiments/interp_stride2*.json; do
  sbatch run_euler_from_config.sh $config maize leaf test
done
```

### Workflow 3: Compare Strides

```bash
# Run different interpolation strides
sbatch run_euler_interp.sh maize OF4D 2 leaf test
sbatch run_euler_interp.sh maize OF4D 3 leaf test
sbatch run_euler_interp.sh maize OF4D 4 leaf test
```

### Workflow 4: All Crops

```bash
# Normal experiments on all crops
sbatch run_euler.sh all OF4D leaf test

# Interpolation on all crops
sbatch run_euler_interp.sh all OF4D 2 leaf test

# Config-based on all crops
for crop in maize sorghum tobacco tomato; do
  sbatch run_euler_from_config.sh configs/experiments/interp_stride2.json $crop leaf test
done
```

## File Organization

```
CanFields/
├── configs/
│   └── experiments/          # Experiment configs (new)
│       ├── normal_of4d.json
│       ├── interp_stride2.json
│       ├── interp_stride3.json
│       └── interp_stride2_ode.json
│
├── run_train_eval.sh         # Normal: single sequence
├── run_crop_sequences.sh     # Normal: multiple sequences
├── run_euler.sh              # Normal: SLURM wrapper
│
├── run_train_eval_interp.sh  # Interpolation: single sequence
├── run_crop_sequences_interp.sh  # Interpolation: multiple sequences
├── run_euler_interp.sh       # Interpolation: SLURM wrapper
│
├── run_experiment_from_config.sh  # Config: single sequence
├── run_batch_experiments.sh  # Config: multiple sequences
└── run_euler_from_config.sh  # Config: SLURM wrapper
```

## Documentation Files

- **`INTERPOLATION_MODE_GUIDE.md`**: Technical details of interpolation mode
- **`CONFIG_BASED_EXPERIMENTS.md`**: How to use config files
- **`RUN_INTERPOLATION_EXPERIMENTS.md`**: Command-line interpolation guide
- **`RUNNING_EXPERIMENTS.md`**: This file - complete overview
- **`CHANGES_SUMMARY.md`**: Summary of all changes

## Backward Compatibility

✅ **All existing workflows still work!**

You can continue using your existing scripts:
- `run_euler.sh` for normal experiments
- `run_crop_sequences.sh` for batches
- `run_train_eval.sh` for single sequences

The new interpolation and config-based approaches are **additions**, not replacements.

## Quick Start Examples

### Example 1: Your First Interpolation Experiment
```bash
# Test locally on one sequence
bash run_train_eval_interp.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1 \
  OF4D \
  2

# If successful, run on cluster for all test leaves
sbatch run_euler_interp.sh maize OF4D 2 leaf test
```

### Example 2: Your First Config-Based Experiment
```bash
# Check available configs
ls configs/experiments/

# Test locally
bash run_experiment_from_config.sh \
  datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1_leaf1 \
  configs/experiments/interp_stride2.json

# Run on cluster
sbatch run_euler_from_config.sh \
  configs/experiments/interp_stride2.json \
  maize \
  leaf \
  test
```

### Example 3: Systematic Comparison
```bash
# Create configs for your experiment variants
# configs/experiments/my_exp_method1.json
# configs/experiments/my_exp_method2.json
# configs/experiments/my_exp_method3.json

# Run all variants
for config in configs/experiments/my_exp_*.json; do
  sbatch run_euler_from_config.sh $config maize leaf test
done
```

## Next Steps

1. **Read the guides**:
   - Interpolation: `INTERPOLATION_MODE_GUIDE.md`
   - Configs: `CONFIG_BASED_EXPERIMENTS.md`

2. **Test locally** before submitting large jobs

3. **Start with one sequence** to verify everything works

4. **Use configs** for reproducible research

5. **Check logs** regularly for any issues

## Support

For issues or questions:
1. Check the relevant guide (see above)
2. Look at example configs in `configs/experiments/`
3. Test locally before submitting to cluster
4. Check SLURM logs in `sbatch_log/`
