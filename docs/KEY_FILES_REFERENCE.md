# CanFields Key Files Quick Reference

## Core Training Files

| File | Purpose |
|------|---------|
| `train_debugV2.py` | Main training entry point |
| `multi_tools.py` | Trainer classes, dataset loading, optimization |
| `utils.py` | Utilities, learning rate schedules, model I/O |
| `eval_metrics.py` | Quantitative evaluation |
| `eval_sphere.py` | Shape reconstruction and visualization |

## Dataset Loaders

| File | Dataset | Class |
|------|---------|-------|
| `datasets/data_pheno4d.py` | Pheno4D (plants) | `DeformSamples` |
| `datasets/data_wmw.py` | Deforming4D (animals) | `DeformSamples` |
| `datasets/data_dfaust.py` | D-FAUST (humans) | `DeformSamples` |
| `datasets/data_medician.py` | Medical | `DeformSamples` |

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/specs.json` | Main training specs (epochs, network arch, loss weights) |
| `configs/others_specs.json` | Specs for alternative methods (4DSDF, TPSR, etc.) |
| `configs/data_splits.json` | Train/Val/Test split definitions (plant-level) |
| `configs/point_splits/*.json` | Point-level train/test splits per timepoint (50/50) |

## Network Architecture Files

| File | Purpose |
|------|---------|
| `networks/multi_velocity_debug.py` | Main decoder with ODE velocity field |
| `networks/other_volocities.py` | Alternative method implementations |
| `networks/modules.py` | Network building blocks |
| `networks/SDF.py` | SDF decoder module |
| `networks/DIGS.py` | DIGS-based methods |

## Loss Functions

| File | Purpose |
|------|---------|
| `losses/recon.py` | Reconstruction loss (point cloud + gradient matching) |
| `losses/part_loss.py` | Part segmentation loss |
| `losses/kill_vec.py` | Velocity field regularization |
| `losses/digs_recon.py` | DIGS-specific reconstruction loss |

## Metrics and Evaluation

| File | Purpose |
|------|---------|
| `metrics/metrics.py` | PlantMetrics class (Chamfer distance, etc.) |
| `metrics/aggregate_results.py` | Results aggregation across runs |

## Data Processing

| File | Purpose |
|------|---------|
| `datasets/prepare_pheno4d.py` | Pheno4D dataset preparation and preprocessing |

## Key Data Flow

```
train_debugV2.py (main entry)
    ↓
load_dataset() [multi_tools.py]
    ↓
data_pheno4d.DeformSamples or other dataset loader
    ↓
Trainer or Trainer_woadjust [multi_tools.py]
    ↓
forward_epoch() → loss computation → optimization
    ↓
compute_and_save_metrics() → metrics JSON
```

## Important Global Parameters

### From specs.json
- `NumEpochs`: 15001
- `CodeLength`: 128
- `SamplesPerScene`: 5000
- Learning rate schedules for 6 different components

### From Command Line
- `--clip "start_idx,gap,stride"`: Frame selection
- `--adjust_weights "l2,elas,prob"`: Adjustment network weights
- `--epoch_velo`: When to start velocity constraints
- `--dataset_name`: animals | defaust | medical | pheno4d

## Dataset Directory Structure

```
datasets/
├── Pheno4D/
│   └── processed/v2/organ_sequences/
│       ├── maize/
│       ├── sorghum/
│       ├── tobacco/
│       └── tomato/
│           └── {sequence_name}/
│               ├── sparse_pc/  (*.ply files)
│               └── dense_pc/   (*.ply files)
├── Deforming4D/notexture/
│   └── {sequence}/4k_pc/
└── D_Faust/ and Medical/
```

## Output Directory Structure

```
results/
└── pheno4d/
    └── {method}/              (ode_solve, 4DSDF, etc.)
        └── {crop}/            (maize, sorghum, etc.)
            └── {sequence}/
                └── {timestamp}_{experiment_name}/
                    ├── checkpoints/
                    │   ├── ModelParameters/
                    │   ├── LatentCodes/
                    │   ├── ShapeLatentCodes/
                    │   └── AdjustLatentCodes/
                    ├── metrics/
                    ├── meshes/
                    └── wandb/
```

## Key Classes and Methods

### Trainer (multi_tools.py)
- `__init__()` - Initialize trainer with dataset and networks
- `forward_epoch(epoch)` - Main training loop for one epoch
- `compute_and_save_metrics(epoch)` - Evaluate and save metrics
- `save_model(epoch, directory)` - Save checkpoints
- `save_mesh(epoch)` - Generate and save mesh reconstruction

### DeformSamples (data_pheno4d.py)
- `__init__(filename, start_idx, gap, train_step, use_dense)` - Load dataset
- `__len__()` - Get dataset size
- `__getitem__(idx)` - Get sample with pts, nms, normed_time
- `get_leaf_tips(idx)` - Get leaf tip point indices
- `evaluate_train(idx)` - Get all points for a timestep (for mesh generation)

### Normalization in DeformSamples
```python
self.mean = big_pc.mean(0, keepdim=True)
self.scale = 1.0 / torch.max(torch.abs(big_pc - self.mean))
normalized = (points - self.mean) * self.scale
```

## Example Training Command

```bash
python train_debugV2.py \
  --specifications configs/specs.json \
  --adjust_weights 0.05,0.05,0.01 \
  --epoch_velo 7000 \
  --adjust_use True \
  --methods ode_solve \
  --multi_ode False \
  --clip 0,9,1 \
  --dataset_name pheno4d \
  --object_name datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1 \
  --base_results_dir results
```
