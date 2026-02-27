# CanFields Data Loading and Processing Guide

## Overview
CanFields is a method for consolidating 4D dynamic shapes from raw scans. It loads plant point cloud data and processes it through a training pipeline that optimizes for both canonical geometry and deformation jointly.

## 1. Main Training Scripts

### Primary Training Script
**File:** `/mnt/Data/jgajardo/code/CanFields/train_debugV2.py`

**Purpose:** Main entry point for training CanFields models.

**Key Functions:**
- `velocity_test()` - Main training loop orchestrator
  - Loads configuration from specs.json
  - Initializes dataset based on dataset_name (animals, defaust, medical, pheno4d)
  - Sets up neural networks (decoder, adaptor if using adjustments)
  - Creates Trainer instance
  - Runs epochs with forward_epoch() calls
  - Saves checkpoints and metrics periodically

**Command-line Arguments:**
```
--specifications    : Path to JSON config file (default: configs/specs.json)
--adjust_weights    : Adjustment network weights (format: "l2,elas,prob")
--epoch_velo        : Starting epoch for velocity constraints
--adjust_use        : Whether to use adjustment network (True/False)
--methods           : Method name (ode_solve, 4DSDF, TPSR, NVFI, OF4D)
--multi_ode         : Whether to use multi_ode
--clip              : Data clipping params (format: "start_idx,gap,stride")
--dataset_name      : Dataset choice (animals, defaust, medical, pheno4d)
--pts_num           : Checkpoint number to load for resuming
--object_name       : Path to specific sequence
```

### Evaluation Scripts
- **`eval_metrics.py`** - Compute and log quantitative metrics
- **`eval_sphere.py`** - Generate consolidated shapes and reconstructed results

## 2. Dataset Classes and Data Loading

### Pheno4D Dataset (Primary Focus)
**File:** `/mnt/Data/jgajardo/code/CanFields/datasets/data_pheno4d.py`

**Class:** `DeformSamples`

**Constructor Parameters:**
```python
def __init__(self, filename, start_idx=0, gap=None, train_step=1, use_dense=False)
```

- `filename` - Path to sequence directory (e.g., "Pheno4D/maize/maize_control_plant2_leaf2")
- `start_idx` - Starting frame index (default: 0)
- `gap` - Number of consecutive frames to include (None = all frames)
- `train_step` - Temporal sampling stride (1=all frames, 2=every other, etc.)
- `use_dense` - Use dense_pc (True) or sparse_pc (False) folder

**Data Loading Process:**
1. Loads .ply files from `{path}/sparse_pc/` or `{path}/dense_pc/`
2. Extracts timestamps from filename (format: "index_name_D##_leaf#.ply")
3. Sorts files by day number and applies start_idx/gap/train_step filtering
4. For each .ply file:
   - Loads point positions and optional normals using Open3D
   - Reads leaf_tip mask if available
   - Deduplicates points
   - Filters points with invalid normals (norm < 1e-5)
   - Estimates normals if not available using Open3D
   - Updates leaf_tip mask after filtering

**Normalization:**
- Computes mean and scale from concatenated point clouds
- Normalizes all points to zero-center with unit scale
- Computes bounding box in normalized space

**Data Format:**
Returns dictionary for each timestep:
```python
{
    "pts": point_cloud_tensor,      # [N, 3] normalized coordinates
    "nms": normals_tensor,          # [N, 3] normalized normals
    "normed_time": time_tensor      # scalar in [0, 1]
}
```

### Other Dataset Loaders

**Animals/Deforming4D:** `/mnt/Data/jgajardo/code/CanFields/datasets/data_wmw.py`
- Class: `DeformSamples`
- Loads from `{path}/4k_pc/` directory
- Format: `{name}_{frame_index}.ply`
- Samples 2000 points at runtime

**D-FAUST:** `/mnt/Data/jgajardo/code/CanFields/datasets/data_dfaust.py`
- Human body sequences
- Uses train_step parameter for temporal sampling

**Medical Data:** `/mnt/Data/jgajardo/code/CanFields/datasets/data_medician.py`
- Medical imaging sequences

### Dataset Loading in Training

**File:** `/mnt/Data/jgajardo/code/CanFields/multi_tools.py` (function: `load_dataset()`)

```python
def load_dataset(args, specs, ajust_coes=None):
    clip_weights = args.clip.split(",")
    clip_coes = [int(clip_weights[0]), int(clip_weights[1]), int(clip_weights[2])]
    
    # clip_coes = [start_idx, gap, stride]
    
    if args.dataset_name == "pheno4d":
        from datasets import data_pheno4d
        sdf_dataset = data_pheno4d.DeformSamples(
            shapepath, 
            start_idx=clip_coes[0],
            gap=clip_coes[1],
            train_step=clip_coes[2],
            use_dense=False
        )
```

## 3. Train/Test Data Splitting

### Data Splits Configuration
**File:** `/mnt/Data/jgajardo/code/CanFields/configs/data_splits.json`

**Strategy:** Plant-level splits to avoid data leakage during leaf stitching

**Train/Val Split:**
- **Val Plants (4):** maize_control_plant1, sorghum_control_plant1, tobacco_control_plant2, tomato2_control_plant3
  - Sequences: 15 leaf sequences
  - Avg tip coverage: 79.2%
  - Purpose: Hyperparameter tuning and model development
  
- **Test Plants (8):** maize_control_plant2, maize_control_plant3, sorghum_control_plant2, sorghum_highlight_plant2, tobacco_control_plant1, tobacco_shade_plant3, tomato1_heat_plant3, tomato1_shade_plant1
  - Sequences: 29 leaf sequences
  - Avg tip coverage: 78.0%
  - Purpose: Final evaluation and reporting
  - Multiple treatments: control, heat, shade, highlight

**Statistics:**
- Total plants: 12
- Total sequences: 44
- Val ratio (by plants): 33.3%
- Test ratio (by plants): 66.7%
- Val ratio (by sequences): 34.1%
- Test ratio (by sequences): 65.9%

### Point-Level Splits (Per Timepoint)
**File:** `/mnt/Data/jgajardo/code/CanFields/configs/point_splits/`

Example: `maize_control_plant1_split.json`

**Structure:**
```json
{
  "sequence_name": "maize_control_plant1",
  "random_seed": 42,
  "train_ratio": 0.5,
  "n_timepoints": 9,
  "timepoints": [
    {
      "timepoint_idx": 0,
      "n_total_points": 10000,
      "n_train": 4999,
      "n_test": 5001,
      "train_indices": [...],
      "test_indices": [...]
    },
    ...
  ]
}
```

**Purpose:** Random 50% train/50% test split at each timepoint for metric evaluation

## 4. Data Format Expectations

### Input Format
- **Point Cloud Format:** PLY files with point positions
- **Optional Attributes:** Normals, leaf_tip mask
- **Directory Structure:**
  ```
  sequence_name/
  ├── sparse_pc/          (sparse point clouds)
  │   ├── 0_seq_D00_leaf1.ply
  │   ├── 1_seq_D01_leaf1.ply
  │   └── ...
  └── dense_pc/           (dense point clouds, optional)
      ├── 0_seq_D00_leaf1.ply
      └── ...
  ```

### Normalized Data Format
After loading and normalization:
- **Point Coordinates:** Normalized to zero-center, unit scale
  - Computed from: `(x - mean) * scale`
  - Scale: `1.0 / max(|x - mean|)`
  
- **Bounding Box:** Computed in normalized space
  - bb_min, bb_max from normalized concatenated point cloud

- **Temporal Encoding:** Normalized time [0, 1]
  - Min/max extracted from file timestamps
  - Formula: `(timestamp - min_timestamp) / (max_timestamp - min_timestamp)`

### Training Data Batch
Each sample returns:
```python
{
    "pts": torch.Tensor,           # [N, 3] normalized point coordinates
    "nms": torch.Tensor,           # [N, 3] normalized point normals
    "normed_time": torch.Tensor    # scalar, normalized time in [0, 1]
}
```

## 5. Training Loop Structure

### Trainer Class
**File:** `/mnt/Data/jgajardo/code/CanFields/multi_tools.py`

**Key Classes:**
- `Trainer` - With adjustment network (uses adaptor)
- `Trainer_woadjust` - Without adjustment network

### Training Epoch Flow (in `forward_epoch()`)

1. **Sample Random Timepoint**
   - Select random frame from dataset
   - Get point cloud, normals, normalized time

2. **Adjustment Network (if enabled)**
   - Adjusts input points and normals
   - Computes confidence scores
   - Calculates adjustment losses (elastic, L2, probability)

3. **Deformation Network**
   - Decodes scene latent code with temporal info
   - Generates predictions for normalized and perturbed points

4. **Loss Computation**
   - Reconstruction loss (point cloud matching)
   - Gradient losses (surface normal consistency)
   - Latent code regularization
   - Adjustment losses
   - Optional velocity constraints (after epoch_velo)

5. **Optimization**
   - Backward pass with gradient clipping
   - Adam optimizer step with learning rate scheduling

### Metrics Computation
**Method:** `compute_and_save_metrics()` in Trainer classes

Runs periodically (every 5000 epochs or at end):
- Samples SDF grid points in bounding box
- Evaluates SDF values across timepoints
- Computes Chamfer distance to ground truth
- Saves metrics to JSON

## 6. Configuration Files

### Main Specifications
**File:** `/mnt/Data/jgajardo/code/CanFields/configs/specs.json`

**Key Parameters:**
- `NumEpochs`: 15001 (training duration)
- `CodeLength`: 128 (latent code dimension)
- `SamplesPerScene`: 5000 (SDF samples per training step)
- Network architecture specifications
- Learning rate schedules for different components
- Loss weights

### Alternative Specs
**File:** `/mnt/Data/jgajardo/code/CanFields/configs/others_specs.json`

Used for non-ode_solve methods (4DSDF, TPSR, NVFI, OF4D)

## 7. Important Implementation Details

### Point Cloud Deduplication
```python
v, idx, _ = pcu.deduplicate_point_cloud(v, 1e-15, return_index=True)
```
Removes duplicate points with tolerance 1e-15

### Normal Filtering
```python
mask = np.linalg.norm(n, axis=-1) > 1e-5
```
Keeps points with normal magnitude > 1e-5

### Normal Estimation (if missing)
- Uses Open3D `estimate_normals()`
- Orients normals consistently using tangent plane (k=30)
- Further refinement with direction alignment

### Temporal Normalization
Time is extracted from filename day markers (D00, D01, etc.) and normalized to [0, 1] range based on min/max timestamps in filtered sequence.

### Data Sampling During Training
- **Full point clouds:** Stored in memory
- **Training batches:** Random sampling during __getitem__
  - Animals: 2000 points per frame
  - Pheno4D: Full point cloud (variable size)

## 8. Running Training

### Example Command (Pheno4D)
```bash
python train_debugV2.py \
  --adjust_weights 0.05,0.05,0.01 \
  --object_name datasets/Pheno4D/processed/v2/organ_sequences/maize/maize_control_plant1 \
  --dataset_name pheno4d \
  --adjust_use True \
  --methods ode_solve \
  --multi_ode False \
  --clip 0,9,1 \
  --epoch_velo 7000 \
  --specifications configs/specs.json
```

**Parameters Explanation:**
- `--clip 0,9,1`: Use frames 0-9 (gap=9) with stride 1
- `--epoch_velo 7000`: Start velocity constraints at epoch 7000
- `--adjust_weights 0.05,0.05,0.01`: L2=0.05, elasticity=0.05, probability=0.01

### Output Directory Structure
```
results/
├── pheno4d/
│   ├── ode_solve/
│   │   └── {crop}/
│   │       └── {sequence_name}/
│   │           └── {timestamp}_{experiment_name}/
│   │               ├── checkpoints/
│   │               ├── metrics/
│   │               ├── meshes/
│   │               └── wandb/
```

