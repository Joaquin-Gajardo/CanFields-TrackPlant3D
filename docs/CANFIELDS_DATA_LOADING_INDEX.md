# CanFields Data Loading and Processing - Complete Index

This index provides a complete guide to understanding how CanFields loads, processes, and trains on plant data.

## Quick Navigation

- **For Quick Reference:** See `KEY_FILES_REFERENCE.md`
- **For Detailed Guide:** See `DATA_LOADING_GUIDE.md`
- **For Code Implementation:** See specific sections below

## 1. Entry Point: Training Script

**File:** `train_debugV2.py` (232 lines)

**Key Function:** `velocity_test()`

This is the main entry point that orchestrates the entire training pipeline:

```python
# Step 1: Parse command-line arguments
args = build_args()

# Step 2: Load configuration from specs.json
specs = json.load(open(args.specifications))

# Step 3: Initialize dataset based on dataset_name parameter
sdf_dataset, obj_name = load_dataset(args, specs, ajust_coes)

# Step 4: Load neural networks (decoder, adaptor)
decoder, start_end_vecs = load_network_model_others(args, specs, start_idx, end_idx)

# Step 5: Create Trainer instance
trainer = Trainer_woadjust(args, specs, decoder, sdf_dataset, ...)

# Step 6: Main training loop
for epoch in range(start_epoch, num_epochs + 1):
    log_dict = trainer.forward_epoch(epoch)
    wandb.log(log_dict)
    if epoch % 5000 == 0:
        metrics_dict = trainer.compute_and_save_metrics(epoch)
```

## 2. Dataset Loading Pipeline

### Entry Point: `load_dataset()` in `multi_tools.py` (lines 294-366)

This function:
1. Parses clip parameters (start_idx, gap, stride)
2. Selects appropriate dataset loader based on `args.dataset_name`
3. Creates dataset instance with temporal filtering parameters

### Pheno4D Dataset Class: `data_pheno4d.DeformSamples` (lines 9-228)

**Constructor Parameters:**
```python
DeformSamples(
    filename,          # Path to sequence directory
    start_idx=0,       # Starting frame index
    gap=None,          # Number of consecutive frames
    train_step=1,      # Temporal sampling stride
    use_dense=False    # Use dense or sparse point clouds
)
```

**Loading Process:**

1. **Locate files:** `{filename}/{'dense_pc' if use_dense else 'sparse_pc'}`
   
2. **List and sort:** 
   - Files must be .ply format
   - Extracted timestamps from filenames (D00, D01, etc.)
   - Sorted by timestamp

3. **Filter by temporal parameters:**
   ```python
   if gap is not None:
       end_idx = min(start_idx + gap, total_lens)
       sub_files = sub_files[start_idx:end_idx:train_step]
   else:
       sub_files = sub_files[start_idx::train_step]
   ```

4. **Load each point cloud:**
   - Read with Open3D tensor API: `o3d.t.io.read_point_cloud()`
   - Extract positions: `pcd_tensor.point.positions.numpy()`
   - Read optional normals: `pcd_tensor.point.normals.numpy()`
   - Read optional leaf_tip mask: `pcd_tensor.point.is_leaf_tip.numpy()`

5. **Deduplicate and filter:**
   - `pcu.deduplicate_point_cloud(v, 1e-15, return_index=True)`
   - Filter normals: `||n|| > 1e-5`

6. **Estimate normals if missing:**
   - Open3D `estimate_normals()`
   - Consistent orientation: `orient_normals_consistent_tangent_plane(k=30)`
   - Direction alignment: `orient_normals_to_align_with_direction()`

7. **Normalize all data:**
   ```python
   big_pc = torch.cat(self.train_xx, dim=0)
   self.mean = big_pc.mean(0, keepdim=True)
   self.scale = 1.0 / torch.max(torch.abs(big_pc - self.mean))
   normalized = (big_pc - self.mean) * self.scale
   self.bb_min = normalized.min(0, keepdim=True)[0]
   self.bb_max = normalized.max(0, keepdim=True)[0]
   ```

8. **Compute temporal encoding:**
   ```python
   timestamps = [extract_timestamp(f) for f in sub_files]
   min_time = min(timestamps)
   max_time = max(timestamps)
   time_range = max_time - min_time if max_time > min_time else 1
   normalized_time = (timestamp - min_time) / time_range
   ```

## 3. Data Representation

### Batch Data Format

Each `__getitem__(idx)` call returns:

```python
{
    "pts": torch.Tensor,           # Shape [N, 3], normalized coordinates
    "nms": torch.Tensor,           # Shape [N, 3], normalized normals
    "normed_time": torch.Tensor    # Shape [], scalar in [0, 1]
}
```

### Normalization Details

The dataset uses **two-level normalization**:

**Level 1: Across all points in sequence**
```python
# After concatenating all point clouds
mean = mean of all points across all timesteps
scale = 1.0 / max(|points - mean|)
```

**Level 2: Per-frame access**
```python
# When accessing frame i
frame_points = (frame_points - dataset.mean) * dataset.scale
```

**Important:** The `mean` and `scale` are computed once during initialization and applied consistently to all frames.

## 4. Data Splitting Strategy

### Plant-Level Splits (from `configs/data_splits.json`)

**Rationale:** Avoid data leakage when stitching leaves from same plant

**Validation Set:**
- 4 plants: maize_control_plant1, sorghum_control_plant1, tobacco_control_plant2, tomato2_control_plant3
- 15 leaf sequences total
- Used for: Hyperparameter tuning and development

**Test Set:**
- 8 plants: maize_control_plant2, maize_control_plant3, sorghum_control_plant2, sorghum_highlight_plant2, tobacco_control_plant1, tobacco_shade_plant3, tomato1_heat_plant3, tomato1_shade_plant1
- 29 leaf sequences total
- Used for: Final evaluation and reporting results

### Point-Level Splits (from `configs/point_splits/`)

Per-timepoint random 50/50 train/test splits:
```json
{
  "sequence_name": "maize_control_plant1",
  "n_timepoints": 9,
  "timepoints": [
    {
      "timepoint_idx": 0,
      "n_total_points": 10000,
      "n_train": 4999,
      "n_test": 5001,
      "train_indices": [...],
      "test_indices": [...]
    }
  ]
}
```

**Usage:** For metric evaluation at each timepoint

## 5. Training Loop Structure

### Main Loop in `train_debugV2.py` (lines 199-213)

```python
for epoch in range(start_epoch, num_epochs + 1):
    # Forward pass and backprop
    log_dict = trainer.forward_epoch(epoch)
    
    # Log to wandb
    wandb.log(log_dict)
    
    # Periodic checkpoints (every 5000 epochs)
    if epoch % 5000 == 0 or epoch == num_epochs - 1:
        trainer.save_model(epoch, experiment_directory)
    
    # Periodic metrics
    if epoch % 5000 == 0 or epoch == num_epochs - 1:
        metrics_dict = trainer.compute_and_save_metrics(epoch)
        wandb.log({
            "eval/mean_chamfer_distance": metrics_dict["mean_chamfer_distance"],
            "eval/std_chamfer_distance": metrics_dict["std_chamfer_distance"],
        })
```

### Forward Epoch in `multi_tools.py` (lines 742-838)

Trainer's `forward_epoch()` method implements one training iteration:

1. **Sample random frame:**
   ```python
   rand_scene_id = random.sample(set(range(self.start_idx, self.end_idx + 1)), 1)[0]
   src_dps = self.sdf_dataset[rand_scene_id]
   src_pts = src_dps["pts"].cuda()
   src_nms = src_dps["nms"].cuda()
   src_normed_time = src_dps["normed_time"].cuda()
   ```

2. **Adjustment network (if enabled):**
   ```python
   adjust_input_dps, ajust_losses, src_prob = self.get_pts(epoch)
   ```

3. **Forward pass through decoder:**
   ```python
   pred_pts = self.decoder(adjust_input_dps, self.template_time, requires_grad=True)
   ```

4. **Compute losses:**
   ```python
   recon_losses = self.recon_loss(pred_pts, src_prob)
   fit_loss, align_loss = self.compute_fit_loss(recon_losses, self.fit_weights)
   code_loss = self.compute_latents()
   total_loss = self.compute_total_loss(epoch, fit_loss, code_loss, align_loss, ajust_losses)
   ```

5. **Optimization step:**
   ```python
   total_loss.backward()
   torch.nn.utils.clip_grad_norm_(self.decoder.warper.parameters(), 0.8)
   self.optimizer_all.step()
   self.optimizer_all.zero_grad()
   ```

## 6. Configuration Files

### `configs/specs.json`

Controls training hyperparameters:

```json
{
  "NumEpochs": 15001,
  "CodeLength": 128,
  "SamplesPerScene": 5000,
  "ScenesPerBatch": 8,
  "ClampingDistance": 0.1,
  "CodeRegularizationLambda": 1e-4,
  "LearningRateSchedule": [
    {
      "Type": "Step",
      "Initial": 0.0005,
      "Interval": 500,
      "Factor": 0.5
    },
    ...
  ],
  "loss": [
    {
      "loss_type": "ReconLoss",
      "properties": {
        "recon_loss_weight": 1.0,
        "grad_loss_weight": 0.1,
        "grad_on_surface_weight": 0.1,
        "latent_reg_weight": 0.001
      }
    }
  ]
}
```

### `configs/data_splits.json`

Defines train/val/test split at plant level

### `configs/point_splits/*.json`

Per-sequence, per-timepoint point-level splits for evaluation

## 7. Important Implementation Details

### Normal Filtering

Points with small normal magnitudes are filtered:
```python
mask = np.linalg.norm(n, axis=-1) > 1e-5
x = v[mask].astype(np.float32)
n = n[mask].astype(np.float32)
n /= np.linalg.norm(n, axis=-1, keepdims=True)
```

**Why:** Small normals indicate problematic or invalid geometry

### Temporal Encoding

Time is normalized based on **actual timestamps in filtered sequence**:

```python
# Extract from filename D00, D01, ..., D##
timestamps = [extract_timestamp(f) for f in sub_files]
min_time = min(timestamps)
max_time = max(timestamps)
time_range = max_time - min_time if max_time > min_time else 1
normalized_time = (timestamp - min_time) / time_range
```

**Stored in:** `self.normalized_times[idx]` as `torch.Tensor`

### Leaf Tip Tracking

Optional leaf_tip mask stored in point cloud attributes:
```python
if "is_leaf_tip" in pcd_tensor.point:
    leaf_tip_mask = pcd_tensor.point.is_leaf_tip.numpy().flatten().astype(bool)
```

**Updated after filtering:** Mask indices adjusted to match deduplicated point cloud

### Bounding Box Computation

Computed in normalized space after all points aggregated:
```python
big_pc = torch.cat(self.train_xx, dim=0)  # All points, all timesteps
self.bb_min = big_pc.min(0, keepdim=True)[0]
self.bb_max = big_pc.max(0, keepdim=True)[0]
```

Used for:
- Mesh generation (marching cubes domain)
- SDF evaluation grid

## 8. Dataset-Specific Implementations

### Animals/Deforming4D (`data_wmw.py`)

```python
class DeformSamples:
    def __init__(self, filename, start_idx=0, gap=16, stride=1):
        pc_path = filename + "/4k_pc/"
        sub_files = sorted(sub_files)[start_idx:start_idx+gap:stride]
        
        # Load 4k point clouds
        # Sample 2000 points per frame at runtime
```

Key differences from Pheno4D:
- Fixed folder: "4k_pc"
- Runtime sampling: 2000 points per batch
- Simpler temporal encoding: `frame_idx / lens`

### D-FAUST (`data_dfaust.py`)

Similar to animals, but with human body sequences.

### Medical (`data_medician.py`)

Similar structure, adapted for medical imaging.

## 9. Metrics Computation

### Method: `compute_and_save_metrics()` (lines 882-945+)

```python
def compute_and_save_metrics(self, epoch):
    self.decoder.eval()
    metrics_calculator = PlantMetrics()
    
    for idx in range(self.start_idx, self.end_idx + 1):
        # Get ground truth
        gt_data = self.sdf_dataset[idx]
        gt_pts = gt_data["pts"]
        
        # Sample from SDF grid
        grid_size = int(num_samples ** (1/3)) + 1
        x = torch.linspace(-1.7, 1.7, grid_size)
        y = torch.linspace(-1.7, 1.7, grid_size)
        z = torch.linspace(-1.7, 1.7, grid_size)
        
        # Create meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid_pts = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        # Evaluate SDF in batches
        # Extract surface points where SDF ≈ 0
        # Compute Chamfer distance to ground truth
        chamfer_distances.append(cd)
    
    # Aggregate and save
    metrics_dict = {
        "mean_chamfer_distance": np.mean(chamfer_distances),
        "std_chamfer_distance": np.std(chamfer_distances),
    }
    
    # Save to JSON
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f)
    
    return metrics_dict
```

## 10. File Locations Summary

### Source Code Structure
```
/mnt/Data/jgajardo/code/CanFields/
├── train_debugV2.py          # Main training script
├── multi_tools.py            # Trainer classes, dataset loading
├── utils.py                  # Utilities
├── eval_metrics.py           # Metrics evaluation
├── eval_sphere.py            # Shape reconstruction
├── datasets/
│   ├── data_pheno4d.py       # Pheno4D dataset loader
│   ├── data_wmw.py           # Animals dataset loader
│   ├── data_dfaust.py        # D-FAUST dataset loader
│   ├── data_medician.py      # Medical dataset loader
│   └── prepare_pheno4d.py    # Data preprocessing
├── networks/
│   ├── multi_velocity_debug.py  # Main decoder
│   ├── other_volocities.py      # Alternative methods
│   ├── SDF.py                   # SDF network
│   └── modules.py               # Building blocks
├── losses/
│   ├── recon.py              # Reconstruction loss
│   ├── part_loss.py          # Part loss
│   └── kill_vec.py           # Regularization
├── metrics/
│   ├── metrics.py            # Metric computation
│   └── aggregate_results.py  # Results aggregation
└── configs/
    ├── specs.json            # Main training specs
    ├── others_specs.json     # Alternative method specs
    ├── data_splits.json      # Plant-level splits
    └── point_splits/         # Per-timepoint splits
```

### Data Directory Structure
```
datasets/
├── Pheno4D/processed/v2/organ_sequences/
│   ├── maize/
│   ├── sorghum/
│   ├── tobacco/
│   └── tomato/
│       └── {sequence_name}/
│           ├── sparse_pc/       # Sparse point clouds
│           └── dense_pc/        # Dense point clouds
├── Deforming4D/notexture/
│   └── {sequence_name}/
│       └── 4k_pc/               # Point clouds
└── [Other datasets]
```

### Output Directory Structure
```
results/
└── pheno4d/                     # Dataset
    └── ode_solve/               # Method
        └── maize/               # Crop
            └── sequence_name/   # Sequence
                └── timestamp_name/
                    ├── checkpoints/
                    │   ├── ModelParameters/
                    │   ├── LatentCodes/
                    │   ├── ShapeLatentCodes/
                    │   └── AdjustLatentCodes/
                    ├── metrics/
                    ├── meshes/
                    └── wandb/
```

## 11. Key Takeaways

1. **Entry Point:** `train_debugV2.py` → `velocity_test()`

2. **Dataset Loading:** 
   - `load_dataset()` in `multi_tools.py`
   - Calls appropriate `DeformSamples` class
   - For Pheno4D: `data_pheno4d.DeformSamples`

3. **Data Processing:**
   - Load PLY files from sparse_pc or dense_pc
   - Deduplicate and filter points
   - Estimate normals if needed
   - Normalize: zero-center and unit-scale
   - Temporal encoding: normalized to [0,1]

4. **Training:**
   - Random frame sampling each epoch
   - Forward pass through decoder
   - Loss computation and backprop
   - Periodic metrics evaluation

5. **Data Splits:**
   - Plant-level: Val (4) vs Test (8)
   - Point-level: 50/50 train/test per timepoint
   - Seeds: reproducible with random_seed=42

