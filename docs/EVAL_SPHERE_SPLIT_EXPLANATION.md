# eval_sphere.py and Point Splits - Technical Explanation

## Question: Why doesn't eval_sphere.py use point splits?

Good question! Here's the detailed explanation:

## What eval_sphere.py Does

`eval_sphere.py` generates predicted meshes by:

**Step 1** (`-s 1`): Create canonical template
- Queries the neural field on a dense 3D grid (200x200x200)
- Generates a mesh via marching cubes

**Step 2** (`-s 2`): Deform to each timestep
- Takes the canonical mesh
- Warps it to each timepoint using the trained velocity field
- Outputs one mesh per timepoint

## Why It Uses the Dataset

The dataset in eval_sphere is used for **metadata only**:

1. **Bounding box** (line 442): To know where to query the 3D grid
   ```python
   sw_pts = sdf_dataset.evaluate_train(int(start_idx + end_idx // 2))
   sw_bbox_min = sw_pts.min(0, keepdim=True)[0]
   sw_bbox_max = sw_pts.max(0, keepdim=True)[0]
   ```

2. **Normalization** (lines 488-489, 526-527): Scale and mean for denormalization
   ```python
   d_s = sdf_dataset.scale
   d_m = sdf_dataset.mean
   ```

3. **Time values** (line 482): What timepoints to generate
   ```python
   backward_time = sdf_dataset.normalized_times
   ```

## The Normalization Problem

**This is the key issue you identified!**

During **training**:
```python
# train_debugV2.py loads with splits
sdf_dataset = DeformSamples(..., split_file="configs/point_splits/...", use_train_points=True)

# Normalization computed in data_pheno4d.py lines 181-187:
big_pc = torch.cat(self.train_xx, dim=0)  # Only TRAIN points!
self.mean = big_pc.mean(0, keepdim=True)
self.scale = 1.0 / torch.max(torch.abs(big_pc - self.mean))
```

During **eval_sphere** (current code):
```python
# eval_sphere.py loads WITHOUT splits
sdf_dataset = DeformSamples(..., split_file=None, use_train_points=True)

# Normalization recomputed on ALL points
big_pc = torch.cat(self.train_xx, dim=0)  # ALL points!
self.mean = big_pc.mean(0, keepdim=True)  # Different!
self.scale = 1.0 / torch.max(torch.abs(big_pc - self.mean))  # Different!
```

**Result**: Scale and mean are DIFFERENT between training and evaluation!

## The Solution

The best solution is to **use the same normalization** in eval_sphere as was used in training.

### Option A: Pass split_file to eval_sphere (Recommended)

Modify eval_sphere to accept and use the same split file:

```python
# In eval_sphere.py line 290:
# Auto-detect split file (same as train_debugV2.py)
split_file = None
if args.dataset_name == "pheno4d":
    sequence_name = args.object_name.split("/")[-1]
    split_file = f"configs/point_splits/{sequence_name}_split.json"
    if not os.path.exists(split_file):
        logging.warning(f"Split file not found: {split_file}")
        split_file = None

sdf_dataset = data_pheno4d.DeformSamples(
    specs["ShapePath"],
    start_idx=start_idx,
    gap=gap,
    train_step=stride,
    use_dense=False,
    split_file=split_file,  # Same as training!
    use_train_points=True,   # Use train points for normalization
)
```

**Why this works:**
- ✅ Normalization computed on same train points as training
- ✅ Bounding box computed on train points (model was trained on these regions)
- ✅ Consistent with training

**Potential issue:**
- ❌ Bounding box might not cover full plant if train points are spatially biased

However, for the Pheno4D dataset with 50/50 random split, this shouldn't be an issue.

### Option B: Load all points but force same normalization (Complex)

Save `scale` and `mean` during training, load them in eval_sphere. This requires:
1. Saving normalization params to experiment directory during training
2. Loading them in eval_sphere instead of recomputing
3. More code changes

### Option C: Accept the difference (Not Recommended)

The normalization difference is small for 50/50 random splits, but it's still inconsistent.

## Recommendation

**Implement Option A** - it's simple and ensures consistency with training.

Let me implement this now...
