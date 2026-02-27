# Refactoring Plan for eval_metrics.py

## Current State Analysis

**File Size**: 1,974 lines total
- `compute_metrics_for_epoch()`: **981 lines** (50% of the file!)
- Other functions: ~993 lines

### Problem: Massive Function with Heavy Code Duplication

The `compute_metrics_for_epoch()` function is extremely long and has significant code duplication due to handling train/test splits separately throughout.

## Breakdown of `compute_metrics_for_epoch()` (981 lines)

| Section | Lines | Issue |
|---------|-------|-------|
| Setup (paths, splits, validation) | ~100 | Acceptable |
| **Load meshes loop** | ~250 | **HEAVY DUPLICATION** - separate logic for splits/no-splits |
| **Compute CD/NC** | ~100 | **DUPLICATION** - nearly identical code for train/test/all |
| **Compute tip EPE (setup + compute)** | ~280 | **DUPLICATION** - loading decoder repeated, metrics formatting repeated |
| **Compute AccS/AccR** | ~180 | **DUPLICATION** - separate logic for splits/no-splits |
| Metadata (storage, runtime) | ~20 | Acceptable |

### Code Duplication Examples

1. **Mesh Loading** (lines 984-1235):
   - If `point_splits`: One complete set of logic (~125 lines)
   - Else: Nearly identical logic for no-splits (~125 lines)
   - **Result**: ~250 lines doing essentially the same thing twice

2. **CD/NC Computation** (lines 1236-1345):
   - Train CD/NC: ~35 lines
   - Test CD/NC: ~35 lines
   - No-split CD/NC: ~35 lines
   - **Same computation pattern repeated 3 times**

3. **Tip EPE Metrics** (lines 1346-1625):
   - Setup decoder (lines 1346-1388): ~42 lines
   - Train tip metrics (lines 1410-1444): ~35 lines
   - Test tip metrics (lines 1446-1477): ~32 lines
   - No-split setup + metrics (lines 1524-1655): ~130 lines
   - **Same decoder setup and metrics formatting repeated**

4. **AccS/AccR Computation** (lines 1626-1820):
   - Split case: ~100 lines
   - No-split case: ~80 lines
   - **Same computation logic duplicated**

## Proposed Refactoring Strategy

### Goal
Reduce `compute_metrics_for_epoch()` from **981 lines to ~200-300 lines** by extracting logical components and eliminating duplication.

### Expected Savings
- Extract 5 new functions: ~630 lines moved out
- Eliminate duplication: ~150 lines removed
- **Final size**: ~200-250 lines

---

## Refactoring Phases

### Phase 1: Mesh Loading & CD/NC (Quick Wins)

#### 1.1. Extract `load_and_prepare_meshes()`
**Purpose**: Load all predicted and ground truth meshes, apply splits uniformly, return structured data.

**Signature**:
```python
def load_and_prepare_meshes(
    pred_files: List[str],
    gt_paths: List[str],
    pred_mesh_dir: str,
    point_splits: Optional[dict],
    original_indices: List[int],
    num_samples: int,
    compute_normals: bool
) -> dict:
    """
    Load and prepare meshes for metric computation.

    Returns:
        {
            "all": {
                "pred_points": List[Tensor],
                "gt_points": List[Tensor],
                "pred_normals": List[Tensor] or None,
                "gt_normals": List[Tensor] or None,
            },
            "train": {...} or None,  # Only if point_splits provided
            "test": {...} or None,   # Only if point_splits provided
        }
    """
```

**Lines saved**: ~150 lines (eliminates split/no-split duplication in loading loop)

**Current code location**: Lines 984-1235

---

#### 1.2. Extract `compute_chamfer_and_normal_metrics()`
**Purpose**: Compute CD and NC for a set of point clouds. Works for any split (train/test/all).

**Signature**:
```python
def compute_chamfer_and_normal_metrics(
    pred_points_list: List[Tensor],
    gt_points_list: List[Tensor],
    pred_normals_list: Optional[List[Tensor]] = None,
    gt_normals_list: Optional[List[Tensor]] = None,
    skip_first_frame: bool = True
) -> dict:
    """
    Compute Chamfer Distance and Normal Consistency metrics.

    Returns:
        {
            "mean_chamfer_distance": float,
            "std_chamfer_distance": float,
            "chamfer_distances_per_timestep": List[float],
            "mean_normal_consistency": float or None,
            "std_normal_consistency": float or None,
            "normal_consistency_per_timestep": List[float] or None,
        }
    """
```

**Lines saved**: ~70 lines (eliminates train/test/all duplication)

**Current code location**:
- Train: Lines 1236-1280
- Test: Lines 1281-1345
- No-split: Lines 1500-1545

---

### Phase 2: Tip Tracking Refactoring

#### 2.1. Extract `setup_tip_tracking()`
**Purpose**: Load specs, dataset, and decoder model for tip tracking.

**Signature**:
```python
def setup_tip_tracking(
    args,
    epoch_num: int,
    point_splits: Optional[dict] = None,
    original_indices: Optional[List[int]] = None
) -> dict:
    """
    Setup decoder and dataset for tip tracking.

    Returns:
        {
            "decoder": Decoder,
            "scale": float,
            "mean": ndarray,
            "normalized_times": Tensor,
            "sdf_dataset": DeformSamples,
        }
    """
```

**Lines saved**: ~100 lines (eliminates repeated decoder loading)

**Current code location**:
- With splits: Lines 1350-1388
- No splits: Lines 1524-1580

---

#### 2.2. Extract `format_tip_epe_metrics()`
**Purpose**: Format tip EPE results into metrics_json structure.

**Signature**:
```python
def format_tip_epe_metrics(
    tip_epe_results: dict,
    split_name: str = "all"  # "all", "train", or "test"
) -> dict:
    """
    Format tip EPE tracking results into metrics dictionary.

    Args:
        tip_epe_results: Results from track_tips_with_warper()
        split_name: Which split to format ("all", "train", "test")

    Returns:
        {
            "num_timesteps_with_tips": int,
            "mean_tip_epe": float or None,
            "std_tip_epe": float or None,
            "tip_epe_per_timestep": List[float],
            "n_tips_per_timepoint": List[int],
            "total_tips_evaluated": int,
        }
    """
```

**Lines saved**: ~100 lines (eliminates train/test/all formatting duplication)

**Current code location**:
- Train: Lines 1410-1444
- Test: Lines 1446-1477
- No-split: Lines 1587-1655

---

### Phase 3: Cleanup & AccS/AccR

#### 3.1. Extract `compute_accuracy_metrics()`
**Purpose**: Compute AccS and AccR from warped tip coordinates.

**Signature**:
```python
def compute_accuracy_metrics(
    warped_coords_list: List[ndarray],
    gt_coords_list: List[ndarray],
    skip_first_frame: bool = True
) -> dict:
    """
    Compute accuracy metrics (AccS, AccR) from tip tracking results.

    Returns:
        {
            "mean_AccS": float,
            "std_AccS": float,
            "AccS_per_timestep": List[float],
            "mean_AccR": float,
            "std_AccR": float,
            "AccR_per_timestep": List[float],
        }
    """
```

**Lines saved**: ~100 lines (eliminates split/no-split duplication)

**Current code location**:
- Split case: Lines 1650-1730
- No-split case: Lines 1735-1820

---

## Refactored `compute_metrics_for_epoch()` Structure

**Before** (981 lines):
```python
def compute_metrics_for_epoch(args, epoch_num):
    # 100 lines: Setup
    # 250 lines: Load meshes with splits duplication
    # 100 lines: Compute CD/NC with train/test/all duplication
    # 280 lines: Setup + compute tip EPE with duplication
    # 180 lines: Compute AccS/AccR with split duplication
    # 20 lines: Metadata
    return metrics_json
```

**After** (~250 lines):
```python
def compute_metrics_for_epoch(args, epoch_num):
    """Compute metrics for a specific epoch."""

    # Setup (50 lines)
    experiment_directory = get_experiment_directory(args)
    pred_mesh_dir = get_prediction_directory(experiment_directory, epoch_num)
    point_splits = load_point_splits_if_needed(args)
    gt_paths, gt_files, original_indices = get_ground_truth_paths(args)
    pred_files = get_prediction_files(pred_mesh_dir)

    # Initialize metrics
    metrics_json = initialize_metrics_json(epoch_num, args, point_splits)

    # Load all meshes (1 line → 150 lines extracted)
    mesh_data = load_and_prepare_meshes(
        pred_files, gt_paths, pred_mesh_dir,
        point_splits, original_indices,
        args.num_samples, args.compute_normal_consistency
    )

    # Compute CD/NC metrics (10-15 lines → 100 lines extracted)
    if point_splits:
        metrics_json["train"].update(
            compute_chamfer_and_normal_metrics(
                mesh_data["train"]["pred_points"],
                mesh_data["train"]["gt_points"],
                mesh_data["train"]["pred_normals"],
                mesh_data["train"]["gt_normals"]
            )
        )
        metrics_json["test"].update(
            compute_chamfer_and_normal_metrics(
                mesh_data["test"]["pred_points"],
                mesh_data["test"]["gt_points"],
                mesh_data["test"]["pred_normals"],
                mesh_data["test"]["gt_normals"]
            )
        )
    else:
        metrics_json.update(
            compute_chamfer_and_normal_metrics(
                mesh_data["all"]["pred_points"],
                mesh_data["all"]["gt_points"],
                mesh_data["all"]["pred_normals"],
                mesh_data["all"]["gt_normals"]
            )
        )

    # Compute tip EPE (20-30 lines → 200 lines extracted)
    if args.compute_tip_epe:
        tracking_setup = setup_tip_tracking(
            args, epoch_num, point_splits, original_indices
        )

        tip_epe_results = track_tips_with_warper(
            tracking_setup["decoder"],
            gt_paths,
            tracking_setup["sdf_dataset"],
            tracking_setup["scale"],
            tracking_setup["mean"],
            tracking_setup["normalized_times"],
            point_splits,
            original_indices,
        )

        if point_splits:
            metrics_json["train"].update(
                format_tip_epe_metrics(tip_epe_results, "train")
            )
            metrics_json["test"].update(
                format_tip_epe_metrics(tip_epe_results, "test")
            )
        else:
            metrics_json.update(
                format_tip_epe_metrics(tip_epe_results, "all")
            )

        # Compute AccS/AccR (10 lines → 100 lines extracted)
        if point_splits:
            metrics_json["test"].update(
                compute_accuracy_metrics(
                    tip_epe_results["warped_coords_test"],
                    tip_epe_results["gt_coords_test"]
                )
            )
        else:
            metrics_json.update(
                compute_accuracy_metrics(
                    tip_epe_results["warped_coords_all"],
                    tip_epe_results["gt_coords_all"]
                )
            )

    # Add metadata (20 lines)
    metrics_json["storage_mb"] = compute_storage_size(
        experiment_directory, str(epoch_num)
    )
    metrics_json["runtime_minutes"] = get_runtime_from_wandb(
        experiment_directory
    )

    # Log summary
    log_metrics_summary(metrics_json, point_splits)

    return metrics_json
```

---

## Benefits of Refactoring

### 1. **Readability**
- Main function becomes a high-level workflow (20-30 lines of logic)
- Each extracted function has a single, clear purpose
- Easier to understand what metrics are being computed

### 2. **Maintainability**
- Changes to CD computation only need to happen in one place
- No risk of forgetting to update train/test/all branches
- Bug fixes apply uniformly across all splits

### 3. **Testability**
- Each extracted function can be unit tested independently
- Mock data is easier to create for smaller functions
- Easier to verify correctness of individual components

### 4. **Reusability**
- `compute_chamfer_and_normal_metrics()` could be used in other scripts
- `format_tip_epe_metrics()` makes it easy to add new split types
- Functions can be moved to a separate `metrics_utils.py` module

### 5. **Performance**
- No performance impact (same operations, just reorganized)
- Potential for future parallelization of independent computations

---

## Implementation Order

### Step 1: Extract Mesh Loading
- Create `load_and_prepare_meshes()`
- Test with existing data
- **Lines reduced**: 981 → 830

### Step 2: Extract CD/NC Computation
- Create `compute_chamfer_and_normal_metrics()`
- Update main function to call it
- **Lines reduced**: 830 → 760

### Step 3: Extract Tip Tracking Setup
- Create `setup_tip_tracking()`
- Test decoder loading works correctly
- **Lines reduced**: 760 → 660

### Step 4: Extract Tip EPE Formatting
- Create `format_tip_epe_metrics()`
- Verify metrics match previous output
- **Lines reduced**: 660 → 560

### Step 5: Extract AccS/AccR
- Create `compute_accuracy_metrics()`
- Test with tracked tip coordinates
- **Lines reduced**: 560 → 460

### Step 6: Final Cleanup
- Add helper functions for setup/logging
- Clean up variable names
- Add docstrings
- **Final size**: ~250 lines

---

## Validation Strategy

After each refactoring step:

1. **Run on test sequence**: Ensure metrics match exactly
2. **Compare JSON outputs**: Diff before/after results
3. **Check logs**: Verify all logging statements still work
4. **Profile performance**: Ensure no slowdown

**Test command**:
```bash
python eval_metrics.py \
    --object_name Pheno4D/processed/v2/organ_sequences/tomato/tomato2_control_plant2 \
    --experiment_name test_experiment \
    --dataset_name pheno4d \
    --epoch 15000 \
    --compute_tip_epe
```

**Validation**:
```python
import json
import numpy as np

# Load before/after results
with open('metrics_before.json') as f:
    before = json.load(f)
with open('metrics_after.json') as f:
    after = json.load(f)

# Compare all numeric values (allow small floating point differences)
def compare_dicts(d1, d2, path=""):
    for key in d1:
        if isinstance(d1[key], dict):
            compare_dicts(d1[key], d2[key], f"{path}.{key}")
        elif isinstance(d1[key], (int, float)):
            assert np.isclose(d1[key], d2[key], rtol=1e-5), \
                f"Mismatch at {path}.{key}: {d1[key]} != {d2[key]}"

compare_dicts(before, after)
print("✓ All metrics match!")
```

---

## Future Improvements (Optional)

### 1. Create `MetricsAccumulator` Class
Instead of separate train/test/all dictionaries, use a unified accumulator:

```python
class MetricsAccumulator:
    def __init__(self, splits=None):
        self.splits = ["train", "test"] if splits else ["all"]
        self.data = {split: {} for split in self.splits}

    def add_metric(self, metric_name, values_dict):
        """Add a metric for all splits at once."""
        for split in self.splits:
            if split in values_dict:
                self.data[split][metric_name] = values_dict[split]

    def to_dict(self):
        return self.data
```

**Usage**:
```python
accumulator = MetricsAccumulator(point_splits)
accumulator.add_metric("chamfer_distance", cd_results)
accumulator.add_metric("tip_epe", tip_results)
metrics_json = accumulator.to_dict()
```

### 2. Separate Module: `metrics_utils.py`
Move all extracted functions to a separate module:
- Cleaner namespace
- Easier to import in other scripts
- Better organization

### 3. Config-based Metrics Selection
```python
METRICS_CONFIG = {
    "chamfer_distance": True,
    "normal_consistency": True,
    "tip_epe": True,
    "accs_accr": True,
}

for metric_name, enabled in METRICS_CONFIG.items():
    if enabled:
        compute_and_add_metric(metric_name, ...)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Regression in metrics | Low | High | Comprehensive validation with test sequences |
| Performance degradation | Very Low | Medium | Profile before/after, optimize if needed |
| Breaking existing scripts | Low | High | Keep function signature compatible, add deprecation warnings |

---

## Timeline Estimate

- **Phase 1** (Mesh + CD/NC): 2-3 hours
- **Phase 2** (Tip tracking): 2-3 hours
- **Phase 3** (AccS/AccR): 1-2 hours
- **Testing & Validation**: 2-3 hours
- **Total**: ~8-12 hours

---

## Success Criteria

✅ `compute_metrics_for_epoch()` reduced to < 300 lines
✅ No code duplication for train/test/all splits
✅ All metrics match previous output (< 1e-5 difference)
✅ Each extracted function has clear purpose and docstring
✅ Test coverage for new functions
✅ Performance unchanged (± 5%)

---

## Questions to Resolve

1. Should we create a separate `metrics_utils.py` module or keep functions in `eval_metrics.py`?
2. Do we want to refactor `track_tips_with_warper()` as well (it's also quite long at ~317 lines)?
3. Should we add type hints throughout the refactored code?
4. Do we want to add progress bars for long-running operations?

---

**Document Version**: 1.0
**Date**: 2025-12-14
**Author**: Refactoring Analysis
