# TODO: Fix Leaf Tip Metrics to Match DOMA-release

## Context

This document describes changes needed to `eval_metrics.py` to ensure leaf tip metrics are computed consistently with DOMA-release, allowing direct comparison of baseline results.

## Current State Comparison

### CanFields (`eval_metrics.py`)
- **Tip matching**: Positional (first N tips matched by array index)
- **Tip tracking**: `tracked_tips[tip_id]` dict with first appearance info
- **First appearance**: SKIPPED from EPE computation (line 716: `if t_idx != first_appear_idx`)
- **N_tips metric**: NOT saved to JSON (only logged to console)
- **Organ IDs**: NOT loaded (only `is_leaf_tip` boolean)

### DOMA-release (`compute_fullplant_metrics.py`)
- **Tip matching**: By organ ID (using `tip_organ_label` attribute)
- **Tip tracking**: Dynamic dict `tips_by_organ[organ_id]`
- **First appearance**: SKIPPED at save time (not in PLY files) - tips added to tracking but not saved until next frame
- **N_tips metric**: SAVED to CSV with per-timepoint counts
- **Organ IDs**: LOADED from `tip_organ_label` attribute in PLY files

## Problem: Positional Matching is Fragile

**Example showing the issue:**

Frame 0: GT tips = [leaf_4]
Frame 1: GT tips = [leaf_1, leaf_2]  ← leaf_4 disappeared!
Frame 2: GT tips = [leaf_4, leaf_2]  ← leaf_4 reappears, leaf_1 disappeared

Current positional matching at Frame 2:
- Tracked tips: [tip_0=leaf_4, tip_1=leaf_1, tip_2=leaf_2]
- GT tips: [leaf_4, leaf_2]
- `num_tips_to_eval = min(3, 2) = 2`
- Matches: tip_0 ↔ GT[0]=leaf_4 ✅, tip_1 ↔ GT[1]=leaf_2 ❌ WRONG!

**Should match**: tip_0 ↔ leaf_4, tip_2 ↔ leaf_2

This happens because tips can appear/disappear in any order in real data (annotation gaps).

---

## Task 1: Add N_tips to JSON Output ✅ PRIORITY

**Goal**: Save the number of evaluated tips per timepoint to the metrics JSON file, so we can compute `sum(N_tips)` across sequences.

### Changes Needed

**Location**: Around line 840-900 in `eval_metrics.py` (where results are saved to JSON)

**Current code** (example structure):
```python
results = {
    "epe": {
        "mean": float(np.nanmean(epe_values_all)),
        "std": float(np.nanstd(epe_values_all)),
        # ... other stats
    },
    "cd": { ... },
    "nc": { ... },
}
```

**Add this**:
```python
# Compute N_tips per timepoint (tips evaluated, excluding first appearance)
n_tips_per_timepoint = []
for t_idx in range(len(epe_values_all)):
    if epe_values_all[t_idx] is not None:
        # Count tips that were evaluated at this timepoint
        # Need to track this in the main loop (see below)
        n_tips_per_timepoint.append(n_tips_evaluated[t_idx])
    else:
        n_tips_per_timepoint.append(0)

results["epe"]["n_tips_per_timepoint"] = n_tips_per_timepoint
results["epe"]["total_tips_evaluated"] = sum(n_tips_per_timepoint)
```

**In the main evaluation loop** (around line 720-760):
```python
# After computing tips_to_include
n_tips_evaluated = []  # Add at top of function

# ... inside the loop at line ~720
if len(tips_to_include) > 0:
    epe_per_tip_filtered = np.linalg.norm(
        warped_to_eval[tips_to_include] - gt_to_eval[tips_to_include],
        axis=1,
    )
    mean_epe = epe_per_tip_filtered.mean()

    # Track how many tips were evaluated
    n_tips_evaluated.append(len(tips_to_include))  # ADD THIS
else:
    mean_epe = None
    n_tips_evaluated.append(0)  # ADD THIS
```

**Verification**:
After making changes, check JSON output:
```bash
python eval_metrics.py --config <config> --compute_tip_epe
cat results/.../metrics_epoch_15000.json | grep -A5 '"epe"'
```

Should see:
```json
"epe": {
    "mean": 12.34,
    "std": 5.67,
    "n_tips_per_timepoint": [2, 3, 3, 4, 4, ...],
    "total_tips_evaluated": 42
}
```

---

## Task 2: Fix Positional Matching → Use Organ IDs 🔧 IMPORTANT

**Goal**: Match tips by organ ID instead of position, making it robust to tips appearing/disappearing in any order.

### Prerequisites

1. **Check if PLY files have `tip_organ_label`**:
   ```python
   import open3d as o3d
   pcd = o3d.t.io.read_point_cloud("path/to/gt.ply")
   print(pcd.point.keys())  # Should include 'tip_organ_label'
   ```

2. **If NOT present**: Need to regenerate PLY files with organ labels for tips
   - Check how DOMA saves tips: `utils/io_utils.py:save_ply_with_organ_labels()`
   - Ensures both `is_leaf_tip` and `tip_organ_label` are saved

### Changes to `load_leaf_tips_from_ply()`

**Current** (line 165-191):
```python
def load_leaf_tips_from_ply(ply_path: str) -> tuple:
    pcd = o3d.t.io.read_point_cloud(ply_path)
    points = pcd.point.positions.numpy()

    if "is_leaf_tip" in pcd.point:
        is_leaf_tip = pcd.point.is_leaf_tip.numpy().flatten().astype(bool)
        leaf_tip_idxs = np.where(is_leaf_tip)[0]
        leaf_tip_coords = points[leaf_tip_idxs]
        return leaf_tip_idxs, leaf_tip_coords
    else:
        return np.array([]), np.array([])
```

**Updated**:
```python
def load_leaf_tips_from_ply(ply_path: str) -> tuple:
    """
    Load leaf tip indices, coordinates, and organ labels from a PLY file.

    Returns:
        Tuple of (leaf_tip_indices, leaf_tip_coordinates, leaf_tip_organ_labels)
        If organ labels not available, third element is None
    """
    pcd = o3d.t.io.read_point_cloud(ply_path)
    points = pcd.point.positions.numpy()

    if "is_leaf_tip" in pcd.point:
        is_leaf_tip = pcd.point.is_leaf_tip.numpy().flatten().astype(bool)
        leaf_tip_idxs = np.where(is_leaf_tip)[0]
        leaf_tip_coords = points[leaf_tip_idxs]

        # Load organ labels if available
        leaf_tip_organ_labels = None
        if "tip_organ_label" in pcd.point:
            tip_organ_label = pcd.point.tip_organ_label.numpy().flatten()
            leaf_tip_organ_labels = tip_organ_label[leaf_tip_idxs]

        return leaf_tip_idxs, leaf_tip_coords, leaf_tip_organ_labels
    else:
        return np.array([]), np.array([]), None
```

### Changes to Tip Tracking Logic

**Current tracking** (line 610-625):
```python
tracked_tips = {}  # tip_id -> (first_appear_idx, first_appear_coord, first_appear_time)

for t_idx, gt_path in enumerate(gt_paths):
    _, gt_tip_coords = load_leaf_tips_from_ply(gt_path)
    all_gt_tips.append(gt_tip_coords)

    if len(gt_tip_coords) > len(tracked_tips):
        num_new_tips = len(gt_tip_coords) - len(tracked_tips)
        for i in range(num_new_tips):
            new_tip_id = len(tracked_tips)
            new_tip_coord = gt_tip_coords[new_tip_id]
            tracked_tips[new_tip_id] = (t_idx, new_tip_coord, normalized_times[t_idx])
```

**Updated tracking** (organ ID based):
```python
tracked_tips = {}  # organ_id -> (first_appear_idx, first_appear_coord, first_appear_time)
all_gt_tips = []  # List of (coords, organ_labels) tuples
organ_id_available = True  # Flag to check if organ IDs are available

for t_idx, gt_path in enumerate(gt_paths):
    _, gt_tip_coords, gt_tip_organ_labels = load_leaf_tips_from_ply(gt_path)

    if gt_tip_organ_labels is None:
        logging.warning("No tip_organ_label found - falling back to positional matching")
        organ_id_available = False
        all_gt_tips.append((gt_tip_coords, None))
        # Continue with old logic...
        continue

    all_gt_tips.append((gt_tip_coords, gt_tip_organ_labels))

    # Track tips by organ ID
    for tip_coord, organ_id in zip(gt_tip_coords, gt_tip_organ_labels):
        if organ_id not in tracked_tips:
            # New tip appearing
            tracked_tips[organ_id] = (t_idx, tip_coord, normalized_times[t_idx])
```

### Changes to Evaluation Loop

**Current evaluation** (line 700-710):
```python
num_tips_to_eval = min(num_warped_tips, num_gt_tips)

if num_tips_to_eval > 0:
    warped_to_eval = warped_tip_coords[:num_tips_to_eval]
    gt_to_eval = gt_tip_coords[:num_tips_to_eval]

    tips_to_include = []
    for tip_id in range(num_tips_to_eval):
        if tip_id in tracked_tips:
            first_appear_idx, _, _ = tracked_tips[tip_id]
            if t_idx != first_appear_idx:
                tips_to_include.append(tip_id)
```

**Updated evaluation** (organ ID matching):
```python
gt_tip_coords, gt_tip_organ_labels = all_gt_tips[t_idx]

if gt_tip_organ_labels is not None:
    # Match tips by organ ID
    common_organ_ids = set(tracked_tips.keys()) & set(gt_tip_organ_labels)

    warped_to_eval = []
    gt_to_eval = []
    tips_to_include = []

    for organ_id in sorted(common_organ_ids):
        first_appear_idx, _, _ = tracked_tips[organ_id]

        # Find GT tip with this organ_id
        gt_idx = np.where(gt_tip_organ_labels == organ_id)[0]
        if len(gt_idx) == 0:
            continue
        gt_tip_coord = gt_tip_coords[gt_idx[0]]

        # Find warped tip with this organ_id
        # warped_tip_coords is ordered by sorted(tracked_tips.keys())
        warped_organ_ids = sorted(tracked_tips.keys())
        warped_idx = warped_organ_ids.index(organ_id)
        warped_tip_coord = warped_tip_coords[warped_idx]

        warped_to_eval.append(warped_tip_coord)
        gt_to_eval.append(gt_tip_coord)

        # Skip if first appearance
        if t_idx != first_appear_idx:
            tips_to_include.append(len(warped_to_eval) - 1)

    warped_to_eval = np.array(warped_to_eval)
    gt_to_eval = np.array(gt_to_eval)
else:
    # Fallback to positional matching
    # ... (keep old code)
```

---

## Testing & Verification

### Step 1: Verify N_tips is Saved
```bash
python eval_metrics.py --config configs/test_config.yaml --compute_tip_epe
python -c "import json; d=json.load(open('results/.../metrics.json')); print('Total tips:', d['epe']['total_tips_evaluated'])"
```

### Step 2: Compare with DOMA
In DOMA repo:
```bash
python scripts/compute_fullplant_metrics.py --experiment_dir results/fullplant/...
# Check CSV: sum up n_tips column
```

In CanFields repo:
```bash
# Sum up n_tips from JSON
python -c "import json; d=json.load(open('results/.../metrics.json')); print(sum(d['epe']['n_tips_per_timepoint']))"
```

**These should match** (or be very close if first-appearance handling differs).

### Step 3: Verify Organ ID Matching Works
```bash
# Enable debug logging
python eval_metrics.py --config ... --compute_tip_epe --log_level DEBUG
# Check logs for "Matched X tips by organ ID" messages
```

---

## Migration Strategy

1. **Phase 1** (Quick): Add N_tips to JSON (Task 1)
   - Test on a few sequences
   - Compare totals with DOMA

2. **Phase 2** (If needed): Add organ ID matching (Task 2)
   - Only if sum(N_tips) differs significantly
   - Requires organ labels in PLY files
   - More robust but bigger change

---

## Notes

- **First appearance handling**: Both skip first appearance, but differently
  - **CanFields**: Saves predictions but excludes from EPE computation (line 716)
  - **DOMA**: Doesn't save predictions at first appearance (skipped during PLY save)
  - **Result**: Both effectively exclude first appearance from metrics
  - **Implementation**: DOMA captures `organs_warped` BEFORE adding new tips, so new tips aren't saved until they've been warped once

- **Organ ID availability**: Check if your baseline PLY files have `tip_organ_label`
  - If not, you may need to regenerate predictions with updated code

- **Backward compatibility**: Keep positional matching as fallback for old results

---

## Questions to Resolve

1. Do current baseline PLY files have `tip_organ_label` attribute?
2. Should CanFields also skip first appearance to match DOMA?
3. What's an acceptable tolerance for sum(N_tips) differences?
