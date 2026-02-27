# Dataset Discrepancy Note

## Issue

The `configs/data_splits.json` file lists **73 sequences** (61 leaves + 12 stems), but your actual dataset at `datasets/Pheno4D/processed/v2/organ_sequences/` only has **44 sequences** (leaves only, no stems).

## What's Available

### Actual Dataset (44 sequences)
- Maize: 10 sequences (leaves only)
- Sorghum: 8 sequences (leaves only)
- Tobacco: 14 sequences (leaves only)
- Tomato: 12 sequences (leaves only)

### Config Lists (73 sequences)
- Includes stems (12 total)
- Includes some extra leaves not in your dataset
- Based on a newer/different version of the dataset

## Split Breakdown

### Val Split
- **Config**: 24 sequences
- **Available**: 15 sequences
- **Missing**: 9 sequences (mostly stems)

### Test Split
- **Config**: 49 sequences
- **Available**: 29 sequences
- **Missing**: 20 sequences (stems + some extra leaves)

### Available Test Sequences by Crop
- Maize: 7 (out of 11 in config)
- Sorghum: 5 (out of 11 in config)
- Tobacco: 9 (out of 14 in config)
- Tomato: 8 (out of 13 in config)

## Solution Implemented

The scripts now **automatically skip missing sequences** with a warning. This means:

✅ You can run the scripts without errors
✅ Only existing sequences will be processed
✅ You'll see warnings about skipped sequences

Example:
```bash
bash run_crop_sequences.sh maize ode_solve leaf test
```

Output will show:
```
Warning: Skipping missing sequence: maize_control_plant2_leaf5
Warning: Skipping missing sequence: maize_control_plant2_stem
...
Note: Skipped 4 missing sequences (e.g., stems not in dataset)
```

## Options Going Forward

### Option 1: Use What You Have (Recommended)
- Keep the current config as-is
- Scripts will automatically use only the 44 available sequences
- Easy and works immediately

### Option 2: Update Config to Match Dataset
Create a new `configs/data_splits_no_stems.json` with only the 44 sequences:

```json
{
  "val": {
    "sequences": [
      "maize_control_plant1_leaf1",
      "maize_control_plant1_leaf2",
      "maize_control_plant1_leaf3",
      "sorghum_control_plant1_leaf1",
      ...
    ]
  },
  "test": {
    "sequences": [
      "maize_control_plant2_leaf1",
      ...
    ]
  }
}
```

Then update scripts to use this file instead.

### Option 3: Download Updated Dataset
If stems are available in a newer version of Pheno4D/TrackPlant3D, you could update your dataset to include them.

## Checking What's Available

Use the preview script to see exactly what will run:

```bash
# Preview maize test sequences
bash preview_sequences.sh maize leaf test

# Shows:
# - Which sequences exist
# - Which are missing
# - Total counts
```

## Impact on Results

Since you have 29/49 test sequences (59%), you still have:
- Good coverage across all 4 crops
- Both control and treatment conditions
- Enough data for baseline comparison

The missing sequences are primarily:
- All 12 stems (not critical for leaf-based methods)
- Some extra leaves (not in original dataset)

Your results will still be valid for comparing CanFields vs DOMA on the **shared 29 test sequences**.

## Recommendation

**Use Option 1** - the current implementation works perfectly and will process all 44 available sequences. The config listing extra sequences is not a problem since the scripts now handle it gracefully.

When comparing with DOMA, make sure you're comparing on the **same 44 sequences** that actually exist in both datasets.
