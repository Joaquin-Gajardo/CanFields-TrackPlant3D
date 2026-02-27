# Aggregate Results Update - Data Split Support

## Summary

Updated `metrics/aggregate_results.py` to support calculating metrics and tables based on val sequences, test sequences, or all sequences (default behavior).

## Changes

### 1. Created Data Splits Configuration File

**File:** `configs/data_splits.json`

This file defines which sequences belong to val and test splits:

```json
{
  "description": "Data splits for train/val/test sequences",
  "splits": {
    "val": [
      "maize_control_plant1",
      "sorghum_control_plant1",
      "tobacco_control_plant1",
      "tomato1_heat_plant3"
    ],
    "test": [
      "maize_control_plant2",
      "maize_control_plant3",
      "sorghum_highlight_plant2",
      "sorghum_highlight_plant3",
      "tobacco_control_plant2",
      "tobacco_shade_plant3",
      "tomato1_control2_plant2",
      "tomato2_control_plant3"
    ]
  }
}
```

You can customize this file to define your own val/test splits.

### 2. New Functions Added

#### `load_data_splits(config_path)`
- Loads the data splits configuration from a JSON file
- Returns a dictionary with 'val' and 'test' sequence lists
- Handles missing files gracefully with warnings

#### `filter_by_data_split(results, split_type, splits_config)`
- Filters experiment results by data split (val/test/all)
- **Supports both plant and organ sequences:**
  - If config contains `maize_control_plant1`, it matches:
    - `maize_control_plant1` (exact match - whole plant)
    - `maize_control_plant1_leaf1` (prefix match - organ)
    - `maize_control_plant1_leaf2` (prefix match - organ)
- Takes a list of results and returns only those matching the specified split
- Returns all results if split_type is "all" or if config is not available

### 3. New Command-Line Arguments

#### `--split {all,val,test}`
- **Default:** `all`
- Specifies which data split to use for aggregation
- Options:
  - `all`: Use all sequences (default behavior)
  - `val`: Use only validation sequences
  - `test`: Use only test sequences

#### `--splits_config PATH`
- **Default:** `configs/data_splits.json`
- Path to the data splits configuration file
- Allows using different split configurations

### 4. Updated Output File Naming

Output files now include the split type in their names:
- CSV: `{prefix}_{split}_{filter}_epoch{epoch}{suffix}.csv`
- LaTeX: `{prefix}_method_by_crop_{split}_{filter}_epoch{epoch}{suffix}.tex`

Examples:
- `results_val_epoch15000.csv` - Val split results
- `results_method_by_crop_test_plants_epoch15000.tex` - Test split, plants only

## Usage Examples

### 1. Calculate metrics for all sequences (default behavior)
```bash
python metrics/aggregate_results.py --epoch 15000
```

### 2. Calculate metrics for validation sequences only
```bash
python metrics/aggregate_results.py --epoch 15000 --split val
```

### 3. Calculate metrics for test sequences only
```bash
python metrics/aggregate_results.py --epoch 15000 --split test
```

### 4. Export val split results to CSV and LaTeX
```bash
python metrics/aggregate_results.py \
    --epoch 15000 \
    --split val \
    --export_csv \
    --export_latex \
    --output_prefix results
```

### 5. Combine with existing filters (e.g., plants only, test split)
```bash
python metrics/aggregate_results.py \
    --epoch 15000 \
    --split test \
    --filter plants \
    --export_csv
```

### 6. Use a custom splits configuration
```bash
python metrics/aggregate_results.py \
    --epoch 15000 \
    --split val \
    --splits_config /path/to/custom_splits.json
```

### 7. Works with organ sequences automatically
```bash
# Config has: "maize_control_plant1"
# This will match BOTH:
#   - maize_control_plant1 (plant)
#   - maize_control_plant1_leaf1 (organ)
#   - maize_control_plant1_leaf2 (organ)
#   - etc.

python metrics/aggregate_results.py \
    --epoch 15000 \
    --split test \
    --filter organs
```

## Backwards Compatibility

All changes are backwards compatible:
- Default behavior (`--split all`) processes all sequences as before
- Existing command-line arguments work unchanged
- If splits config is missing, the script falls back to using all sequences with a warning

## Implementation Details

### Filtering Order

The filtering happens in the following order:
1. Collect all experiment results
2. **Filter by data split** (new)
3. Filter by sequence type (organs/plants) if requested
4. Aggregate and export results

This ensures that both filters can be combined as needed.

### How Plant/Organ Matching Works

The data split filter intelligently handles both plant and organ sequences:

**Config file** (`configs/data_splits.json`):
```json
{
  "splits": {
    "val": ["maize_control_plant1"],
    "test": ["maize_control_plant2"]
  }
}
```

**What gets matched:**

For `--split val`:
- ✅ `maize_control_plant1` (exact match - whole plant)
- ✅ `maize_control_plant1_leaf1` (prefix match - individual leaf)
- ✅ `maize_control_plant1_leaf2` (prefix match - individual leaf)
- ❌ `maize_control_plant2` (different plant - in test split)

For `--split test`:
- ✅ `maize_control_plant2` (exact match)
- ✅ `maize_control_plant2_leaf1` (prefix match)
- ❌ `maize_control_plant1` (different plant - in val split)

**Combined with `--filter` option:**

```bash
# Get only val split + only organ sequences
python metrics/aggregate_results.py --split val --filter organs

# This will match:
# ✅ maize_control_plant1_leaf1 (in val split AND is organ)
# ❌ maize_control_plant1 (in val split but NOT an organ)
```

The prefix matching ensures you only need to specify the plant name in your config, and all associated organ sequences will be automatically included.
