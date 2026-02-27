# Split Filtering Guide for CanFields

## Overview

The updated `run_crop_sequences.sh` now supports filtering sequences by val/test split, making it easy to run experiments on specific data splits.

## Usage

```bash
bash run_crop_sequences.sh <crop> [method] [type] [split]
```

### Arguments

1. **crop** (required): `maize`, `sorghum`, `tobacco`, `tomato`
2. **method** (optional, default: `ode_solve`): `ode_solve`, `4DSDF`, `TPSR`, `NVFI`, `OF4D`
3. **type** (optional, default: `plant`): `plant`, `leaf`
4. **split** (optional, default: `test`): `test`, `val`, `all`

### Examples

```bash
# Run test sequences (default) - this is what you typically want!
bash run_crop_sequences.sh maize ode_solve leaf test

# Run val sequences for validation
bash run_crop_sequences.sh maize ode_solve leaf val

# Run all sequences (val + test)
bash run_crop_sequences.sh maize ode_solve leaf all

# Using defaults (plant sequences, test split)
bash run_crop_sequences.sh maize

# Different method
bash run_crop_sequences.sh tomato 4DSDF leaf test
```

## Preview Sequences

Before running, you can preview which sequences will be processed:

```bash
bash preview_sequences.sh <crop> [type] [split]
```

### Examples

```bash
# Preview maize test sequences
bash preview_sequences.sh maize leaf test

# Preview tomato val sequences
bash preview_sequences.sh tomato leaf val

# Preview all sorghum sequences
bash preview_sequences.sh sorghum leaf all
```

**Output:**
```
=================================================================
Preview: maize leaf sequences (split: test)
=================================================================

Found 11 sequences:
  Leaves: 9
  Stems: 2

Sequences to run:
  1. maize_control_plant2_leaf1
  2. maize_control_plant2_leaf2
  ...
  11. maize_control_plant3_stem

Command to run:
  bash run_crop_sequences.sh maize ode_solve leaf test
```

## Split Statistics

### Val Split (24 sequences from 4 plants)
- **Maize**: 5 sequences (4 leaves + 1 stem) from plant1
- **Sorghum**: 4 sequences (3 leaves + 1 stem) from plant1
- **Tobacco**: 7 sequences (6 leaves + 1 stem) from plant2
- **Tomato**: 8 sequences (7 leaves + 1 stem) from plant3

### Test Split (49 sequences from 8 plants) ← **DEFAULT**
- **Maize**: 11 sequences (9 leaves + 2 stems) from plant2 & plant3
- **Sorghum**: 11 sequences (9 leaves + 2 stems) from plant2 & highlight_plant2
- **Tobacco**: 14 sequences (12 leaves + 2 stems) from plant1 & shade_plant3
- **Tomato**: 13 sequences (11 leaves + 2 stems) from heat_plant3 & shade_plant1

### All Sequences (73 total)
- **Maize**: 16 sequences
- **Sorghum**: 15 sequences
- **Tobacco**: 21 sequences
- **Tomato**: 21 sequences

## Recommended Workflow

### Step 1: Preview sequences
```bash
bash preview_sequences.sh maize leaf test
```

### Step 2: Run experiments on test split (default)
```bash
# Test split is the default - use for final evaluation
bash run_crop_sequences.sh maize ode_solve leaf test
bash run_crop_sequences.sh sorghum ode_solve leaf test
bash run_crop_sequences.sh tobacco ode_solve leaf test
bash run_crop_sequences.sh tomato ode_solve leaf test
```

### Step 3: (Optional) Run on val split for validation
```bash
# Use val split for hyperparameter tuning or validation
bash run_crop_sequences.sh maize ode_solve leaf val
```

## Common Use Cases

### 1. Run all test sequences (for final evaluation)
```bash
for crop in maize sorghum tobacco tomato; do
    echo "Running $crop test sequences..."
    bash run_crop_sequences.sh $crop ode_solve leaf test
done
```

### 2. Run val sequences only (for debugging/validation)
```bash
for crop in maize sorghum tobacco tomato; do
    echo "Running $crop val sequences..."
    bash run_crop_sequences.sh $crop ode_solve leaf val
done
```

### 3. Quick test on single crop val split
```bash
# Fastest - only 5 sequences
bash run_crop_sequences.sh maize ode_solve leaf val
```

### 4. Full baseline comparison (test split only)
```bash
# Run CanFields on all test sequences for all crops
bash run_crop_sequences.sh maize ode_solve leaf test
bash run_crop_sequences.sh sorghum ode_solve leaf test
bash run_crop_sequences.sh tobacco ode_solve leaf test
bash run_crop_sequences.sh tomato ode_solve leaf test

# Then aggregate results
python metrics/aggregate_results.py --method ode_solve --split test
```

## Important Notes

1. **Default is test split**: If you don't specify a split, it defaults to `test` (the 49 sequences)
2. **Val split is for validation**: Use val (24 sequences) for hyperparameter tuning
3. **All split runs everything**: Use `all` (73 sequences) only if you need both val+test
4. **Test split is for final evaluation**: Use test (49 sequences) for comparing with DOMA

## Comparison with DOMA

Your DOMA method likely uses the same split logic. You can now run CanFields on the exact same sequences:

**DOMA:**
```bash
cd ../DOMA-release
python experiments/run_fullplant_batch.py --split test
```

**CanFields (equivalent):**
```bash
cd CanFields
bash run_crop_sequences.sh maize ode_solve leaf test
# ... repeat for other crops
```

Both will now process the exact same 49 test sequences!

## Troubleshooting

### "No sequences found for split"
- Check that `configs/data_splits.json` exists and has the correct split definitions
- Verify the crop name is correct (lowercase: maize, sorghum, tobacco, tomato)

### "Point split file not found"
- You need to generate point splits for the organ sequences
- See `SEQUENCE_LISTS_UPDATE.md` for instructions on generating splits

### Preview shows wrong sequences
- Check that `configs/data_splits.json` matches your expectations
- Verify you're using the correct split parameter (test/val/all)
