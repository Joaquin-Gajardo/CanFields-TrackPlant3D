# Sequence Lists for CanFields Baseline Experiments

## Summary

The `run_crop_sequences.sh` script contains the sequences used for CanFields baseline experiments. These sequences **exactly match** the DOMA val+test splits for fair comparison.

## Current Sequences

### Plant Sequences (12 total)
- **Val (4 plants)**:
  - maize_control_plant1
  - sorghum_control_plant1
  - tobacco_control_plant2
  - tomato2_control_plant3

- **Test (8 plants)**:
  - maize_control_plant2
  - maize_control_plant3
  - sorghum_control_plant2
  - sorghum_highlight_plant2
  - tobacco_control_plant1
  - tobacco_shade_plant3
  - tomato1_heat_plant3
  - tomato1_shade_plant1

### Organ Sequences (73 total) 🆕 **INCLUDES STEMS**

**Distribution by crop:**
- Maize: 16 sequences (13 leaves + 3 stems)
- Sorghum: 15 sequences (12 leaves + 3 stems)
- Tobacco: 21 sequences (18 leaves + 3 stems)
- Tomato: 21 sequences (18 leaves + 3 stems)

**Split:**
- **Val: 24 sequences** (20 leaves + 4 stems from 4 plants)
- **Test: 49 sequences** (41 leaves + 8 stems from 8 plants) ← **This is the "49 organs"!**
- **Total: 73 sequences** (61 leaves + 12 stems)

## Point Splits Status

Currently available point splits (in `configs/point_splits/`):
- ✅ 12 plant sequences (all have splits)
- ⏳ 73 organ sequences (61 leaves + 12 stems - need to generate splits)

**Note**: The 12 plant-level point splits you currently have match the 12 plant sequences listed above.

## Generating Missing Point Splits

To generate point splits for the 73 organ sequences (61 leaves + 12 stems), use the script from your DOMA repo:

```bash
cd ../DOMA-release
python experiments/create_point_splits.py \
    --mode leaf \
    --sequences_file experiments/configs/data_splits.json \
    --output_dir ../CanFields/configs/point_splits/ \
    --train_ratio 0.5 \
    --random_seed 42
```

Or batch generate all:

```bash
cd ../DOMA-release
for seq in $(python -c "
import json
with open('experiments/configs/data_splits.json') as f:
    splits = json.load(f)
seqs = splits['splits']['val']['sequences'] + splits['splits']['test']['sequences']
print('\n'.join(seqs))
"); do
    echo "Generating split for $seq"
    python experiments/create_point_splits.py \
        --sequence_name $seq \
        --output_dir ../CanFields/configs/point_splits/ \
        --train_ratio 0.5 \
        --random_seed 42
done
```

## About the "49 Organs"

The **49 organs** refers to the **49 test sequences** in the updated splits:
- 41 test leaf sequences
- 8 test stem sequences
- Total: 49 test organ sequences

Combined with the 24 val sequences, there are **73 total organ sequences** (61 leaves + 12 stems).

These sequences are specifically chosen for:
1. **Train/val/test split consistency** (plant-level splitting)
2. **Good leaf tip coverage** (~78-79%)
3. **Diverse treatment conditions**

If you want to use a different set (e.g., all sequences with ≥5 timesteps), you would need to:
1. Define the selection criteria
2. Generate new point splits for those sequences
3. Update `run_crop_sequences.sh`

## Usage

Run all leaf sequences for a crop:
```bash
bash run_crop_sequences.sh maize ode_solve leaf
bash run_crop_sequences.sh sorghum ode_solve leaf
bash run_crop_sequences.sh tobacco ode_solve leaf
bash run_crop_sequences.sh tomato ode_solve leaf
```

Run plant sequences:
```bash
bash run_crop_sequences.sh maize ode_solve plant
```

## Validation

To verify sequences match updated splits:
```bash
python -c "
import json
with open('configs/data_splits.json') as f:
    splits = json.load(f)
val = splits['splits']['val']['sequences']
test = splits['splits']['test']['sequences']
leaves = [s for s in val+test if 'leaf' in s]
stems = [s for s in val+test if 'stem' in s]
print(f'Total: {len(val) + len(test)}')
print(f'Val: {len(val)}, Test: {len(test)}')
print(f'Leaves: {len(leaves)}, Stems: {len(stems)}')
"
```

Expected output:
```
Total: 73
Val: 24, Test: 49
Leaves: 61, Stems: 12
```

## Next Steps

1. ✅ **Sequences are correct** - 73 organ sequences (61 leaves + 12 stems) match updated splits
2. ⏳ **Generate point splits for organ sequences** - need splits for all 73 sequences
3. ✅ **Point splits for plant sequences** - already available (12 plants)
4. 🚀 **Ready to run experiments** once organ splits are generated

Once point splits are generated for all 73 organ sequences, you can run the full baseline comparison!
