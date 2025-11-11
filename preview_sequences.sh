#!/bin/bash
# Preview which sequences will be run with given parameters
# Usage: bash preview_sequences.sh <crop_name> [type] [split]
# Example: bash preview_sequences.sh maize leaf test

CROP=$1
TYPE=${2:-"plant"}
SPLIT=${3:-"test"}

if [ -z "$CROP" ]; then
    echo "Error: No crop specified"
    echo "Usage: bash preview_sequences.sh <crop_name> [type] [split]"
    echo "Available crops: maize, sorghum, tobacco, tomato"
    echo "Default type: plant, from options: plant, leaf"
    echo "Default split: test, from options: test, val, all"
    exit 1
fi

echo "================================================================="
echo "Preview: $CROP $TYPE sequences (split: $SPLIT)"
echo "================================================================="
echo ""

# Load data_splits.json and filter sequences
python3 << EOF
import json

# Load data_splits.json
with open('configs/data_splits.json') as f:
    splits = json.load(f)

crop = "$CROP"
split_type = "$SPLIT"
seq_type = "$TYPE"

# Get sequences for the split
if split_type == "all":
    sequences = splits['splits']['val']['sequences'] + splits['splits']['test']['sequences']
    plants = splits['splits']['val']['plants'] + splits['splits']['test']['plants']
else:
    sequences = splits['splits'][split_type]['sequences']
    plants = splits['splits'][split_type]['plants']

# Filter by crop and type
filtered = []
if seq_type == "plant":
    # Plant-level sequences
    for plant in plants:
        if plant.startswith(crop):
            filtered.append(plant)
else:
    # Leaf/organ sequences
    for seq in sequences:
        if seq.startswith(crop):
            filtered.append(seq)

# Count leaves vs stems
leaves = [s for s in filtered if 'leaf' in s]
stems = [s for s in filtered if 'stem' in s]

print(f"Found {len(filtered)} sequences:")
if seq_type == "leaf":
    print(f"  Leaves: {len(leaves)}")
    print(f"  Stems: {len(stems)}")
print()

print("Sequences to run:")
for i, seq in enumerate(filtered, 1):
    print(f"  {i}. {seq}")

print()
print(f"Command to run:")
print(f"  bash run_crop_sequences.sh {crop} ode_solve {seq_type} {split_type}")
EOF
