#!/bin/bash
# Run batch experiments using config files
# Usage: bash run_batch_experiments.sh <config_file> <crop> [type] [split] [sequences...]
# Example: bash run_batch_experiments.sh configs/experiments/interp_stride2.json maize leaf test
# Example: bash run_batch_experiments.sh configs/experiments/interp_stride2.json maize leaf test leaf1 leaf2

EXPERIMENT_CONFIG=$1
CROP=$2
TYPE=${3:-"leaf"}
SPLIT=${4:-"test"}

shift 4 2>/dev/null || shift $#
SPECIFIC_SEQUENCES="$@"

if [ -z "$EXPERIMENT_CONFIG" ] || [ -z "$CROP" ]; then
    echo "Usage: bash run_batch_experiments.sh <config_file> <crop> [type] [split] [sequences...]"
    echo ""
    echo "Arguments:"
    echo "  config_file: Path to experiment config JSON"
    echo "  crop: maize, sorghum, tobacco, tomato, all"
    echo "  type: leaf (default), plant"
    echo "  split: test (default), val, all"
    echo "  sequences: (optional) specific sequences to run"
    echo ""
    echo "Examples:"
    echo "  # All test leaves for maize"
    echo "  bash run_batch_experiments.sh configs/experiments/interp_stride2.json maize leaf test"
    echo ""
    echo "  # Specific sequences"
    echo "  bash run_batch_experiments.sh configs/experiments/interp_stride2.json maize leaf test \\"
    echo "    maize_control_plant1_leaf1 maize_control_plant1_leaf2"
    echo ""
    echo "Available configs:"
    ls -1 configs/experiments/*.json 2>/dev/null | sed 's/^/  /'
    exit 1
fi

if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "Error: Config file not found: $EXPERIMENT_CONFIG"
    echo ""
    echo "Available configs:"
    ls -1 configs/experiments/*.json 2>/dev/null | sed 's/^/  /'
    exit 1
fi

# Get experiment description from config
EXP_DESC=$(python3 -c "import json; print(json.load(open('$EXPERIMENT_CONFIG')).get('description', 'No description'))" 2>/dev/null)

echo "================================================================="
echo "BATCH EXPERIMENT RUN"
echo "================================================================="
echo "Config: $EXPERIMENT_CONFIG"
echo "Description: $EXP_DESC"
echo "Crop: $CROP"
echo "Type: $TYPE"
echo "Split: $SPLIT"
if [ -n "$SPECIFIC_SEQUENCES" ]; then
    echo "Specific sequences: $SPECIFIC_SEQUENCES"
fi
echo "================================================================="
echo ""

# Define sequences (simplified version - you can source from run_crop_sequences.sh)
declare -A CROP_LEAF_SEQUENCES
CROP_LEAF_SEQUENCES[maize]="
organ_sequences/maize/maize_control_plant1_leaf1
organ_sequences/maize/maize_control_plant1_leaf2
organ_sequences/maize/maize_control_plant1_leaf3
organ_sequences/maize/maize_control_plant1_leaf4
organ_sequences/maize/maize_control_plant2_leaf1
organ_sequences/maize/maize_control_plant2_leaf2
organ_sequences/maize/maize_control_plant2_leaf3
organ_sequences/maize/maize_control_plant2_leaf4
organ_sequences/maize/maize_control_plant2_leaf5
organ_sequences/maize/maize_control_plant3_leaf1
organ_sequences/maize/maize_control_plant3_leaf2
organ_sequences/maize/maize_control_plant3_leaf3
organ_sequences/maize/maize_control_plant3_leaf4
"

CROP_LEAF_SEQUENCES[sorghum]="
organ_sequences/sorghum/sorghum_control_plant1_leaf1
organ_sequences/sorghum/sorghum_control_plant1_leaf2
organ_sequences/sorghum/sorghum_control_plant1_leaf3
organ_sequences/sorghum/sorghum_control_plant2_leaf1
organ_sequences/sorghum/sorghum_control_plant2_leaf2
organ_sequences/sorghum/sorghum_control_plant2_leaf3
organ_sequences/sorghum/sorghum_control_plant2_leaf4
organ_sequences/sorghum/sorghum_highlight_plant2_leaf1
organ_sequences/sorghum/sorghum_highlight_plant2_leaf2
organ_sequences/sorghum/sorghum_highlight_plant2_leaf3
organ_sequences/sorghum/sorghum_highlight_plant2_leaf4
organ_sequences/sorghum/sorghum_highlight_plant2_leaf5
"

# Get sequences
SEQUENCES="${CROP_LEAF_SEQUENCES[$CROP]}"

# Filter by specific sequences if provided
if [ -n "$SPECIFIC_SEQUENCES" ]; then
    FILTERED_SEQUENCES=""
    for REQUESTED_SEQ in $SPECIFIC_SEQUENCES; do
        for SEQUENCE in $SEQUENCES; do
            [ -z "$SEQUENCE" ] && continue
            SEQUENCE_NAME=$(basename "$SEQUENCE")
            if [ "$SEQUENCE_NAME" == "$REQUESTED_SEQ" ]; then
                FILTERED_SEQUENCES="$FILTERED_SEQUENCES
$SEQUENCE"
                break
            fi
        done
    done
    SEQUENCES="$FILTERED_SEQUENCES"
fi

# Validate paths and count
VALID_SEQUENCES=""
for SEQUENCE in $SEQUENCES; do
    [ -z "$SEQUENCE" ] && continue
    SEQUENCE_PATH="datasets/Pheno4D/processed/v2/$SEQUENCE"
    if [ -d "$SEQUENCE_PATH" ]; then
        VALID_SEQUENCES="$VALID_SEQUENCES
$SEQUENCE"
    fi
done
SEQUENCES="$VALID_SEQUENCES"

VALID_COUNT=$(echo "$SEQUENCES" | sed '/^$/d' | wc -l)
if [ $VALID_COUNT -eq 0 ]; then
    echo "Error: No valid sequences found"
    exit 1
fi

echo "Found $VALID_COUNT sequences to process"
echo ""

# Run experiments
COUNTER=0
FAILED_COUNT=0
SUCCESS_COUNT=0

for SEQUENCE in $SEQUENCES; do
    [ -z "$SEQUENCE" ] && continue

    COUNTER=$((COUNTER + 1))
    SEQUENCE_PATH="datasets/Pheno4D/processed/v2/$SEQUENCE"
    SEQUENCE_NAME=$(basename $SEQUENCE)

    echo ""
    echo "================================================================="
    echo "[$COUNTER/$VALID_COUNT] $SEQUENCE_NAME"
    echo "================================================================="

    if bash run_experiment_from_config.sh "$SEQUENCE_PATH" "$EXPERIMENT_CONFIG"; then
        echo "✓ SUCCESS: $SEQUENCE_NAME"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=$?
        echo "✗ FAILED: $SEQUENCE_NAME (exit code: $EXIT_CODE)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "================================================================="
echo "BATCH EXPERIMENT SUMMARY"
echo "================================================================="
echo "Config: $EXPERIMENT_CONFIG"
echo "Total sequences: $VALID_COUNT"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo "================================================================="

exit 0
