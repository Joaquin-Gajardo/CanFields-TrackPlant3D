#!/bin/bash
# Launch one sbatch job per (method, sequence) on the test split.
#
# Each job runs a single sequence so it comfortably fits in the 24h wall-time
# limit (run_euler_24GB.sh). Jobs are submitted for both plant sequences and
# organ (leaf/stem) sequences of the test split, for every method.
#
# Usage:
#   bash launch_test_jobs.sh [options]
#
# Options:
#   --dry-run            Print the sbatch commands instead of submitting them.
#   --n_runs N           Repeat runs per sequence (default: 5).
#   --methods "a b c"    Space-separated methods (default: "4DSDF ode_solve NVFI TPSR").
#   --type plant|leaf|both   Which sequence types to launch (default: both).
#   -h, --help           Show this help and exit.
#
# Examples:
#   bash launch_test_jobs.sh --dry-run            # preview every job
#   bash launch_test_jobs.sh                      # submit everything (5 runs each)
#   bash launch_test_jobs.sh --methods "4DSDF" --type leaf

set -euo pipefail

# ----------------------------- configuration ------------------------------
METHODS=(4DSDF ode_solve NVFI TPSR)
N_RUNS=5
TYPE_SEL="both"          # plant | leaf | both
DRY_RUN=false
SPLITS_JSON="configs/data_splits.json"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-run_euler_24GB.sh}"
SPLIT="test"

# Organ sequences that are too short to be worth running.
EXCLUDE=(
    maize_control_plant2_leaf5
    maize_control_plant3_leaf4
    sorghum_highlight_plant2_leaf5
    tobacco_control_plant1_leaf6
)

# ------------------------------- arg parsing ------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --n_runs)    N_RUNS="$2"; shift 2 ;;
        --n_runs=*)  N_RUNS="${1#*=}"; shift ;;
        --methods)   read -r -a METHODS <<< "$2"; shift 2 ;;
        --type)      TYPE_SEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

if ! [[ "$N_RUNS" =~ ^[0-9]+$ ]] || [ "$N_RUNS" -lt 1 ]; then
    echo "Error: --n_runs must be a positive integer (got: $N_RUNS)" >&2
    exit 1
fi
if [[ "$TYPE_SEL" != "plant" && "$TYPE_SEL" != "leaf" && "$TYPE_SEL" != "both" ]]; then
    echo "Error: --type must be plant, leaf, or both (got: $TYPE_SEL)" >&2
    exit 1
fi
if [ ! -f "$SPLITS_JSON" ]; then
    echo "Error: $SPLITS_JSON not found (run from the repo root)" >&2
    exit 1
fi

# --------------------------------- helpers --------------------------------
# Derive the crop key from a sequence name:
#   maize_control_plant2        -> maize
#   tomato1_heat_plant3         -> tomato
#   tomato2_control_plant3_leaf1-> tomato
crop_of() {
    local token=${1%%_*}        # first underscore-delimited token
    echo "${token%%[0-9]*}"     # strip trailing digits (tomato1 -> tomato)
}

# Build a lookup of excluded sequences.
declare -A IS_EXCLUDED
for s in "${EXCLUDE[@]}"; do IS_EXCLUDED["$s"]=1; done

# Pull the test plant + organ sequence lists straight from the splits file.
mapfile -t PLANTS < <(python3 -c "
import json
d = json.load(open('$SPLITS_JSON'))
print('\n'.join(d['splits']['$SPLIT']['plants']))
")
mapfile -t ORGANS < <(python3 -c "
import json
d = json.load(open('$SPLITS_JSON'))
print('\n'.join(d['splits']['$SPLIT']['sequences']))
")

submit() {
    local crop=$1 method=$2 type=$3 seq=$4
    local cmd=(sbatch --job-name="${method}_${seq}" "$SBATCH_SCRIPT" \
               "$crop" "$method" "$type" "$SPLIT" "$seq" --n_runs "$N_RUNS")
    if $DRY_RUN; then
        echo "${cmd[*]}"
    else
        "${cmd[@]}"
    fi
}

# --------------------------------- launch ---------------------------------
echo "================================================================="
echo "Methods : ${METHODS[*]}"
echo "n_runs  : $N_RUNS"
echo "type    : $TYPE_SEL"
echo "dry-run : $DRY_RUN"
echo "================================================================="

count=0
skipped=0
for method in "${METHODS[@]}"; do
    if [[ "$TYPE_SEL" == "plant" || "$TYPE_SEL" == "both" ]]; then
        for seq in "${PLANTS[@]}"; do
            [ -z "$seq" ] && continue
            submit "$(crop_of "$seq")" "$method" plant "$seq"
            count=$((count + 1))
        done
    fi
    if [[ "$TYPE_SEL" == "leaf" || "$TYPE_SEL" == "both" ]]; then
        for seq in "${ORGANS[@]}"; do
            [ -z "$seq" ] && continue
            if [ -n "${IS_EXCLUDED[$seq]:-}" ]; then
                skipped=$((skipped + 1))
                continue
            fi
            submit "$(crop_of "$seq")" "$method" leaf "$seq"
            count=$((count + 1))
        done
    fi
done

echo "================================================================="
echo "Total jobs $( $DRY_RUN && echo 'to submit' || echo 'submitted'): $count"
echo "Excluded short organ sequences skipped: $skipped (x${#METHODS[@]} methods)"
echo "================================================================="
