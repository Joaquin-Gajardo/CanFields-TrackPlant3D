#!/bin/bash
# Regenerate deformed meshes + metrics for the LATEST experiment folder of each
# sequence, on existing checkpoints (NO retraining). Fixes the stale-mesh CD.
#
# Usage:
#   bash scripts/reeval_latest.sh [METHOD ...]      # default: all methods found
# Env vars:
#   EPOCH=15000        epoch to evaluate (default 15000)
#   SEQ_FILTER=substr  only process sequences whose name contains <substr>
#   DRYRUN=1           print the plan, run nothing
#
# Examples:
#   SEQ_FILTER=maize_control_plant2_leaf1 bash scripts/reeval_latest.sh NVFI
#   EPOCH=15000 bash scripts/reeval_latest.sh ode_solve 4DSDF NVFI TPSR

set -u
EPOCH="${EPOCH:-15000}"
SEQ_FILTER="${SEQ_FILTER:-}"
TEST_ONLY="${TEST_ONLY:-0}"   # 1 = only process sequences in configs/data_splits.json test split
DRYRUN="${DRYRUN:-0}"
RESULTS=results/pheno4d
DATA=datasets/Pheno4D/processed/v2

METHODS="${*:-$(ls "$RESULTS" 2>/dev/null)}"

ALLOWED=""
if [ "$TEST_ONLY" = "1" ]; then
  ALLOWED=$(python3 -c "import json;d=json.load(open('configs/data_splits.json'))['splits']['test'];print(' '.join(d['plants']+d['sequences']))")
fi

for M in $METHODS; do
  [ -d "$RESULTS/$M" ] || { echo "skip method $M (no dir)"; continue; }
  for CROPDIR in "$RESULTS/$M"/*/; do
    [ -d "$CROPDIR" ] || continue
    CROP=$(basename "$CROPDIR")
    for SEQDIR in "$CROPDIR"*/; do
      [ -d "$SEQDIR" ] || continue
      SEQNAME=$(basename "$SEQDIR")
      [ -n "$SEQ_FILTER" ] && [[ "$SEQNAME" != *"$SEQ_FILTER"* ]] && continue
      [ "$TEST_ONLY" = "1" ] && [[ " $ALLOWED " != *" $SEQNAME "* ]] && continue

      # latest experiment folder: timestamp-prefixed -> lexicographic sort is chronological
      EXP=$(ls -1 "$SEQDIR" 2>/dev/null | grep -v '_copy$' | grep -v 'interp' | sort | tail -n1)
      [ -z "$EXP" ] && { echo "skip $M/$CROP/$SEQNAME (no experiment folder)"; continue; }

      # locate dataset sequence path (organ vs plant)
      if   [ -d "$DATA/organ_sequences/$CROP/$SEQNAME" ]; then SEQ="$DATA/organ_sequences/$CROP/$SEQNAME"
      elif [ -d "$DATA/plant_sequences/$CROP/$SEQNAME" ]; then SEQ="$DATA/plant_sequences/$CROP/$SEQNAME"
      else echo "SKIP $M/$CROP/$SEQNAME: dataset path not found"; continue; fi

      SPLIT="configs/point_splits/${SEQNAME}_split.json"
      SPLIT_ARG=""; [ -f "$SPLIT" ] && SPLIT_ARG="--point_split_file $SPLIT"

      echo "=== $M / $CROP / $SEQNAME / $EXP (epoch $EPOCH) ==="
      if [ "$DRYRUN" = "1" ]; then
        echo "  SEQ=$SEQ  SPLIT_ARG=$SPLIT_ARG"
        continue
      fi

      python3 eval_sphere.py  -d pheno4d -e "$EPOCH" -s 1 -o "$SEQ" --experiment_name "$EXP" --method "$M" -m False --overwrite || { echo "  step1 FAILED"; continue; }
      python3 eval_sphere.py  -d pheno4d -e "$EPOCH" -s 2 -o "$SEQ" --experiment_name "$EXP" --method "$M" -m False --overwrite || { echo "  step2 FAILED"; continue; }
      python3 eval_metrics.py -d pheno4d -e "$EPOCH" -o "$SEQ" --experiment_name "$EXP" --methods "$M" $SPLIT_ARG       || { echo "  metrics FAILED"; continue; }

      # print resulting CD
      MJ="$SEQDIR$EXP/metrics/metrics_epoch_${EPOCH}.json"
      [ -f "$MJ" ] && python3 -c "
import json,sys
d=json.load(open('$MJ'))
cd=d.get('test',{}).get('mean_chamfer_distance', d.get('mean_chamfer_distance'))
ep=d.get('test',{}).get('mean_tip_epe', d.get('mean_tip_epe'))
print(f'  -> CD={cd:.2f}  tipEPE={ep}' if cd is not None else '  -> (no CD)')
"
    done
  done
done
echo "Done."
