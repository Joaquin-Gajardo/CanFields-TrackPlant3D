#!/bin/bash
#SBATCH -J CanFields
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_log/%j.out
#SBATCH --error=sbatch_log/%j.out

# Options passed to sbatch override these, e.g. sbatch --gres=gpumem:80g run_euler.sh ...

module purge

# sbatch copies the submitting shell's environment: run `conda activate canfields` before
# submitting. The module lines below are for ETH's Euler cluster; adapt them to yours.

module load stack/2024-05
module load gcc/13.2.0 cuda/12.2.1 eth_proxy
echo $(module list)

nvidia-smi

# Run crop sequences with crop name and optional method, type, split, and specific sequences.
# Repeat runs are controlled by the optional "--n_runs N" flag (default 1), which may appear
# anywhere in the arguments (it is not positional, so it never collides with sequence names).
# Usage: sbatch run_euler.sh <crop_name> [method] [type] [split] [sequence1] [sequence2] ... [--n_runs N]
# Example: sbatch run_euler.sh maize ode_solve plant test
# Example: sbatch run_euler.sh maize ode_solve leaf test maize_control_plant1_leaf1
# Example: sbatch run_euler.sh all 4DSDF plant --n_runs 5
# Example: sbatch run_euler.sh maize NVFI leaf test maize_control_plant2_leaf4 maize_control_plant2_stem --n_runs 5
# Example: sbatch --gres=gpumem:80g run_euler.sh all TPSR plant test   # more GPU memory
bash run_crop_sequences.sh "$@"