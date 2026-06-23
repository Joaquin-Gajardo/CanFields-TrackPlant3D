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

module purge

/cluster/project/cropsci/jgajardo/miniconda3/bin/conda init
conda activate base
conda activate canfields # this doesn't work, need to activate environment in terminal where I send the job with sbatch run.sh and then it works
conda list

module load stack/2024-05
module load gcc/13.2.0 cuda/12.2.1 eth_proxy
echo $(module list)

nvidia-smi

# # Run the training and evaluation script with object name argument
#bash run_train_eval.sh $1


# Run crop sequences with crop name and optional method, type, split, and specific sequences.
# Repeat runs are controlled by the optional "--n_runs N" flag (default 1), which may appear
# anywhere in the arguments (it is not positional, so it never collides with sequence names).
# Usage: sbatch run_euler_24GB.sh <crop_name> [method] [type] [split] [sequence1] [sequence2] ... [--n_runs N]
# Example: sbatch run_euler_24GB.sh maize ode_solve plant test
# Example: sbatch run_euler_24GB.sh maize ode_solve leaf test maize_control_plant1_leaf1
# Example: sbatch run_euler_24GB.sh all 4DSDF plant --n_runs 5
# Example: sbatch run_euler_24GB.sh maize NVFI leaf test maize_control_plant2_leaf5 maize_control_plant2_stem --n_runs 5
bash run_crop_sequences.sh "$@"