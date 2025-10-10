#!/bin/bash
#SBATCH -J CanFields
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
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


# Run crop sequences with crop name and optional method and type
# Usage: sbatch run_euler.sh <crop_name> [method] [type]
# Example: sbatch run_euler.sh maize ode_solve plant
# Example: sbatch run_euler.sh maize ode_solve leaf
bash run_crop_sequences.sh $1 $2 $3