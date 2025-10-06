#!/bin/bash
#SBATCH -J CanFields
#SBATCH --mail-type=END,FAIL
#SBATCH --time=12:00:00
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:80g
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=2
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

python train_debugV2.py -a 0.05,0.05,0.01 -o datasets/Pheno4D/plants/tomato2_control_plant2 -d pheno4d -u True -m False -c 0,9,1 -ev 15000