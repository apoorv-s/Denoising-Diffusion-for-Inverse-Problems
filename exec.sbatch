#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J PM_MLP
#SBATCH -t 240:00:00
#SBATCH -o PM_gpu2.out
#SBATCH -e PM_gpu2.err
#SBATCH --partition=gpu-tesla

ml load cuda/11.8
# nvidia-smi

conda activate diff
echo $CONDA_PREFIX

# python Train.py --run_number 6 --model_name 'branin'
# python Train.py --run_number 1 --model_name 'pose_transformer'
python Train.py --run_number 1 --model_name 'pose_mlp'
