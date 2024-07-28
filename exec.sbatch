#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J BrTFM
#SBATCH -t 240:00:00
#SBATCH -o BrTFM.out
#SBATCH -e BrTFM.err
#SBATCH --partition=gpu-tesla

ml load cuda/11.8
# nvidia-smi

conda activate diff
echo $CONDA_PREFIX

# python Train.py --run_number 50 --model_name 'branin_mlp'
python Train.py --run_number 50 --model_name 'branin_transformer'
# python Train.py --run_number 50 --model_name 'pose_transformer'
# python Train.py --run_number 50 --model_name 'pose_mlp'
