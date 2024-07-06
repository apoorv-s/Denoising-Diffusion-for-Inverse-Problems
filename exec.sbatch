#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J BranInv2
#SBATCH -t 240:00:00
#SBATCH -o BranInv_gpu2.out
#SBATCH -e BranInv_gpu2.err
#SBATCH --partition=gpu-ampere


conda activate diff
echo $CONDA_PREFIX
python Train.py --run_number 6 --model_name 'branin'