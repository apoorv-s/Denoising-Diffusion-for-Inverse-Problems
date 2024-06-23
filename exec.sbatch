#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J BranInv
#SBATCH -t 240:00:00
#SBATCH -o BranInv_gpu.out
#SBATCH -e BranInv_gpu.err
#SBATCH --partition=gpu-ampere


conda activate diff
echo $CONDA_PREFIX
python Branin.py --run_number 2 --train