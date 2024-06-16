#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -J BraninInv
#SBATCH -t 240:00:00
#SBATCH -o BraninInv_cpu.out
#SBATCH -e BraninInv_cpu.err
## SBATCH --partition=gpu-ampere

conda activate cs236
python Branin.py --run_number 1