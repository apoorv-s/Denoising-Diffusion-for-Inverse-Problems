#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -J BraninInv
#SBATCH -t 240:00:00
#SBATCH -o BraninInv.out
#SBATCH -e BraninInv.err

conda activate cs236
