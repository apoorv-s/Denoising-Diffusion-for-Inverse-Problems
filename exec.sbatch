#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J BrSample
#SBATCH -t 240:00:00
#SBATCH -o BrSample.out
#SBATCH -e BrSample.err
#SBATCH --partition=gpu-tesla

ml load cuda/11.8
# nvidia-smi

conda activate diff
echo $CONDA_PREFIX

# python Train.py --run_number 1 --model_name 'branin_mlp'
# python Train.py --run_number 1 --model_name 'branin_transformer'
# python Train.py --run_number 1 --model_name 'pose_transformer'
# python Train.py --run_number 1 --model_name 'pose_mlp'

# python Sample.py --model_name 'branin_transformer' --model_number 1 --target_y 10 --n_test_points 5000 --cond_weight 0.01;
# python Sample.py --model_name 'branin_transformer' --model_number 1 --target_y 50 --n_test_points 5000 --cond_weight 0.01;
# python Sample.py --model_name 'branin_transformer' --model_number 1 --target_y 100 --n_test_points 5000 --cond_weight 0.01;
# python Sample.py --model_name 'branin_transformer' --model_number 1 --target_y 200 --n_test_points 5000 --cond_weight 0.01;

python Sample.py --model_name 'pose_transformer' --model_number 1 --cond_weight 0.01;