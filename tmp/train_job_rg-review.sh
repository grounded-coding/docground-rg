#!/bin/bash

#SBATCH -o runs/rg-review-llama-7b-peft-16fp/train_job.out
#SBATCH -e runs/rg-review-llama-7b-peft-16fp/train_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:3
#SBATCH --time=96:00:00
#SBATCH -p gpu_24gb
#SBATCH --mem=64G
#SBATCH --chdir=/u/nils.hilgers/setups/dstc11-track5

/u/nils.hilgers/py-dstc/bin/accelerate launch --num_processes=3 baseline.py --params_file baseline/configs/generation/llama-7b-peft-16fp_params.json --task generation --dataroot data --history_max_tokens 256 --knowledge_max_tokens 256 --knowledge_file knowledge.json --exp_name rg-review-llama-7b-peft-16fp --deepspeed
