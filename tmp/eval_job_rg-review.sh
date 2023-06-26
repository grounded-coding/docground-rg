#!/bin/bash

#SBATCH -o runs/rg-review-llama-7b-peft-16fp/eval_job.out
#SBATCH -e runs/rg-review-llama-7b-peft-16fp/eval_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH -p gpu_24gb
#SBATCH --mem=64G
#SBATCH --chdir=/u/nils.hilgers/setups/dstc11-track5

/u/nils.hilgers/py-dstc/bin/accelerate launch --num_processes=3 baseline.py --generate runs/rg-review-llama-7b-peft-16fp --generation_params_file baseline/configs/generation/generation_params.json --task generation --dataroot data --eval_dataset val --labels_file data/val/labels.json --knowledge_file knowledge.json --output_file pred/val/rg.llama-7b-peft-16fp.json --deepspeed
