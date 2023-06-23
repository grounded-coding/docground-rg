#!/bin/bash

#SBATCH -o runs/rg-review-llama-7b-16fp/train_job.out
#SBATCH -e runs/rg-review-llama-7b-16fp/train_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH -p gpu_24gb
#SBATCH --mem=24G
#SBATCH --chdir=/u/nils.hilgers/setups/dstc11-track5

/u/nils.hilgers/py-dstc/bin/python -m torch.distributed.run --nproc_per_node=2 baseline.py --params_file baseline/configs/generation/llama-7b-16fp_params.json --task generation --dataroot data --history_max_tokens 256 --knowledge_max_tokens 256 --knowledge_file knowledge.json --exp_name rg-review-llama-7b-16fp --deepspeed_config baseline/configs/deepspeed/ds_config.json
