#!/bin/bash

#SBATCH -o runs/rg-review-llama-7b-16fp/eval_job.out
#SBATCH -e runs/rg-review-llama-7b-16fp/eval_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH -p gpu_24gb
#SBATCH --mem=24G
#SBATCH --chdir=/u/nils.hilgers/setups/dstc11-track5

/u/nils.hilgers/py-dstc/bin/python -m torch.distributed.run --nproc_per_node=2 baseline.py --generate runs/rg-review-llama-7b-16fp --generation_params_file baseline/configs/generation/generation_params.json --task generation --dataroot data --eval_dataset val --labels_file data/val/labels.json --knowledge_file knowledge.json --output_file pred/val/rg.llama-7b-16fp.json --deepspeed_config baseline/configs/deepspeed/ds_config.json
