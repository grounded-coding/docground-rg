#!/bin/bash

# The first command-line argument is the model alias
model_alias=$1

# Check if a model alias argument is provided
if [ -z "$1" ]
then
  echo "Error: No model alias provided"
  exit 1
fi

# Set CUDA environments
versions_cuda="11.6"
versions_cudnn="8.4"
versions_acml="4.4.0"

export CUDA_HOME="/usr/local/cuda-${versions_cuda}"
export LD_LIBRARY_PATH="/usr/local/cudnn-11.X-v${versions_cudnn}/lib:/usr/local/cuda-${versions_cuda}/lib64:/usr/local/cuda-${versions_cuda}/extras/CUPTI/lib64:/usr/local/acml-${versions_acml}/cblas_mp/lib:/usr/local/acml-${versions_acml}/gfortran64/lib:/usr/local/acml-${versions_acml}/gfortran64_mp/lib/"
export HDF5_USE_FILE_LOCKING='FALSE'
export PATH=$PATH:/u/nils.hilgers/py-dstc/bin

# Optional second and third arguments for partition and GPUs
partition=${2:-gpu_24gb}  # default to 'gpu_24gb' if not specified
gpus=${3:-1}  # default to 1 if not specified
cpu_mem=${4:-24}  # default to 24 if not specified

train_command="/u/nils.hilgers/py-dstc/bin/accelerate launch --num_processes=${gpus} baseline.py --params_file baseline/configs/generation/${model_alias}_params.json --task generation --dataroot data --history_max_tokens 256 --knowledge_max_tokens 256 --knowledge_file knowledge.json --exp_name rg-review-${model_alias} --deepspeed"
eval_command="/u/nils.hilgers/py-dstc/bin/accelerate launch --num_processes=${gpus} baseline.py --generate runs/rg-review-${model_alias} --generation_params_file baseline/configs/generation/generation_params.json --task generation --dataroot data --eval_dataset val --labels_file data/val/labels.json --knowledge_file knowledge.json --output_file pred/val/rg.${model_alias}.json --deepspeed"

mkdir -p runs/rg-review-"${model_alias}"
mkdir -p pred/val

# Create the sbatch scripts dynamically
cat << EOF > tmp/train_job_rg-review.sh
#!/bin/bash

#SBATCH -o runs/rg-review-${model_alias}/train_job.out
#SBATCH -e runs/rg-review-${model_alias}/train_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:${gpus}
#SBATCH --time=96:00:00
#SBATCH -p ${partition}
#SBATCH --mem=${cpu_mem}G
#SBATCH --chdir=/u/nils.hilgers/setups/dstc11-track5

${train_command}
EOF

cat << EOF > tmp/eval_job_rg-review.sh
#!/bin/bash

#SBATCH -o runs/rg-review-${model_alias}/eval_job.out
#SBATCH -e runs/rg-review-${model_alias}/eval_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:${gpus}
#SBATCH --time=24:00:00
#SBATCH -p ${partition}
#SBATCH --mem=${cpu_mem}G
#SBATCH --chdir=/u/nils.hilgers/setups/dstc11-track5

${eval_command}
EOF

# Submit the first script and get its job ID

output=$(sbatch tmp/train_job_rg-review.sh | tee /dev/fd/2)
jobid=$(echo "$output" | awk '{print $4}')

# Submit the second script, making it dependent on the first
sbatch --dependency=afterok:"$jobid" tmp/eval_job_rg-review.sh
