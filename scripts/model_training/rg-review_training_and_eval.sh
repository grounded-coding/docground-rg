#!/bin/bash

# The first command-line argument is the model alias
model_alias=$1
partition=${2:-gpu_24gb}  # default to 'gpu_24gb' if not specified
gpus=${3:-1}  # default to 1 if not specified
cpu_mem=${4:-24}  # default to 24 if not specified
max_desired_len=${5:-1024}
debug_level=$6
debug_fill=$7

export MAX_DESIRED_LEN=$max_desired_len

# Check if a model alias argument is provided
if [ -z "$1" ]
then
  echo "Error: No model alias provided"
  exit 1
fi

debug_fill=""
if [ -n "$debug_fill" ]
then
  debug_fill="--debug_fill"
fi

export ACCELERATE_HOME="baseline/configs/accelerate/"
accelerate_config="${ACCELERATE_HOME}multi_gpu.yaml"

# For one gpu we dont need mixed precision training but can fit up to 7B parameters using native fp16 weights
# For multiple gpus we assume deepspeed which only supports mixed precision training, so note that mixed fp16 training is always enabled
export ACCELERATE_MIXED_PRECISION="fp16"
if [ "$gpus" -le 1 ]
then
  accelerate_config="${ACCELERATE_HOME}single_gpu.yaml"
  export ACCELERATE_MIXED_PRECISION="no"
fi

debug_flag=""
if [ -n "$debug_level" ]
then
  debug_flag="--debug ${debug_level}"
fi

params_file="baseline/configs/generation/${model_alias}_params.json"
generation_params_file="baseline/configs/generation/generation_params.json"
suffix=$(date +"%m%d%H%M%S")

train_command="singularity exec --nv --bind /work:/work runs/nlp_torch_sis.sif accelerate launch --config_file ${accelerate_config} --main_process_port=25678 --num_processes=${gpus} baseline.py --params_file ${params_file} --task generation --dataroot data --knowledge_file knowledge.json --exp_name rg-review-${model_alias}-${suffix} ${debug_flag} ${debug_fill}"
eval_command="singularity exec --nv --bind /work:/work runs/nlp_torch_sis.sif accelerate launch --config_file ${accelerate_config} --main_process_port=25679 --num_processes=${gpus} baseline.py --generate runs/rg-review-${model_alias}-${suffix} --generation_params_file ${generation_params_file} --task generation --dataroot data --eval_dataset val --labels_file data/val/labels.json --knowledge_file knowledge.json --output_file pred/val/rg.${model_alias}-${suffix}.json ${debug_flag} ${debug_fill}"

mkdir -p runs/rg-review-"${model_alias}-${suffix}"
mkdir -p pred/val

# Create the sbatch scripts dynamically
cat << EOF > tmp/train_rg-review-"${model_alias}".sh
#!/bin/bash

#SBATCH -o runs/rg-review-${model_alias}-${suffix}/train_job.out
#SBATCH -e runs/rg-review-${model_alias}-${suffix}/train_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:${gpus}
#SBATCH --time=168:00:00
#SBATCH -p ${partition}
#SBATCH --mem=${cpu_mem}G
#SBATCH --chdir=/u/nils.hilgers/setups/dstc11-track5

${train_command}
EOF

cat << EOF > tmp/eval_rg-review-"${model_alias}".sh
#!/bin/bash

#SBATCH -o runs/rg-review-${model_alias}-${suffix}/eval_job.out
#SBATCH -e runs/rg-review-${model_alias}-${suffix}/eval_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:${gpus}
#SBATCH --time=168:00:00
#SBATCH -p ${partition}
#SBATCH --mem=${cpu_mem}G
#SBATCH --chdir=/u/nils.hilgers/setups/dstc11-track5

${eval_command}
EOF

# Submit the first script and get its job ID

output=$(sbatch tmp/train_rg-review-"${model_alias}".sh | tee /dev/fd/2)
jobid=$(echo "$output" | awk '{print $4}')

# Submit the second script, making it dependent on the first
sbatch --dependency=afterok:"$jobid" tmp/eval_rg-review-"${model_alias}".sh
