#!/bin/bash

# The first command-line argument is the model alias
model_alias=$1
partition=${2:-gpu_24gb}  # default to 'gpu_24gb' if not specified
gpus=${3:-1}  # default to 1 if not specified
cpu_mem=${4:-24}  # default to 24 if not specified
max_desired_len=${5:-1024}
debug_level=$6
debug_fill=$7
eval_split=${8:-val}

# Check if a model alias argument is provided
if [ -z "$1" ]
then
  echo "Error: No model alias provided"
  exit 1
fi

# Bind /work directory if necessary
binds=""
if test -d /work; then
  binds=" --bind /work:/work"
fi

debug_fill=""
if [ -n "$debug_fill" ]
then
  debug_fill="--debug_fill"
fi

acc_home="baseline/configs/accelerate/"
# default config uses multi gpu
accelerate_config="${acc_home}multi_gpu.yaml"

acc_mp="no"
# if model_alias contains peft use bf16
if [[ "$model_alias" == *"peft"* ]]
then
  acc_mp="bf16"
fi

if [ "$gpus" -le 1 ]
then
  accelerate_config="${acc_home}single_gpu.yaml"
fi

debug_flag=""
if [ -n "$debug_level" ]
then
  debug_flag="--debug ${debug_level}"
fi

params_file="baseline/configs/generation/${model_alias}_params.json"
generation_params_file="baseline/configs/generation/generation_params.json"
suffix=$(date +"%m%d%H%M%S")

train_command="export MAX_DESIRED_LEN=${max_desired_len}; singularity exec --nv${binds} runs/nlp_torch_sis.sif accelerate launch --mixed_precision=${acc_mp} --config_file ${accelerate_config} --main_process_port=25678 --num_processes=${gpus} baseline.py --params_file ${params_file} --task generation --dataroot data --knowledge_file knowledge.json --exp_name ${model_alias}-rg-review-${suffix} ${debug_flag} ${debug_fill}"
eval_command="export MAX_DESIRED_LEN=${max_desired_len}; singularity exec --nv${binds} runs/nlp_torch_sis.sif accelerate launch --mixed_precision=${acc_mp} --config_file ${accelerate_config} --main_process_port=25679 --num_processes=${gpus} baseline.py --generate runs/${model_alias}-rg-review-${suffix} --generation_params_file ${generation_params_file} --task generation --dataroot data --eval_dataset ${eval_split} --labels_file data/${eval_split}/labels.json --knowledge_file knowledge.json --output_file pred/${eval_split}/rg.${model_alias}-${suffix}.json ${debug_flag} ${debug_fill}"

mkdir -p runs/"${model_alias}-rg-review-${suffix}"

# create all directories by extracting from model_alias the folder names before last /
mkdir -p pred/${eval_split}/"${model_alias}"-b
mkdir -p tmp/"${model_alias}"-b

create_script_file() {
    local script_file=$1
    local command=$2

    cat << EOF > "${script_file}"
#!/bin/bash
${command}
EOF
    chmod +x "${script_file}"
}

if command -v sbatch &>/dev/null; then
  # Create the sbatch scripts dynamically
  cat << EOF > tmp/"${model_alias}"-train_rg-review.sh
#!/bin/bash

#SBATCH -o runs/${model_alias}-rg-review-${suffix}/train_job.out
#SBATCH -e runs/${model_alias}-rg-review-${suffix}/train_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:${gpus}
#SBATCH --time=168:00:00
#SBATCH -p ${partition}
#SBATCH --mem=${cpu_mem}G
#SBATCH --chdir=/home/nhilgers/setups/dstc11-track5

${train_command}
EOF

  cat << EOF > tmp/"${model_alias}"-eval_rg-review.sh
#!/bin/bash

#SBATCH -o runs/${model_alias}-rg-review-${suffix}/eval_job.out
#SBATCH -e runs/${model_alias}-rg-review-${suffix}/eval_job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:${gpus}
#SBATCH --time=168:00:00
#SBATCH -p ${partition}
#SBATCH --mem=${cpu_mem}G
#SBATCH --chdir=/home/nhilgers/setups/dstc11-track5

${eval_command}
EOF

  # Submit the first script and get its job ID

  output=$(sbatch tmp/"${model_alias}"-train_rg-review.sh | tee /dev/fd/2)
  jobid=$(echo "$output" | awk '{print $4}')

  # Submit the second script, making it dependent on the first
  sbatch --dependency=afterok:"$jobid" tmp/"${model_alias}"-eval_rg-review.sh

else
    create_script_file "runs/${model_alias}-rg-review-${suffix}/train_job.sh" "${train_command} > runs/${model_alias}-rg-review-${suffix}/train_job.out 2> runs/${model_alias}-rg-review-${suffix}/train_job.err"
    create_script_file "runs/${model_alias}-rg-review-${suffix}/eval_job.sh" "${eval_command} > runs/${model_alias}-rg-review-${suffix}/eval_job.out 2> runs/${model_alias}-rg-review-${suffix}/eval_job.err"
fi


if [ "$gpus" -le 1 ]
then
    gpu_status=$(nvidia-smi)

    # Initialize CUDA_VISIBLE_DEVICES as empty
    export CUDA_VISIBLE_DEVICES=

    # Loop through each GPU and check if there are any processes running on it
    for gpu_id in {0..9}; do
        # Check if the GPU is listed in nvidia-smi output
        if echo "$gpu_status" | grep -q " ${gpu_id}  NVIDIA "; then
            # Check if there are no processes running on this GPU
            if ! echo "$gpu_status" | grep -q "|    ${gpu_id}  "; then
                # If no processes are found, set CUDA_VISIBLE_DEVICES to this GPU and break the loop
                export CUDA_VISIBLE_DEVICES=$gpu_id
                break
            fi
        fi
    done
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


    # Run the first script
    bash "runs/${model_alias}-rg-review-${suffix}/train_job.sh"
    bash "runs/${model_alias}-rg-review-${suffix}/eval_job.sh"
fi