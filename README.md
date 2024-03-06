## Installation

Start by setting up the python environment using `requirements.txt` 

## Main Procedure

### Data Retrieval

The DSTC11 Track5 dataset is included in the repository

### Model Training

Please start by training a model using one of the model configuration json files in `baseline/configs/generation`
You can create your own model configuraiton by copying the template.

The easiest way to run the training is by using the shell script

scripts/model_training/rg-review_training_and_eval.sh 

which requires only one parameter, the *model configuration name*:

`llama-7b-peft-opt_params`

### Optional Parameters

partition=${2:-gpu_24gb}  # default to 'gpu_24gb' if not specified
gpus=${3:-1}  # default to 1 if not specified
cpu_mem=${4:-24}  # default to 24 if not specified
max_desired_len=${5:-1024}
debug_level=$6
debug_fill=$7
eval_split=${8:-val}

It will automatically attempt to submit the job to the partition using Slurm.
