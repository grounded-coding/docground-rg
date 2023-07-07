
import torch
import numpy as np
from pynvml import *
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, logging, \
    AutoModelForSeq2SeqLM, AutoModelForCausalLM
from argparse import Namespace
from accelerate import Accelerator
import contextlib


# This script checks the consumed memory of large language models for training with different configurations, specifically
# regarding the sequence length.


nvmlInit()

def print_gpu_utilization():
    device_count = nvmlDeviceGetCount()
    total_memory_used = 0
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        total_memory_used += info.used
        print(f"Total GPU memory occupied: {total_memory_used // 1024 ** 2} MB.\n")


def get_random_training_data(config):
    seq_len, dataset_size = config.seq_len, config.num_seq
    dummy_data = {
        "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
        "labels": np.random.randint(4, 5, (dataset_size, seq_len)),
    }
    ds = Dataset.from_dict(dummy_data)
    ds.set_format("pt")
    return ds

llm = True
if llm:
    model_name = "huggyllama/llama-7b"
    gen_task = "causal_lm"
else:
    model_name = "facebook/bart-base"
    gen_task = "causal_lm"

default_config = {"torch_dtype": torch.float16,
            "use_peft": True,
            "gradient_checkpointing": True,
            "eval_only": False,
            "seq_len": 1024,
            "num_seq": 10,
            "training_args": {
                "per_device_train_batch_size": 1,
                "output_dir": "tmp",
                "evaluation_strategy": "steps",
                "num_train_epochs": 1,
                "log_level": "error",
                "report_to": "none",
                }
            }

configs = [default_config]
seq_lens = [1024]

model = None
with open('gpu_mem_output.txt', 'w') as file:
    with contextlib.redirect_stdout(file):
        for config in configs:
            config = Namespace(**config)
            for seq_len in seq_lens:
                del model
                torch.cuda.empty_cache()
                print_gpu_utilization()
                config.seq_len = seq_len
                print(f"Using config with values\n{config}")
                model_load_args = {"low_cpu_mem_usage": True, "torch_dtype": config.torch_dtype}
                print(f"Loading model {model_name} for task {gen_task}.")
                model = AutoModelForCausalLM.from_pretrained(f"{model_name}", **model_load_args)
                print("Model loaded.")
                print_gpu_utilization()

                ds = get_random_training_data(config)

                if config.use_peft:
                    from peft import LoraConfig, prepare_model_for_int8_training, get_peft_model, TaskType, PeftModel
                    if gen_task in ["causal_lm"]:
                        lora_task_type = TaskType.CAUSAL_LM
                        target_modules = ["q_proj", "v_proj"]
                    elif gen_task in ["seq2seq_lm"]:
                        lora_task_type = TaskType.SEQ_2_SEQ_LM
                        target_modules = ["q", "v"]
                    else:
                        raise NotImplementedError()
                    peft_config = LoraConfig(
                        task_type=lora_task_type, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
                        target_modules=target_modules
                    )
                    if config.gradient_checkpointing:
                        model.enable_input_require_grads()

                if config.gradient_checkpointing:
                    model.gradient_checkpointing_enable()

                if config.eval_only:
                    print("Running inference only.")
                    input_ids = ds['input_ids']
                    y = model.generate(input_ids)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    z = tokenizer.batch_decode(y)
                    print_gpu_utilization()
                else:
                    accelerator = Accelerator()
                    print("Running training on test data.")
                    training_args = TrainingArguments(**config.training_args)
                    model = accelerator.prepare(model)
                    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
                    result = trainer.train()
                    print("Model training finished.")
                    print_gpu_utilization()
