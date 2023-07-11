import argparse
import os
import random
import pandas as pd
import json

from typing import Dict
from argparse import Namespace
from accelerate.utils import DistributedType

import numpy as np
from peft import PeftModel
import logging
from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
from .dataset import ResponseGenerationEvalDataset, ResponseGenerationDataset, SPECIAL_TOKENS, DEFAULT_PAD_TOKEN

from .utils.argument import update_additional_params
from .utils.model import run_batch_generation_sample
from .utils.metrics import (
    DataCacheMetric,
    UnigramMetric, NGramDiversity,
    CorpusNGramDiversity,
    BLEU, METEOR, ROUGE,
    print_gpu_utilization
)
from .utils.data import write_generation_preds

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def convert_to_padded_tensor(tensor_list, pad_dim=0, pad_token_id=-1, accelerator=None):
    tensor_list = accelerator.pad_across_processes(tensor_list, dim=pad_dim, pad_index=pad_token_id)
    logger.debug(f"The padded tensor at this point has shape {[t.shape for t in tensor_list]}")
    max_len = max(tensor.shape[pad_dim] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        padding_size = list(tensor.shape)
        padding_size[pad_dim] = max_len - tensor.shape[pad_dim]
        pad_tensor = torch.full(padding_size, pad_token_id, device=tensor.device)
        padded_tensor = torch.cat([tensor, pad_tensor], dim=pad_dim)
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors)


def evaluate(args, eval_dataset, model, tokenizer, desc="", accelerator=None, gen_task="seq2seq_lm") -> Dict:
    """ Generate responses and report the eval performance if references are available """
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_device_eval_batch_size

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,  # only support batch_size=1 for sampling right now
        collate_fn=eval_dataset.collate_fn
    )

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    metrics = [
        DataCacheMetric(),
        UnigramMetric(metric="recall"),
        UnigramMetric(metric="precision"),
        NGramDiversity(n=1),
        NGramDiversity(n=2),
        NGramDiversity(n=3),
        NGramDiversity(n=4),
        CorpusNGramDiversity(n=1),
        CorpusNGramDiversity(n=2),
        CorpusNGramDiversity(n=3),
        CorpusNGramDiversity(n=4),
        BLEU(),
        ROUGE(),
        METEOR(),
    ]

    args.tokenizer = tokenizer
    dialog_ids = []
    all_sampled_outputs = []
    all_ground_truths = []
    do_evaluate = False
    model.eval()

    run_batch_generation_func = run_batch_generation_sample

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=False):
        with torch.no_grad():
            sampled_output_ids, ground_truth, dialog_id = run_batch_generation_func(args, model, tokenizer, batch, eval_dataset, accelerator=accelerator, gen_task=gen_task)
        dialog_ids.append(dialog_id)

        if ground_truth.strip() != "":
            do_evaluate = True
            ground_truth_ids = tokenizer.encode(ground_truth, return_tensors='pt').squeeze()
            all_sampled_outputs.append(sampled_output_ids.to(args.device))
            all_ground_truths.append(ground_truth_ids.to(args.device))

    # wait for all processes to finish
    accelerator.wait_for_everyone()

    logger.info(f"The list of tensors for outputs has shape {[t.shape for t in all_sampled_outputs]}")
    logger.info(f"The list of tensors for gts has shape {[t.shape for t in all_ground_truths]}")

    # Convert lists to padded tensors
    all_sampled_outputs = convert_to_padded_tensor(all_sampled_outputs, pad_dim=1, pad_token_id=tokenizer.pad_token_id, accelerator=accelerator).to(args.device)
    all_ground_truths = convert_to_padded_tensor(all_ground_truths, pad_token_id=tokenizer.pad_token_id, accelerator=accelerator).to(args.device)

    # Gather the tensors
    logger.info("Gathering results from all GPUs")
    all_sampled_outputs = accelerator.gather(all_sampled_outputs)
    all_ground_truths = accelerator.gather(all_ground_truths)
    result = dict()

    metrics_list = []
    if accelerator.is_main_process:
        # Remove padding and convert back to lists
        logger.info(f"Gathered results from all GPUs. Preparing results for decoding.")
        all_sampled_outputs = all_sampled_outputs.tolist()
        all_ground_truths = all_ground_truths.tolist()
        logger.info("Converted tensors back to lists on CPU. Continuing with decoding")

        all_sampled_texts = [[tokenizer.decode(sampled_output, skip_special_tokens=True) for sampled_output in sampled_outputs] for sampled_outputs in all_sampled_outputs]
        all_ground_truths_text = [tokenizer.decode(ground_truth, skip_special_tokens=True) for ground_truth in all_ground_truths]
        logger.info("Finished decoding. Continuing with text stripping and writing the predictions.")
        
        # Remove leading white spaces if necessary
        all_sampled_texts = [[text.lstrip() for text in texts] for texts in all_sampled_texts]
        all_ground_truths_text = [text.lstrip() for text in all_ground_truths_text]
        
        if args.output_file:
            write_generation_preds(eval_dataset.dataset_walker, args.output_file, dialog_ids, all_sampled_texts)

        if do_evaluate:
            output_eval_file = os.path.join(eval_output_dir, f"eval_results_{args.task}.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results %s *****" % desc)
                writer.write("***** Eval results %s *****\n" % desc)
                for sampled_text, ground_truth_text in zip(all_sampled_texts, all_ground_truths_text):
                    sample_metrics = {}
                    for metric_class in [BLEU]:
                        metric = metric_class()
                        metric.update((sampled_text[0], ground_truth_text))

                        name = metric.name()
                        score = metric.compute()

                        if metric.is_single:
                            sample_metrics[name] = score
                        else:
                            for _name, _score in zip(name, score):
                                sample_metrics[_name] = _score

                    # Append the metrics for this sample to the DataFrame
                    metrics_list.append(sample_metrics)

                for metric in metrics:
                    for sampled_text, ground_truth_text in zip(all_sampled_texts, all_ground_truths_text):
                        metric.update((sampled_text[0], ground_truth_text))         
                    name = metric.name()
                    score = metric.compute()
                    if metric.is_single:
                        result[name] = score
                        logger.info("  %s = %s", name, str(score))
                        writer.write("%s = %s\n" % (name, str(score)))
                    else:
                        for _name, _score in zip(name, score):
                            result[_name] = _score
                            logger.info("  %s = %s", _name, str(_score))
                            writer.write("%s = %s\n" % (_name, str(_score)))
        metrics_df = pd.DataFrame(metrics_list)
        logger.info(f"Writing evaluation metrics to {eval_output_dir}/eval_metrics.csv")
        metrics_df.to_csv(os.path.join(eval_output_dir, "eval_metrics.csv"), index=False)
    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument('--generate', action='store_true')
    parser.add_argument("--debug", type=int, default=0,
                        help="If set, will only use a small number (==debug) of data for training and test.")
    parser.add_argument("--task", type=str, default='',
                        help="to specify eval task if different from training")
    parser.add_argument("--generation_params_file", type=str, default="",
                        help="JSON configuration file for generation-related configurations.")
    parser.add_argument("--dataroot", type=str, default="",
                        help="Path to dataset, will override the path in config.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                             "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--prompting", type=str, default="no", help="Adds instruction prompts to the dataset for causal LM")
    parser.add_argument("--debug_fill", action="store_true", help="If set, will fill all inputs by max_desired_len up with a random number")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # load args from params file and update the args Namespace
    args.params_file = os.path.join(args.checkpoint, "params.json")
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        update_additional_params(params, args)
        args.update(params)
        args["torch_dtype"] = torch.float32 if not args["fp16"] else torch.float16
        if len(args["generation_params_file"]) > 0:
            with open(args["generation_params_file"]) as fg:
                generation_params = json.load(fg)
            args.update(generation_params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    dataset_args = Namespace(**args.dataset_args)
    dataset_args.task = args.task
    dataset_args.generate = args.generate
    dataset_args.debug = args.debug
    args.output_dir = args.checkpoint
    accelerator = Accelerator()
    args.device = accelerator.device

    print_gpu_utilization(args, "after loading the drivers")
    model_load_kwargs = {"torch_dtype": args.torch_dtype, "low_cpu_mem_usage": accelerator.distributed_type != DistributedType.DEEPSPEED}
    if args.load_in_8bit:
        model_load_kwargs["device_map"] = "auto"
        model_load_kwargs["load_in_8bit"] = True

    # Set seed
    set_seed(args.seed)

    gen_task = dataset_args.gen_task
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if gen_task.lower() == "seq2seq_lm":
        model_class = AutoModelForSeq2SeqLM
    elif gen_task.lower() == "causal_lm":
        model_class = AutoModelForCausalLM
    else:
        raise ValueError(f"Unknown task {gen_task}")

    logger.info("Generation parameters %s", args)
    logger.info("Model inference parameters %s", model_load_kwargs)

    if args.use_peft:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            logger.info(f"Loaded config class: {type(config)}")
            model = model_class.from_pretrained(args.model_name_or_path, config=config, **model_load_kwargs)
            model = PeftModel.from_pretrained(model, model_id=args.checkpoint)
    else:
        model = model_class.from_pretrained(args.checkpoint, **model_load_kwargs)
    model = model.to(args.device)
    print_gpu_utilization(args, "after loading the model weights")

    # Evaluation
    eval_dataset = ResponseGenerationEvalDataset(dataset_args, tokenizer, split_type=args.eval_dataset,
                                                 labels_file=args.labels_file)

    result = evaluate(args, eval_dataset, model, tokenizer, desc=args.eval_desc or "val", accelerator=accelerator, gen_task=gen_task)
    print_gpu_utilization(args, "after finishing the inference")
    return result


if __name__ == "__main__":
    main()
