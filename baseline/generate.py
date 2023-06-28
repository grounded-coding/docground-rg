import argparse
import os
import random
import json

from typing import Dict
from argparse import Namespace

import numpy as np
import logging
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import set_seed

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
from .dataset import ResponseGenerationEvalDataset, ResponseGenerationDataset

from .utils.argument import update_additional_params
from .utils.model import run_batch_generation_sample
from .utils.metrics import (
    DataCacheMetric,
    UnigramMetric, NGramDiversity,
    CorpusNGramDiversity,
    BLEU, METEOR, ROUGE
)
from .utils.data import write_generation_preds

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


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
            all_sampled_outputs.append(sampled_output_ids)
            all_ground_truths.append(ground_truth_ids)

    # wait for all processes to finish
    accelerator.wait_for_everyone()

    # gather outputs and ground truths on all processes
    all_sampled_outputs = accelerator.gather(all_sampled_outputs)
    all_ground_truths = accelerator.gather(all_ground_truths)

    result = dict()
    if accelerator.is_main_process:
        # Decode on main process
        all_sampled_texts = [[tokenizer.decode(_sampled_output_ids, skip_special_tokens=True).lstrip() for _sampled_output_ids in sampled_outputs] for sampled_outputs in all_sampled_outputs]
        all_ground_truths_text = [tokenizer.decode(_ground_truth, skip_special_tokens=True).lstrip() for _ground_truth in all_ground_truths]

        if args.output_file:
            write_generation_preds(eval_dataset.dataset_walker, args.output_file, dialog_ids, all_sampled_texts)

        if do_evaluate:
            output_eval_file = os.path.join(eval_output_dir, f"eval_results_{args.task}.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results %s *****" % desc)
                writer.write("***** Eval results %s *****\n" % desc)
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

    # Set seed
    set_seed(args.seed)
    
    model_load_kwargs = {}

    accelerator = Accelerator()
    args.device = accelerator.device

    args.output_dir = args.checkpoint
    gen_task = dataset_args.gen_task
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if gen_task.lower() == "seq2seq_lm":
        model_class = AutoModelForSeq2SeqLM
    elif gen_task.lower() == "causal_lm":
        model_class = AutoModelForCausalLM
    else:
        raise ValueError(f"Unknown task {gen_task}")
        
    if args.load_in_8bit:
        model_load_kwargs["device_map"] = "auto"
        model_load_kwargs["load_in_8bit"] = True

    logger.info("Generation parameters %s", args)
    logger.info("Model inference parameters %s", model_load_kwargs)

    if args.use_peft:
            from peft import PeftModel
            model = model_class.from_pretrained(args.model_name_or_path, **model_load_kwargs)
            model = PeftModel.from_pretrained(model, model_id=args.checkpoint)
    else:
        model = model_class.from_pretrained(args.checkpoint, **model_load_kwargs)
    model.resize_token_embeddings(len(tokenizer))

    # Evaluation
    eval_dataset = ResponseGenerationEvalDataset(dataset_args, tokenizer, split_type=args.eval_dataset,
                                                 labels_file=args.labels_file)

    result = evaluate(args, eval_dataset, model, tokenizer, desc=args.eval_desc or "val", accelerator=accelerator, gen_task=gen_task)

    return result


if __name__ == "__main__":
    main()
