import argparse
from datetime import timedelta
import logging
import os
import random
import json
import shutil
import inspect
from typing import Dict, Tuple

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed, DistributedType
import transformers
from argparse import Namespace
from .utils.metrics import print_gpu_utilization

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, average_precision_score, classification_report, f1_score
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import get_peft_model_state_dict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
    )

from .dataset import (
    KnowledgeTurnDetectionDataset,
    KnowledgeSelectionDataset,
    ResponseGenerationDataset,
    SPECIAL_TOKENS,
    DEFAULT_PAD_TOKEN
)
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from .utils.model import (
    run_batch_detection_train,
    run_batch_detection_eval,
    run_batch_selection_train,
    run_batch_selection_eval,
    run_batch_generation_train,
    run_batch_generation_eval,
)
from .utils.data import write_selection_preds, write_detection_preds

from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

os.environ['TRANSFORMERS_OFFLINE'] = '1'
MAX_DESIRED_LENGTH = int(os.getenv('MAX_DESIRED_LEN', 1024))


def find_all_linear_names(args, model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_cls_report(y_true, y_pred):
    """ Get the report of precision, recall, and f1-score for a classification output """
    return {"precision": precision_score(y_true, y_pred, average=None, zero_division=0)[1],
            "recall": recall_score(y_true, y_pred, average=None, zero_division=0)[1],
            "f1-score": f1_score(y_true, y_pred, average=None, zero_division=0)[1]}


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    smart_init=False
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if smart_init and num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_classes(args):
    """ Get classes for dataset, model, training func, and eval func """
    task, model = args.task, args.model_name_or_path
    if task.lower() == "generation":
        gen_task = args.dataset_args["gen_task"]
        if gen_task.lower() == "seq2seq_lm":
            return ResponseGenerationDataset, AutoModelForSeq2SeqLM, run_batch_generation_train, run_batch_generation_eval
        elif gen_task.lower() == "causal_lm":
            return ResponseGenerationDataset, AutoModelForCausalLM, run_batch_generation_train, run_batch_generation_eval
        else:
            raise ValueError("args.gen_task not in ['seq2seq_lm', 'causal_lm'], got %s" % gen_task)
    elif task.lower() == "selection":
        return KnowledgeSelectionDataset, AutoModelForSequenceClassification, run_batch_selection_train, run_batch_selection_eval
    elif task.lower() == 'detection':
        return KnowledgeTurnDetectionDataset, AutoModelForSequenceClassification, run_batch_detection_train, run_batch_detection_eval
    else:
        raise ValueError(
            "args.task not in ['generation_review', 'selection_review', 'detection_review'], got %s" % task)


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
          run_batch_fn_train, run_batch_fn_eval, accelerator=None) -> Tuple[int, float]:
    """ Model training and evaluation """
    log_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir)

    args.train_batch_size = args.per_device_train_batch_size

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if 0 < args.warmup_steps < 1:
        args.warmup_steps = int(args.warmup_steps * t_total)

    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=False
    )
    set_seed(args.seed)  # for reproducibility
    val_loss = float('inf')

    # Calculate perplexity of base model
    results = evaluate(args, eval_dataset, model, run_batch_fn_eval, desc=str(global_step), accelerator=accelerator)
    if accelerator.is_main_process:
        for key, value in results.items():
            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

    for epoch in train_iterator:
        local_steps = 0  # update step
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for (batch_n, batch) in enumerate(epoch_iterator):
            with accelerator.accumulate(model):
                model.train()
                for loss, _, _ in run_batch_fn_train(args, model, batch, batch_n=batch_n, global_step=global_step, tokenizer=tokenizer):
                    accelerator.backward(loss)
                    tr_loss += loss.item()

                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    local_steps += 1
                    epoch_iterator.set_postfix(Loss=tr_loss / local_steps)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)

        if epoch == 0:
            # Save model checkpoint at least once
            accelerator.wait_for_everyone()
            save_model(args, args.output_dir, model, tokenizer, accelerator=accelerator)


        # Evaluate at the end of each epoch
        results = evaluate(args, eval_dataset, model, run_batch_fn_eval, desc=str(global_step), accelerator=accelerator)
        
        if accelerator.is_main_process:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)

        # Only save model if validation loss has improved
        accelerator.wait_for_everyone()

        save_happening = False
        if accelerator.is_main_process:
            if results['val_measure'] < val_loss:
                logger.info(f"Found a smaller val loss measure {results['val_measure']}")
                val_loss = results['val_measure']
                save_happening = True

        if save_happening:
            save_model(args, args.output_dir, model, tokenizer, accelerator=accelerator)

        else:
            if accelerator.is_main_process:
                logger.info(f"The val loss measure {results['val_measure']} is larger than "
                            f"the smallest val loss {val_loss}, continue to train ... ")

    print_gpu_utilization(args, "after finishing the training")
    tb_writer.flush()
    tb_writer.close()

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        output_dir = args.output_dir
        logger.info("Converting fp32 state dict weights %s", output_dir)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(output_dir) # already on cpu
        if args.use_peft:
            state_dict = get_peft_model_state_dict(model, state_dict=state_dict)
            accelerator.save(state_dict, os.path.join(output_dir, "adapter_model.bin"))
        else:
            accelerator.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        all_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        for directory in all_dirs:
            if directory.startswith("global_steps"):
                shutil.rmtree(os.path.join(output_dir, directory))

    return global_step, tr_loss / local_steps


def save_model(args, output_dir, model, tokenizer, accelerator=None):
    """ Save model, tokenizer, and params to the output dir """
    os.makedirs(output_dir, exist_ok=True)
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        logger.info("Saving deepspeed model checkpoints to %s", output_dir)
        # When Zero 3 is used we need to gather the truncated adapter bins from multiple GPUs for Lora compatibility
        model.save_checkpoint(output_dir)
        accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        logger.info("Saving model checkpoint to %s", output_dir)
        model.save_pretrained(output_dir)

        tokenizer.save_pretrained(output_dir)
        accelerator.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
        logger.info("Saving model config to %s", output_dir)


def evaluate(args, eval_dataset, model: PreTrainedModel, run_batch_fn, desc="", accelerator=None) -> Dict:
    """ Model evaluation for knowledge seeking turn detection and knowledge selection
        Report evaluation results if gold labels are available
    """
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle different number of candidates
    args.eval_batch_size = 1

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    eval_dataloader = accelerator.prepare(eval_dataloader)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=False):
        with torch.no_grad():
            loss, logits, labels = run_batch_fn(args, model, batch)

            if args.task in ["selection", "detection"]:
                data_infos.append(batch[-1])
                all_preds.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
            eval_loss += loss.item()
        nb_eval_steps += 1

    # Eval loss could be slightly off with multiple GPUs due to duplicates on batch distribution
    all_losses = accelerator.gather(torch.tensor(eval_loss).to(accelerator.device))
    all_steps = len(eval_dataloader.dataset)
    eval_loss = torch.sum(all_losses) / all_steps

    if args.task == "generation":
        pass
    elif args.task == "selection":
        if args.output_file:
            eval_threshold = args.params["dataset_args"]["eval_threshold"]
            sorted_pred_ids = [np.argsort(logits.squeeze())[::-1][:(logits > eval_threshold).sum()] for logits in all_preds]
            write_selection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, sorted_pred_ids,
                                  all_preds=all_preds)
    elif args.task == "detection":
        all_pred_ids = np.where(np.concatenate(all_preds) > 0, 1, 0)
        if args.output_file:
            write_detection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, all_pred_ids)
    else:
        raise ValueError("args.task not in ['generation', 'selection', 'detection'], got %s" % args.task)

    if not args.eval_only:
        if accelerator.is_main_process:
            return get_eval_performance(args, eval_output_dir, eval_loss, all_preds, all_labels, desc)


def get_eval_performance(args, eval_output_dir, eval_loss, all_preds, all_labels, desc):
    """ Get evaluation performance when the gold labels are available """
    if args.task == "generation":
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {"perplexity": perplexity, "loss": eval_loss, "val_measure": eval_loss}

    elif args.task == "selection":
        def get_eval_performance_selection(all_preds, all_labels, threshold):
            report_list, average_precison_list = [], []
            all_preds_micro, all_labels_micro = [], []
            for preds, labels in zip(all_preds, all_labels):
                preds = preds.reshape(-1)
                preds_labels = np.zeros_like(preds)
                preds_labels[preds > threshold] = 1
                average_precison_list.append(average_precision_score(labels, preds))
                all_preds_micro.extend(preds_labels)
                all_labels_micro.extend(labels)
                report_list.append(get_cls_report(labels, preds_labels))

            cls_report = get_cls_report(all_labels_micro, all_preds_micro)
            result = {"loss": eval_loss, "mean_ave_prec": np.mean(average_precison_list),
                      "micro_prec": cls_report['precision'],
                      "micro_recall": cls_report['recall'],
                      "micro_f1": cls_report['f1-score'],
                      "macro_prec": np.mean([report['precision'] for report in report_list]),
                      "macro_recall": np.mean([report['recall'] for report in report_list]),
                      "macro_f1": np.mean([report['f1-score'] for report in report_list]),
                      "val_measure": -1 * np.mean([report['f1-score'] for report in report_list]),
                      }
            return result

        best_result, best_threshold = {"val_measure": 0}, None
        for threshold in list(np.linspace(-5, 5, num=41)):
            result = get_eval_performance_selection(all_preds, all_labels, threshold)
            if result['val_measure'] < best_result['val_measure']:
                best_result, best_threshold = result, threshold
        best_result.update({"threshold": best_threshold})
        args.params["dataset_args"]["eval_threshold"] = best_threshold
        result = best_result

    elif args.task == "detection":
        all_labels = np.concatenate(all_labels)
        all_pred_ids = np.where(np.concatenate(all_preds) > 0, 1, 0)
        print(classification_report(all_labels, all_pred_ids, labels=[0, 1]))
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        report = get_cls_report(all_labels, all_pred_ids)
        result = {"loss": eval_loss, "val_measure": -1 * report['f1-score'], "accuracy": accuracy,
                  "precision": report['precision'], "recall": report['recall'], 'f1-score': report['f1-score']}
    else:
        raise ValueError("args.task not in ['generation', 'selection', 'detection'], got %s" % args.task)

    logger.info(str(result))

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results %s *****" % desc)
        writer.write("***** Eval results %s *****\n" % desc)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--model_name_or_path", type=str, help="model_name_or_path")
    parser.add_argument("--gen_task", type=str, choices=("causal_lm", "seq2seq_lm"),
                        help="Specify the way data is processed for generation task.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--task", type=str, choices=('detection', 'selection', 'generation'),
                        help="to specify the task. Will overwrite the setting in params.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Saved checkpoint directory")
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--knowledge_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--no_labels", action="store_true",
                        help="Read a dataset without labels.json. This option is useful when running "
                             "knowledge-seeking turn detection on test dataset where labels.json is not available.")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                             "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["all", "mix", "oracle"],
                        default="",
                        help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--debug", type=int, default=0,
                        help="If set, will only use a small number (==debug) of data for training and test.")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--debug_fill", action="store_true", help="If set, will fill all inputs by max_desired_len up with a random number")

    args = parser.parse_args()

    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        # Set dtype for the model weights
        args.update(params)
        args["torch_dtype"] = torch.float32 if not args["fp16"] else torch.float16
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)

    set_default_dataset_params(dataset_args)
    dataset_args.task = args.task
    dataset_args.eval_only = args.eval_only
    dataset_args.debug = args.debug

    peft_args = Namespace(**{})
    if args.use_peft:
        peft_args = Namespace(**args.peft_args)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        )

    set_seed(args.seed)
    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(args)

    model_load_kwargs = {"torch_dtype": args.torch_dtype, "low_cpu_mem_usage": accelerator.distributed_type != DistributedType.DEEPSPEED and "bart" not in args.model_name_or_path}
    if args.load_in_8bit:
        model_load_kwargs["device_map"] = "auto"
        model_load_kwargs["load_in_8bit"] = True

    logger.info("Training/evaluation parameters %s", args)
    
    if args.use_peft:
        from peft import LoraConfig, prepare_model_for_int8_training, get_peft_model, TaskType, PeftModel
        if args.task == "generation":
            if dataset_args.gen_task in ["causal_lm"]:
                lora_task_type = TaskType.CAUSAL_LM
            elif dataset_args.gen_task in ["seq2seq_lm"]:
                lora_task_type = TaskType.SEQ_2_SEQ_LM
            else:
                raise NotImplementedError()
            if peft_args.target_modules == "linear":
                peft_args.target_modules = find_all_linear_names(args, model_class.from_pretrained(args.model_name_or_path, **model_load_kwargs))
            peft_config = LoraConfig(
                    task_type=lora_task_type, inference_mode=False, r=peft_args.lora_r, lora_alpha=peft_args.lora_alpha,
                    lora_dropout=peft_args.lora_dropout,
                    target_modules=peft_args.target_modules,
                    modules_to_save=peft_args.modules_to_save
                )
        else:
            raise NotImplementedError("PEFT is only supported for generation task.")
        
    args.device = accelerator.device
    args.output_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
    print_gpu_utilization(args, "after loading the drivers")

    if args.eval_only:
        args.output_dir = args.checkpoint
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        if args.use_peft:
            model = model_class.from_pretrained(args.model_name_or_path, **model_load_kwargs)
            if args.load_in_8bit:
                model = prepare_model_for_int8_training(model, peft_config)
            model = PeftModel.from_pretrained(model, model_id=args.checkpoint)
        else:
            model = model_class.from_pretrained(args.checkpoint, **model_load_kwargs)

        model = model.to(args.device)


        # Evaluation
        eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset,
                                     labels=not args.no_labels, labels_file=args.labels_file)
        result = evaluate(args, eval_dataset, model, run_batch_fn_eval, desc=args.eval_desc or "val", accelerator=accelerator)
        return result

    else:
        if args.checkpoint is not None:
            if args.use_peft:
                model = model_class.from_pretrained(args.model_name_or_path, **model_load_kwargs)
                if args.gradient_checkpointing:
                    model.enable_input_require_grads()
                model = PeftModel.from_pretrained(model, model_id=args.checkpoint)
            else:
                model = model_class.from_pretrained(args.checkpoint, **model_load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            logger.info(f"Loaded config class: {type(config)}")
            model = model_class.from_pretrained(args.model_name_or_path, config=config, **model_load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            logger.info(f"Loaded tokenizer: {type(tokenizer)}")
            tokenizer.model_max_length = min(MAX_DESIRED_LENGTH, tokenizer.model_max_length)

            if args.special_tokens:
                if tokenizer.pad_token is None:
                    SPECIAL_TOKENS["pad_token"] = DEFAULT_PAD_TOKEN

                logger.info(f"Tokenizer sequence length set to {min(MAX_DESIRED_LENGTH, tokenizer.model_max_length)}")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=SPECIAL_TOKENS,
                    tokenizer=tokenizer,
                    model=model,
                    smart_init=args.smart_init
                )
            
            else:
                if tokenizer.pad_token is None:
                    tokenizer.pad_token_id = tokenizer.unk_token_id

            if args.use_peft:
                if args.gradient_checkpointing:
                    model.enable_input_require_grads()
                model = get_peft_model(model, peft_config)

        model = model.to(args.device)
        print_gpu_utilization(args, "after loading the model weights")

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # load datasets and train the model
        train_dataset = dataset_class(dataset_args, tokenizer, split_type="train")
        eval_dataset = dataset_class(dataset_args, tokenizer,
                                     split_type="val")  # main difference is during evaluation, val need to go through all snippets
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train,
                                     run_batch_fn_eval, accelerator=accelerator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
