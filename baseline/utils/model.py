import copy
import traceback
import torch
from transformers import PreTrainedModel
from ..dataset import IGNORE_INDEX

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="DEBUG")


def run_batch_detection_train(args, model, batch, **kwargs):
    """ Run batch knowledge turn detection during training time """
    cls_loss, cls_logits, labels = run_batch_detection_eval(args, model, batch, **kwargs)
    yield cls_loss, cls_logits, None


def run_batch_detection_eval(args, model, batch, **kwargs):
    """ Run batch knowledge turn detection during evaluation time """
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, attention_mask, labels = batch
    model_outputs = model(
        input_ids=input_ids,
        token_type_ids=None if model.base_model_prefix in ['roberta'] else token_type_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    cls_loss = model_outputs.loss
    cls_logits = model_outputs.logits
    return cls_loss, cls_logits, labels


def run_batch_selection_train(args, model, batch, **kwargs):
    """ Run batch knowledge selection during training time """
    candidates_per_forward = args.max_candidates_per_forward_train
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, attention_mask, labels = batch
    for index in range(0, input_ids.size(0), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[index:index + candidates_per_forward],
            token_type_ids=None if model.base_model_prefix in ['roberta'] else
                           token_type_ids[index:index + candidates_per_forward],
            attention_mask=attention_mask[index:index + candidates_per_forward],
            labels=labels[index:index + candidates_per_forward],
        )
        loss, logits = model_outputs[0], model_outputs[1]
        yield loss, logits, None


def run_batch_selection_eval(args, model, batch, **kwargs):
    """ Run batch knowledge selection during evaluation time """
    # return: loss, logits, labels
    candidates_per_forward = args.max_candidates_per_forward_eval
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, attention_mask, labels = batch
    original_labels = copy.deepcopy(labels)

    all_logits = []
    eval_loss = 0
    for index in range(0, input_ids.size(0), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[index:index + candidates_per_forward],
            token_type_ids=None if model.base_model_prefix in ['roberta'] else
                           token_type_ids[index:index + candidates_per_forward],
            attention_mask=attention_mask[index:index + candidates_per_forward],
            labels=labels[index:index + candidates_per_forward]
        )
        eval_loss += model_outputs.loss * len(input_ids[index:index + candidates_per_forward])
        logits = model_outputs.logits
        all_logits.append(logits.detach())
    all_logits = torch.cat(all_logits, dim=0)
    return eval_loss, all_logits, original_labels


def run_batch_generation_train(args, model, batch, batch_n=-1, **kwargs):
    """ Run batch generation during training time """
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch[:4])
    input_ids, attention_mask, lm_labels = batch
    tokenizer = kwargs.get('tokenizer')  # Get the tokenizer from kwargs
    if batch_n % 75 == 0:
        logger.debug(f"In current batch using as (decoded) input {[tokenizer.decode(ids) for ids in input_ids]}")
        for l in lm_labels:
            l = torch.where(l != IGNORE_INDEX, l, tokenizer.pad_token_id)
            logger.debug(tokenizer.decode(l))

    try:
        model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels)
    except ValueError as e:
        traceback.print_exc()
        print("Exception occurred:", str(e))
        print("input_ids shape: ", input_ids.shape)
        print("attention_mask shape: ", attention_mask.shape)
        print("lm_labels shape: ", lm_labels.shape)
        print("Decoded input_ids: ")
        for i in input_ids:
            print(tokenizer.decode(i))
            
        print("Decoded lm_labels: ")
        for l in lm_labels:
            l = torch.where(l != IGNORE_INDEX, l, tokenizer.pad_token_id)
            print(tokenizer.decode(l))
            
    loss = model_outputs[0]
    lm_logits = model_outputs[1]
    yield loss, lm_logits, torch.tensor([])


def run_batch_generation_eval(args, model, batch, **kwargs):
    """ Run batch generation during evaluation time """
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch[:4])
    input_ids, attention_mask, lm_labels = batch
    model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels)
    loss = model_outputs[0]
    lm_logits = model_outputs[1]
    return loss, lm_logits, torch.tensor([])


def run_batch_generation_sample(args, model, tokenizer, batch, dataset, accelerator=None, gen_task="seq2seq_lm"):
    """ Run batch generation during test time
        Responses are decoded using beam search + sampling
    """
    current_output = []

    example = batch[0]
    knowledge, history = example["knowledge"], example["history"]
    response_text = example["response_text"]
    dialog_id = example["dialog_id"]

    instance, sequence = dataset.build_input_from_segments(
        knowledge, history, response=[], prompt=dataset.prompt, prompt_postfix=dataset.prompt_postfix
    )

    input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
    input_text = tokenizer.decode(instance["input_ids"])
    attention_mask = 1 - (input_ids == tokenizer.pad_token_id).int()

    if gen_task.lower() == "causal_lm":
        current_output = model.generate(input_ids=input_ids, num_beams=args.num_beams, attention_mask=attention_mask,
                                        min_new_tokens=args.min_length, max_new_tokens=args.max_length,
                                        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                                    pad_token_id=tokenizer.pad_token_id, do_sample=args.do_sample, num_return_sequences=1)
        input_len = len(instance["input_ids"])
        current_output = current_output[:, input_len:]
    else:
        current_output = model.generate(input_ids=input_ids, num_beams=args.num_beams,
                                min_length=args.min_length, max_length=args.max_length,
                                eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                            pad_token_id=tokenizer.pad_token_id, do_sample=args.do_sample, num_return_sequences=1)
    return current_output, response_text, dialog_id
