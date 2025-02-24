import copy
import random
from collections import defaultdict
from itertools import chain

import torch
from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences
)
from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

SPECIAL_TOKENS = {
    "additional_special_tokens": ["Assistant:", "User:", ":Doc:", "\n\n ## Conversation\n"],
}
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.SPECIAL_TOKENS = SPECIAL_TOKENS

        def list_conv(x):
            if not isinstance(x, list):
                return [x]
            else:
                return x
        self.speaker1, self.speaker2, self.knowledge_sep, self.conv_start = [list_conv(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))) for x in self.SPECIAL_TOKENS["additional_special_tokens"]]

        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]
        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        self.dialogs = self._prepare_conversations()
        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.snippets = self._prepare_knowledge()
        if not hasattr(args, "prompting") or args.prompting not in ["alpaca", "oasst", "vicuna"]:
            self.prompt, self.prompt_postfix = [], []
        else:
            self.prompt, self.prompt_postfix = self._prepare_prompt()
        self._create_examples()

    def _prepare_prompt(self):
        """ Tokenize and encode the instruction-based prompt if necessary"""
        base_prompt = ""
        if self.args.prompting == "oasst":
            base_prompt += "<|prompter|>"
        elif self.args.prompting == "vicuna":
            base_prompt += "USER: "
        base_prompt += "You are a helpful Assistant who can make bookings and reservations. Below is a context with customer reviews and FAQs for hotels and restaurants, paired with a conversation between Assistant and User. If the context info is not sufficient or contradictory, inform User.\n\n"        
        tokenized_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(base_prompt))
        if self.args.prompting == "alpaca":
            base_prompt_postfix = "\n\n## Task\nNow give a concise answer as Assistant.\n\n### Response:"
        elif self.args.prompting == "vicuna":
            base_prompt_postfix = "\n\n## Task\nNow give a concise answer as Assistant.\nASSISTANT:"
        elif self.args.prompting == "oasst":
            base_prompt_postfix = "\n\n## Task\nNow give a concise answer as Assistant.<|endoftext|><|assistant|>"
        else:
            raise NotImplementedError()
        tokenized_prompt_postfix = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(base_prompt_postfix))
        return tokenized_prompt, tokenized_prompt_postfix
    
    def _prepare_conversations(self):
        """ Tokenize and encode the dialog data """
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=False, desc='tokenizing...')):
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                    )
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs

    def _prepare_knowledge(self):
        """ Tokenize and encode the knowledge snippets """
        self.knowledge_docs = self._get_snippet_list()

        tokenized_snippets = defaultdict(dict)
        for snippet_id, snippet in enumerate(self.knowledge_docs):
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
            knowledge = self._knowledge_to_string(snippet["doc"], name=snippet["entity_name"] or "")

            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
            tokenized_snippets[key]['token_ids'] = tokenized_knowledge[:self.args.knowledge_max_tokens]
        return tokenized_snippets

    def _get_snippet_list(self):
        """ Get all knowledge snippets in the dataset """
        result = []
        for domain in self.knowledge_reader.get_domain_list():
            for entity_id in self.knowledge_reader.knowledge[domain].keys():
                for review_doc_id in self.knowledge_reader.get_review_doc_ids(domain, entity_id):
                    review_doc = self.knowledge_reader.get_review_doc(domain, entity_id, review_doc_id)
                    for review_sent_id, review_sent in review_doc['sentences'].items():
                        result.append(
                            {'domain': domain, 'entity_id': entity_id, 'entity_name': review_doc['entity_name'],
                             'doc_id': f"{review_doc_id}-{review_sent_id}",
                             'doc': {'body': review_sent}})
                for faq_doc_id in self.knowledge_reader.get_faq_doc_ids(domain, entity_id):
                    faq_doc = self.knowledge_reader.get_faq_doc(domain, entity_id, faq_doc_id)
                    result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': faq_doc['entity_name'],
                                   'doc_id': faq_doc_id,
                                   'doc': {'body': f"{faq_doc['question']} {faq_doc['answer']}"}})
        return result

    def _knowledge_to_string(self, doc, name=""):
        """ Convert a knowledge snippet to a string """
        doc_body = f"{name.title()}: {doc['body']}"
        return doc_body
    
    def _clean_prefixes_for_generation(self, rel_ids, used_knowledge, same_document, cur_initials):
        """For all the found label sentences from the knowledge we now check
        if we can reduce the token size by removing the similar initials (e.g. Document:Alton House:)
        i f the sentence belongs to the same document"""
        if len(used_knowledge) > 0 and cur_initials == []:
            last_snippet = used_knowledge[-1]
            for (i, token_id) in enumerate(rel_ids):
                if i + len(self.knowledge_sep) < len(last_snippet):
                    if token_id == last_snippet[i + len(self.knowledge_sep)]:
                        cur_initials.append(token_id)
                else:
                    break
        if same_document:
            if rel_ids[:len(cur_initials)] == cur_initials:
                rel_ids = rel_ids[len(cur_initials):]
        else:
            rel_ids = self.knowledge_sep + rel_ids
        return cur_initials, rel_ids

    def _create_examples(self):
        """ Creating examples for model training and evaluation """
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=False, desc='creating examples'):
            if self.args.debug > 0 and len(self.examples) >= self.args.debug:
                break
            dialog_id = dialog["id"]
            label = dialog["label"]

            dialog = dialog["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task != "detection":
                # we only care about non-knowledge-seeking turns in turn detection task
                continue

            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
                for turn in dialog
            ]
            gt_resp = label.get("response", "")
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left 
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

            if target:
                knowledge_keys = []
                knowledge_candidates = defaultdict(lambda: 0)
                used_knowledge = []
                knowledge_prefix_visited = set()

                if "knowledge" not in label:
                    raise ValueError("Please run entity matching before running knowledge selection")

                label_knowledge = label["knowledge"]
                # We sort this by sent_id
                if self.args.clean_knowledge:
                    label_knowledge.sort(key=lambda x: (x['entity_id'], x['doc_type'], x['doc_id'], x.get('sent_id', float('-inf'))))

                same_document = False
                cur_initials = []
                last_doc_id = -1
                for knowledge in label_knowledge:
                    if not (self.args.task == 'selection' and self.args.eval_only):
                        if knowledge['doc_type'] == 'review':
                            knowledge_key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}-{knowledge['sent_id']}"
                            if knowledge['doc_id'] == last_doc_id:
                                same_document = True
                            else:
                                cur_initials = []
                                same_document = False
                        else:
                            knowledge_key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}"

                    # find snippets with same entity as candidates
                    prefix = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
                    if prefix not in knowledge_prefix_visited:
                        knowledge_prefix_visited.add(prefix)
                        _knowledge_candidates = [
                            cand
                            for cand in self.snippets.keys()
                            if "__".join(cand.split("__")[:-1]) == prefix
                        ]

                        for _knowledge_cand_idx, _knowledge_cand in enumerate(_knowledge_candidates):
                            knowledge_candidates[_knowledge_cand] = 1
                    if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                        # if there's not enough candidates during training, we just skip this example
                        if len(knowledge_candidates) < self.args.n_candidates or len(knowledge_candidates) <= len(
                                label["knowledge"]):
                            logger.info("Not enough candidates. Skip this example...")
                            continue

                    if not (self.args.task == 'selection' and self.args.eval_only):
                        rel_ids = self.snippets[knowledge_key]['token_ids'][:self.args.knowledge_max_tokens]

                        if self.args.clean_knowledge:
                            cur_initials, rel_ids = self._clean_prefixes_for_generation(rel_ids, used_knowledge, same_document, cur_initials)

                        used_knowledge.append(rel_ids)
                        knowledge_keys.append(knowledge_key)
                        last_doc_id = knowledge['doc_id']
                knowledge_candidates = [k for k, v in knowledge_candidates.items()]

            else:
                knowledge_candidates = None
                used_knowledge = []
                knowledge_keys = []

            self.examples.append({
                "history": truncated_history,
                "knowledge": used_knowledge,
                "knowledge_keys": knowledge_keys,
                "candidates": knowledge_candidates,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "label": label,
                "knowledge_seeking": target,
                "dialog_id": dialog_id
            })

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class KnowledgeTurnDetectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeTurnDetectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def build_input_from_segments(self, history):
        """ Build a sequence of input from history """
        instance = {}

        sequence = [[self.cls]] + history[:-1] + [history[-1]]
        sequence_with_speaker = []
        for i, s in enumerate(sequence[1:]):
            if (len(sequence) - i) % 2 == 0:
                speaker = self.speaker1
            else:
                speaker = self.speaker2
                # User
            sequence_with_speaker.append(speaker + s)
        sequence0 = [sequence[0]] + sequence_with_speaker[:-1] + [[self.sep]]
        sequence0 = list(chain(*sequence0))
        sequence1 = sequence_with_speaker[-1]

        instance["input_ids"] = sequence0 + sequence1
        instance["token_type_ids"] = [0 for _ in sequence0] + [1 for _ in sequence1]
        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["history"])
        instance["label"] = example["knowledge_seeking"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]
        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, attention_mask, labels, data_info


class KnowledgeSelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            # Negative sampling method for knowledge selection
            # all: use all knowledge snippets of all entities as candidates
            # oracle: use all knowledge snippets of oracle entities as candidates
            # mix: use oracle candidates & equally sized candidates sampled from other entities
            raise ValueError(
                "negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    def _knowledge_to_string(self, doc, name=""):
        """ convert a knowlege snippet to a string """
        join_str = " %s " % self.knowledge_sep_token
        doc_body = doc['body']
        knowledge_string = join_str.join([name.title(), doc_body])
        return knowledge_string

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": []
        }

        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates with no sampling
            if self.args.eval_all_snippets:
                candidates = list(self.snippets.keys())
            else:
                candidates = example["candidates"]
        else:
            if self.args.negative_sample_method == "all":
                candidates = list(self.snippets.keys())
            elif self.args.negative_sample_method == "mix":
                candidates = example["candidates"] + random.sample(list(self.snippets.keys()),
                                                                   k=len(example["candidates"]))
            elif self.args.negative_sample_method == "oracle":
                candidates = example["candidates"]
            else:  # although we have already checked for this, still adding this here to be sure
                raise ValueError(
                    "negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

        candidate_keys = candidates
        this_inst["candidate_keys"] = candidate_keys
        candidates = [self.snippets[cand_key]['token_ids'] for cand_key in candidates]

        if self.split_type == "train":
            candidates = self._shrink_label_cands(example["knowledge"], candidates)

        label_idx = [candidates.index(knowledge) for knowledge in example["knowledge"]]

        this_inst["label_idx"] = label_idx
        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])

        return this_inst

    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}

        sequence = [[self.cls]] + history
        sequence_with_speaker = []
        for i, s in enumerate(sequence[1:]):
            if (len(sequence) - i) % 2 == 0:
                speaker = self.speaker1
            else:
                speaker = self.speaker2
                # User
            sequence_with_speaker.append(speaker + s)

        sequence_with_speaker = list(chain(*sequence_with_speaker))

        sequence0 = [self.cls] + sequence_with_speaker + [self.sep]
        sequence1 = knowledge + [self.sep]

        if 'roberta' in str(type(self.tokenizer)):
            sequence0 += [self.sep]
        instance["input_ids"] = sequence0 + sequence1
        instance["token_type_ids"] = [0 for _ in sequence0] + [1 for _ in sequence1]
        return instance, sequence

    def _shrink_label_cands(self, label, candidates):
        """ remove positive knowledge snippets from the candidates """
        shrunk_label_cands = candidates.copy()
        for l in label:
            if l in shrunk_label_cands:
                shrunk_label_cands.remove(l)
        sample_size = min(len(label), len(shrunk_label_cands))
        shrunk_label_cands = random.sample(shrunk_label_cands, k=sample_size)

        shrunk_label_cands.extend(label)
        random.shuffle(shrunk_label_cands)
        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        label_idx = [1 if i in ins['label_idx'] else 0 for ins in batch for i in range(len(ins['input_ids']))]
        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        token_type_ids = torch.tensor(pad_ids(token_type_ids, 0))
        label_idx = torch.tensor(label_idx)
        return input_ids, token_type_ids, attention_mask, label_idx, data_info


class ResponseGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        prompt = self.prompt
        prompt_postfix = self.prompt_postfix
        instance, _ = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"],
            prompt,
            prompt_postfix
        )
        return instance

    def build_input_from_segments(self, knowledge, history, response, prompt=[], prompt_postfix=[]):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}
        if self.args.clean_knowledge:
            knowledge = [k for k in knowledge]
        else:
            knowledge = [self.knowledge_sep + k for k in knowledge]
        knowledge = [w for k in knowledge for w in k]

        # 3: special tokens; len(history): special speaker tokens; prompt for instruction-based models
        total_prompt_len = len(prompt) + len(prompt_postfix)
        entire_input_len = self.tokenizer.model_max_length - total_prompt_len - 3

        entire_knowledge_len, entire_history_len = len(knowledge), len(list(chain(*history)))
        max_history_len = int((entire_history_len * entire_input_len) / (entire_knowledge_len + entire_history_len))
        max_history_len = min(entire_history_len + len(history), max(max_history_len, 256))
        max_knowledge_len = entire_input_len - max_history_len  # - len(history)

        if max_knowledge_len < entire_knowledge_len:
            logger.warning(
                f"Knowledge too long! Have been truncated from {entire_knowledge_len} to {max_knowledge_len}")
            knowledge = knowledge[:max_knowledge_len]
        if max_history_len < entire_history_len:
            logger.warning(f"History too long! Have been truncated from {entire_history_len} to {max_history_len}")

        sequence = [knowledge] + history + [response]
        full_history = history + [response]
        sequence_with_speaker = []
        for i, s in enumerate(full_history):
            if (len(sequence) - i) % 2 == 0:
                sequence_with_speaker.append(self.speaker1 + s)
            else:
                sequence_with_speaker.append(self.speaker2 + s)
                # User

        if self.args.clean_knowledge:
            history = list(chain(*sequence_with_speaker[:-1]))[-max_history_len:]
        else:
            history = list(chain(*sequence_with_speaker[:-1]))[:max_history_len]

        # If we have a T5 tokenizer, we need to add EOS token to the end of the sequence
        if self.args.debug_fill:
                instance["input_ids"] = [36] * entire_input_len
                instance["lm_labels"] = [36] * entire_input_len
        else:
            if self.args.gen_task.lower() == "seq2seq_lm":
                if 't5' in str(type(self.tokenizer)):
                    sequence = [sequence[0]] + [self.conv_start]  + [history] + [[self.eos]]
                    instance["input_ids"] = list(chain(*sequence))
                    instance["lm_labels"] = sequence_with_speaker[-1] + [self.eos]
                    # else we assume BART architecture with both BOS and EOS
                else:
                    sequence = [[self.bos]] + [sequence[0]] + [self.conv_start] + [history] + [[self.eos]]
                    instance["input_ids"] = list(chain(*sequence))
                    instance["lm_labels"] = [self.bos] + sequence_with_speaker[-1] + [self.eos]
            # For causal LM, we have to copy the input_ids to lm_labels
            elif self.args.gen_task.lower() == "causal_lm":
                # For inference of causal models
                if response == []:
                    end_of_sequence = []
                else:
                    end_of_sequence = [sequence_with_speaker[-1]] + [[self.eos]]
                end_len = len(list(chain(*end_of_sequence)))
                sequence = [prompt] + [sequence[0]] + [self.conv_start] + [history] + [prompt_postfix] + end_of_sequence
                
                instance["input_ids"] = list(chain(*sequence))
                labels = copy.deepcopy(instance["input_ids"])
                label_len = len(labels)
                assert end_len < label_len

                labels[:-end_len] = [IGNORE_INDEX] * (label_len - end_len)
                instance["lm_labels"] = labels

            else:
                raise ValueError(f"Unknown generation task: {self.args.gen_task}")
        return instance, sequence

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        lm_labels = torch.tensor(pad_ids(lm_labels, IGNORE_INDEX))
        assert (lm_labels != -100).any()
        return input_ids, attention_mask, lm_labels


class ResponseGenerationEvalDataset(ResponseGenerationDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationEvalDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch
