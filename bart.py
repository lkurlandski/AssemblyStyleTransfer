"""

Source:
    https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_bart_dlm_flax.py
"""

from datetime import datetime
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from pprint import pformat, pprint
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch
from torch import tensor
from tqdm import tqdm

from datasets import load_dataset, Dataset, DatasetDict
import evaluate
import transformers
from transformers import (
    is_tensorboard_available,
    set_seed,
    AutoTokenizer,
    BartConfig,
    BatchEncoding,
    BartForConditionalGeneration,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.utils import get_full_repo_name, send_example_telemetry

import cfg
import tokenization
from utils import count_parameters, mem, OutputManager


BR = "|" + "-" * 88 + "|"
VERBOSE_DATASET_STATS = False


@dataclass
class ModelArguments:
    encoder_layers: int = field(default=12, metadata={"help": ""})
    encoder_ffn_dim: int = field(default=4096, metadata={"help": ""})
    encoder_attention_heads: int = field(default=16, metadata={"help": ""})
    decoder_layers: int = field(default=12, metadata={"help": ""})
    decoder_ffn_dim: int = field(default=4096, metadata={"help": ""})
    decoder_attention_heads: int = field(default=16, metadata={"help": ""})
    d_model: int = field(default=1024, metadata={"help": ""})
    downsize: Optional[int] = field(default=None, metadata={"help": "1 -> 370M; 2 -> 50.M; 4 -> 8.7M"})

    def __post_init__(self):
        if self.downsize:
            self.encoder_layers = self.encoder_layers // self.downsize
            self.encoder_ffn_dim = self.encoder_ffn_dim // self.downsize
            self.encoder_attention_heads = self.encoder_attention_heads // self.downsize
            self.decoder_layers = self.decoder_layers // self.downsize
            self.decoder_ffn_dim = self.decoder_ffn_dim // self.downsize
            self.decoder_attention_heads = self.decoder_attention_heads // self.downsize
            self.d_model = self.d_model // self.downsize


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "5% < 26; 10% < 37; 15% < 46; 20% < 57; 25% < 68; 30% < 79; 35% < 92; "
            "40% < 107; 45% < 125; 50% < 144; 55% < 165; 60% < 192; 65% < 221; "
            "70% < 255; 75% < 303; 80% < 367; 85% < 471; 90% < 647; 95% < 1036"
        },
    )
    n_files: Optional[int] = field(default=None, metadata={"help": ""})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": ""})
    mlm_probability: float = field(default=0.3, metadata={"help": ""})
    permute_sentence_ratio: float = field(default=1.0, metadata={"help": ""})
    poisson_lambda: float = field(default=3.0, metadata={"help": ""})


@dataclass
class TokenizerArguments:
    tok_model: str = field(default="WordLevel", metadata={"help": ""})
    tok_vocab_size: int = field(default=None, metadata={"help": ""})
    tok_use_cached: bool = field(default=True, metadata={"help": ""})
    tok_overwrite: bool = field(default=False, metadata={"help": ""})
    tok_batch_size: int = field(default=512, metadata={"help": ""})


@dataclass
class DataCollatorForBartDenoisingLM:
    max_length: int
    padding = "max_length"
    pad_to_multiple_of = None
    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0
    permute_sentence_ratio: float = 1.0

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token or eos token token which is necessary for denoising"
                " language modeling. "
            )

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        encodings = [BatchEncoding({"input_ids": examples[i]["input_ids"]}) for i in range(len(examples))]
        batch = self.tokenizer.pad(
            encodings,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.clone(batch["input_ids"])
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )
        # permuting sentences
        do_permute = False
        if self.permute_sentence_ratio > 0.0:
            batch["input_ids"] = tensor(self.permute_sentences(batch["input_ids"].cpu().detach().numpy()))
            do_permute = True

        # masking span of tokens (text infilling in the paper)
        if self.mask_ratio:
            tmp_input_ids, tmp_labels = self.span_mask_tokens(
                batch["input_ids"].cpu().detach().numpy(),
                batch["labels"].cpu().detach().numpy(),
                do_permute,
            )
            batch["input_ids"], batch["labels"] = tensor(tmp_input_ids), tensor(tmp_labels)

        # ignore pad tokens
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).int()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).int()
        return batch

    def permute_sentences(self, input_ids):
        """
        Shuffle sentences in each document.
        """
        results = input_ids.copy()

        # find end locations of sentences
        end_sentence_mask = input_ids == self.tokenizer.pad_token_id
        sentence_ends = np.argwhere(end_sentence_mask)
        sentence_ends[:, 1] += 1
        example_has_multiple_sentences, num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)
        num_sentences_map = dict(zip(example_has_multiple_sentences, num_sentences))

        num_to_permute = np.ceil(num_sentences * self.permute_sentence_ratio).astype(int)
        num_to_permute_map = dict(zip(example_has_multiple_sentences, num_to_permute))

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])
        sentence_ends_map = dict(zip(example_has_multiple_sentences, sentence_ends))

        for i in range(input_ids.shape[0]):
            if i not in example_has_multiple_sentences:
                continue
            substitutions = np.random.permutation(num_sentences_map[i])[: num_to_permute_map[i]]
            ordering = np.arange(0, num_sentences_map[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute_map[i])]

            # write shuffled sentences into results
            index = 0
            for j in ordering:
                sentence = input_ids[i, (sentence_ends_map[i][j - 1] if j > 0 else 0) : sentence_ends_map[i][j]]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def span_mask_tokens(self, input_ids, labels, do_permute):
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """

        special_tokens_mask_labels = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask_inputs = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ]
        special_tokens_mask_labels = np.array(special_tokens_mask_labels, dtype=bool)
        special_tokens_mask_inputs = np.array(special_tokens_mask_inputs, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token_mask = ~(input_ids == self.tokenizer.pad_token_id) & ~special_tokens_mask_inputs
        num_tokens_to_mask = int(math.ceil(is_token_mask.astype(float).sum() * self.mask_ratio))
        if num_tokens_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        span_lengths = np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))
        while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
            span_lengths = np.concatenate(
                [span_lengths, np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))]
            )

        # remove all spans of length 0
        # note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        span_lengths = span_lengths[span_lengths > 0]

        # trim to about num_tokens_to_mask tokens
        cutoff_idx = np.argmin(np.abs(np.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
        span_lengths = span_lengths[:cutoff_idx]

        # randomly choose starting positions for masking
        token_indices = np.argwhere(is_token_mask == 1)
        span_starts = np.random.permutation(token_indices.shape[0])[: span_lengths.shape[0]]
        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(input_ids, fill_value=False)

        # mask starting positions
        for mi in masked_indices:
            mask[tuple(mi)] = True
        span_lengths -= 1

        # fill up spans
        max_index = input_ids.shape[1] - 1
        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            span_lengths -= 1
            remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask_inputs)] = False
        input_ids[np.where(mask)] = self.tokenizer.mask_token_id
        if not do_permute:
            labels[np.where(mask == 0)] = -100
        else:
            labels[np.where(special_tokens_mask_labels)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_input_ids = np.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(input_ids):
            new_example = example[~to_remove[i]]
            new_input_ids[i, : new_example.shape[0]] = new_example

        return new_input_ids, labels


def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


def get_compute_metric_fn(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def get_stats(lengths: list[float | int]) -> tuple:
    return (
        statistics.mean(lengths),
        statistics.median(lengths),
        statistics.stdev(lengths),
        min(lengths),
        max(lengths),
        statistics.quantiles(lengths, n=20),
    )


def str_stats(stats: tuple, digits: Optional[int] = 0):
    return (
        f"mean={round(stats[0], digits)} "
        f"median={round(stats[1], digits)} "
        f"stdev={round(stats[2], digits)} "
        f"min={round(stats[3], digits)} "
        f"max={round(stats[4], digits)} "
        f"quantiles={tuple(round(q, digits) for q in stats[5])}"
    )


def get_dataset(
    files: list[Path],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    n_workers: int = 1,
    verbose: bool = True,
) -> DatasetDict:
    def split_instructions_fn(example):
        # cannot be used with map and batched=True
        # use pad token as end of sentence indicator
        sents = example["text"].split(cfg.INS)
        new_text = tokenizer.bos_token + f"{tokenizer.pad_token}".join(sents) + tokenizer.eos_token
        return {"text": new_text}

    def tokenize_fn(examples):
        """Tokenize every text, then concatenate them together before splitting
        them in smaller parts. Since we make sure that all sequences are of the
        same length, no attention_mask is needed."""
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )

    def group_fn(examples):
        """Main data processing function that will concatenate all
        texts from our dataset and generate chunks of max_seq_length."""
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    raw_datasets = tokenization.get_raw_assembly_dataset(files, min_lines=4, num_proc=n_workers).train_test_split()
    if verbose:
        print(f"{raw_datasets=}\n{raw_datasets['train'][0]}", flush=True)

    split_datasets = raw_datasets.map(
        split_instructions_fn,
        batched=False,
        remove_columns="text",
    )
    if verbose:
        print(f"{split_datasets=}\n{split_datasets['train'][0]}", flush=True)

    tokenized_datasets = split_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns="text",
    )
    if verbose:
        print(f"{tokenized_datasets=}\n{tokenized_datasets['train'][0]}", flush=True)
        if VERBOSE_DATASET_STATS:
            lengths = [len(d) for d in tokenized_datasets["test"]["input_ids"]]
            print(f"input_ids:\n{str_stats(get_stats(lengths))}\n{BR}", flush=True)

    grouped_datasets = tokenized_datasets.map(group_fn, batched=True)
    return grouped_datasets


def main():
    slurm_env_variables = [
        "SLURM_JOB_NUM_NODES",
        "SLURM_TASKS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
    ]
    for v in slurm_env_variables:
        print(f"{v}={os.environ.get(v, None)}")
    print(f"{BR}", flush=True)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TokenizerArguments, Seq2SeqTrainingArguments))
    model_args, data_args, tokenizer_args, training_args = parser.parse_args_into_dataclasses()
    print(f"{model_args=}\n{data_args=}\n{tokenizer_args}\n{training_args=}\n{BR}", flush=True)

    set_seed(training_args.seed)
    om = OutputManager()

    files = sorted(list(om.merged.glob("*.txt")))[0 : data_args.n_files]
    print(f"{len(files)=}\n{round(mem(files), 2)=}\n{BR}", flush=True)

    print(f"TOKENIZATION START @{datetime.now()}\n{BR}", flush=True)
    _tokenizer = tokenization.get_tokenizer(
        om.tokenizers,
        tokenizer_args.tok_model,
        files,
        tokenizer_args.tok_batch_size,
        tokenizer_args.tok_vocab_size,
        tokenizer_args.tok_use_cached,
        tokenizer_args.tok_overwrite,
    )
    tokenizer = tokenization.get_fast_tokenizer(
        _tokenizer,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
    )
    print(f"TOKENIZATION FINISH @{datetime.now()}\n{BR}", flush=True)
    print(f"{tokenizer=}\n{BR}", flush=True)

    print(f"DATA PROCESSING START @{datetime.now()}\n{BR}", flush=True)
    datasets = get_dataset(
        files,
        tokenizer,
        data_args.max_seq_length,
        data_args.preprocessing_num_workers,
        verbose=True,
    )
    print(f"DATA PROCESSING FINISH @{datetime.now()}\n{BR}", flush=True)
    print(f"{datasets=}\n{datasets['train'][0]}\n{BR}", flush=True)
    if VERBOSE_DATASET_STATS:
        lengths = [len(d) for d in datasets["test"]["input_ids"]]
        print(f"input_ids:\n{str_stats(get_stats(lengths))}\n{BR}", flush=True)

    config = BartConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=data_args.max_seq_length,
        encoder_layers=model_args.encoder_layers,
        encoder_ffn_dim=model_args.encoder_ffn_dim,
        encoder_attention_heads=model_args.encoder_attention_heads,
        decoder_layers=model_args.decoder_layers,
        decoder_ffn_dim=model_args.decoder_ffn_dim,
        decoder_attention_heads=model_args.decoder_attention_heads,
        d_model=model_args.d_model,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
    )
    model = BartForConditionalGeneration(config)
    print(f"{config=}\n{count_parameters(model)=}\n{BR}", flush=True)

    # This one will take care of randomly masking the tokens and permuting the sentences.
    data_collator = DataCollatorForBartDenoisingLM(
        max_length=data_args.max_seq_length,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        mask_ratio=data_args.mlm_probability,
        poisson_lambda=data_args.poisson_lambda,
        permute_sentence_ratio=data_args.permute_sentence_ratio,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    print(f"{data_collator=}\n{callbacks=}\n{BR}", flush=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=get_compute_metric_fn(tokenizer),
        callbacks=callbacks,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if training_args.do_train:
        print(f"TRAINING START @{datetime.now()}\n{BR}", flush=True)
        trainer.train()
        print(f"TRAINING FINISH @{datetime.now()}\n{BR}", flush=True)


if __name__ == "__main__":
    print(f"{BR}\nSTARTS @{datetime.now()}\n{BR}", flush=True)
    main()
    print(f"{BR}\nFINISH @{datetime.now()}\n{BR}", flush=True)
