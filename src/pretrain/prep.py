"""
Utilities for pretraining scripts.

"5% < 26; 10% < 37; 15% < 46; 20% < 57; 25% < 68; 30% < 79; 35% < 92; "
"40% < 107; 45% < 125; 50% < 144; 55% < 165; 60% < 192; 65% < 221; "
"70% < 255; 75% < 303; 80% < 367; 85% < 471; 90% < 647; 95% < 1036"
"""

from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets import DatasetDict
from transformers import EarlyStoppingCallback, PreTrainedTokenizerBase, TrainerCallback

from src.cfg import *
from src.output_manager import OutputManager
from src.tokenization import (
    get_tokenizer,
    get_fast_tokenizer,
    get_raw_assembly_dataset,
)
from src.utils import disk_usage


@dataclass
class TokenizerArguments:
    max_length: Optional[int] = field(default=128, metadata={"help": ""})
    vocab_size: Optional[int] = field(default=1024, metadata={"help": ""})
    tok_algorithm: Optional[str] = field(default="WordLevel", metadata={"help": ""})
    tok_use_cached: Optional[bool] = field(default=True, metadata={"help": ""})
    tok_overwrite: Optional[bool] = field(default=False, metadata={"help": ""})
    tok_batch_size: Optional[int] = field(default=512, metadata={"help": ""})
    tok_n_files: Optional[int] = field(default=None, metadata={"help": ""})


@dataclass
class DatasetArguments:
    dat_n_examples: Optional[int] = field(default=None, metadata={"help": ""})
    dat_n_files: Optional[int] = field(default=None, metadata={"help": ""})
    num_proc: Optional[int] = field(default=None, metadata={"help": ""})
    validation_split: Optional[float] = field(default=0.1, metadata={"help": ""})


def get_callbacks(patience: int = 5, threshold: int = 0.001) -> list[TrainerCallback]:
    return [EarlyStoppingCallback(patience, threshold)]


def get_dataset(
    files: list[Path],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_proc: int = 1,
) -> DatasetDict:
    def split_instructions_fn(example):
        # cannot be used with map and batched=True
        # use pad token as end of sentence indicator
        s = example["text"].split(INS)
        s = f"{tokenizer.pad_token}".join(s)
        s = tokenizer.bos_token + s + tokenizer.eos_token
        return {"text": s}

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
        texts from our dataset and generate chunks of max_length."""
        # Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the small remainder
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    raw_datasets = get_raw_assembly_dataset(files, min_lines=4, num_proc=num_proc)
    print(f"{raw_datasets=}")
    print(f"{raw_datasets[0]}")
    print(BR, flush=True)

    split_datasets = raw_datasets.map(
        split_instructions_fn,
        batched=False,
        remove_columns="text",
    )
    print(f"{split_datasets=}")
    print(f"{split_datasets[0]}")
    print(BR, flush=True)

    tokenized_datasets = split_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns="text",
    )
    print(f"{tokenized_datasets=}")
    print(f"{tokenized_datasets[0]}")
    print(BR, flush=True)

    grouped_datasets = tokenized_datasets.map(group_fn, batched=True)
    return grouped_datasets


def get_tokenizer_and_dataset(
    tokenizer_args: TokenizerArguments,
    dataset_args: DatasetArguments,
) -> tuple[PreTrainedTokenizerBase, DatasetDict]:
    om = OutputManager()
    files = sorted(list(om.merged.glob("*.txt")))
    print(f"{len(files)=}")
    print(f"{round(disk_usage(files), 2)=}")
    print(BR, flush=True)

    tokenizer = get_tokenizer(
        om.tokenizers,
        tokenizer_args.tok_algorithm,
        files[:tokenizer_args.tok_n_files],
        tokenizer_args.tok_batch_size,
        tokenizer_args.vocab_size,
        tokenizer_args.tok_use_cached,
        tokenizer_args.tok_overwrite,
    )
    tokenizer = get_fast_tokenizer(
        tokenizer,
        model_max_length=tokenizer_args.max_length,
        padding_side="right",
    )
    print(f"{tokenizer=}")
    print(BR, flush=True)

    dataset = get_dataset(
        files[:dataset_args.dat_n_files],
        tokenizer,
        tokenizer_args.max_length,
        dataset_args.num_proc,
    )
    if dataset_args.dat_n_examples:
        dataset = dataset.select(range(dataset_args.dat_n_examples))
    dataset = dataset.train_test_split(dataset_args.validation_split)
    print(f"{dataset=}")
    print(BR, flush=True)

    return tokenizer, dataset
