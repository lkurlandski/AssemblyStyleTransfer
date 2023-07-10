"""
Utilities for pretraining scripts.

"5% < 26; 10% < 37; 15% < 46; 20% < 57; 25% < 68; 30% < 79; 35% < 92; "
"40% < 107; 45% < 125; 50% < 144; 55% < 165; 60% < 192; 65% < 221; "
"70% < 255; 75% < 303; 80% < 367; 85% < 471; 90% < 647; 95% < 1036"
"""

from datetime import datetime
from itertools import chain
import os
from pathlib import Path
from pprint import pformat, pprint
import shutil
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets import Dataset, DatasetDict
from transformers import HfArgumentParser, PreTrainedTokenizerBase

from src.cfg import *
from src.output_manager import OutputManager
from src.pretrain.arg_classes import TokenizerArguments, DatasetArguments
from src.tokenization import (
    get_tokenizer,
    get_fast_tokenizer,
    get_raw_assembly_dataset,
)
from src.utils import disk_usage


class ProcessDatasetPipeline:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        dat_path: Path,
        num_proc: int = 1,
        use_saved: bool = True,
        overwrite: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dat_path = Path(dat_path)
        self.num_proc = num_proc
        self.use_saved = use_saved
        self.overwrite = overwrite
        self.raw_path = self.dat_path / "raw"
        self.split_path = self.dat_path / "split"
        self.tokenized_path = self.dat_path / "tokenized"
        self.grouped_path = self.dat_path / "grouped"

    def __call__(self, files: list[Path] = None) -> None:
        self.dat_path.mkdir(parents=True, exist_ok=True)

        if not self.use_saved or self.overwrite:
            dataset = self.raw(files)
            dataset = self.split(dataset)
            dataset = self.tokenize(dataset)
            dataset = self.group(dataset)
            return dataset

        if self.grouped_path.exists():
            return self.group()

        dataset = None
        sequence = [self.group]
        if not self.grouped_path.exists():
            sequence.insert(0, self.tokenize)
        if not self.tokenized_path.exists():
            sequence.insert(0, self.split)
        if not self.split_path.exists():
            dataset = self.raw(files)

        for func in sequence:
            dataset = func(dataset)
        return dataset

    def raw(self, files: list[Path] = None) -> Dataset:
        if self.use_saved and self.raw_path.exists():
            print(f"Retrieving raw dataset from {self.raw_path.as_posix()}")
            raw_dataset = Dataset.load_from_disk(self.raw_path.as_posix())
        else:
            print(f"Constructing raw dataset from {files[0].parent.as_posix()}")
            raw_dataset = get_raw_assembly_dataset(files, min_lines=4)
        if self.overwrite or not self.raw_path.exists():
            shutil.rmtree(self.raw_path, ignore_errors=True)
            raw_dataset.save_to_disk(self.raw_path.as_posix())
        print(f"{raw_dataset=}")
        print(f"{raw_dataset[0]}")
        print(BR, flush=True)
        return raw_dataset

    def split(self, raw_dataset: Dataset = None) -> Dataset:
        if self.use_saved and self.split_path.exists():
            print(f"Retrieving split dataset from {self.split_path.as_posix()}", flush=True)
            split_dataset = Dataset.load_from_disk(self.split_path.as_posix())
        else:
            print("Constructing split dataset.", flush=True)
            split_dataset = raw_dataset.map(
                self._split(),
                batched=False,
                remove_columns="text",
                num_proc=self.num_proc,
            )
        if self.overwrite or not self.split_path.exists():
            shutil.rmtree(self.split_path, ignore_errors=True)
            split_dataset.save_to_disk(self.split_path.as_posix())
        print(f"{split_dataset=}")
        print(f"{split_dataset[0]}")
        print(BR, flush=True)
        return split_dataset

    def tokenize(self, split_dataset: Dataset = None) -> Dataset:
        if self.use_saved and self.tokenized_path.exists():
            print(f"Retrieving tokenized dataset from {self.tokenized_path.as_posix()}", flush=True)
            tokenized_dataset = Dataset.load_from_disk(self.tokenized_path.as_posix())
        else:
            print("Constructing tokenized dataset.", flush=True)
            tokenized_dataset = split_dataset.map(
                self._tokenize(),
                batched=True,
                remove_columns="text",
                num_proc=self.num_proc,
            )
        if self.overwrite or not self.tokenized_path.exists():
            shutil.rmtree(self.tokenized_path, ignore_errors=True)
            tokenized_dataset.save_to_disk(self.tokenized_path.as_posix())
        print(f"{tokenized_dataset=}")
        print(f"{tokenized_dataset[0]}")
        print(BR, flush=True)
        return tokenized_dataset

    def group(self, tokenized_dataset: Dataset = None) -> Dataset:
        if self.use_saved and self.grouped_path.exists():
            print(f"Retrieving grouped dataset from {self.grouped_path.as_posix()}", flush=True)
            grouped_dataset = Dataset.load_from_disk(self.grouped_path.as_posix())
        else:
            print("Constructing grouped dataset.", flush=True)
            grouped_dataset = tokenized_dataset.map(self._group(), batched=True)
        if self.overwrite or not self.grouped_path.exists():
            shutil.rmtree(self.grouped_path, ignore_errors=True)
            grouped_dataset.save_to_disk(self.grouped_path.as_posix())
        print(f"{grouped_dataset=}")
        print(f"{grouped_dataset[0]}")
        print(f"{len(grouped_dataset[0]['input_ids'])=}")
        print(BR, flush=True)
        return grouped_dataset

    def _split(self):
        """Cannot be used with map and batched=True"""

        def fn(example):
            s = example["text"].split(INS)
            s = f"{self.tokenizer.pad_token}".join(s)
            s = self.tokenizer.bos_token + s + self.tokenizer.eos_token
            return {"text": s}

        return fn

    def _tokenize(self):
        def fn(examples):
            return self.tokenizer(
                examples["text"],
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False,
            )

        return fn

    def _group(self):
        """Main data processing function that will concatenate all
        texts from our dataset and generate chunks of max_length."""

        def fn(examples):
            # Concatenate all texts
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # Drop the small remainder
            if total_length >= self.max_length:
                total_length = (total_length // self.max_length) * self.max_length
            # Split by chunks of max_len
            result = {
                k: [t[i : i + self.max_length] for i in range(0, total_length, self.max_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        return fn


def get_tokenizer_and_dataset(
    tokenizer_args: TokenizerArguments,
    dataset_args: DatasetArguments,
) -> tuple[PreTrainedTokenizerBase, DatasetDict]:
    om = OutputManager()

    files = []
    if not tokenizer_args.tok_use_saved or dataset_args.dat_use_saved:
        files = sorted(list(om.merged.glob("*.txt")))
        print(f"{len(files)=}")
        print(f"{round(disk_usage(files), 2)=}")
        print(BR, flush=True)

    tokenizer = get_tokenizer(
        om.tokenizers,
        tokenizer_args.tok_algorithm,
        files[: tokenizer_args.tok_n_files],
        tokenizer_args.tok_batch_size,
        tokenizer_args.vocab_size,
        tokenizer_args.tok_use_saved,
        tokenizer_args.tok_overwrite,
    )
    tokenizer = get_fast_tokenizer(
        tokenizer,
        model_max_length=tokenizer_args.max_length,
        padding_side="right",
    )
    print(f"{tokenizer=}")
    print(BR, flush=True)

    if dataset_args.dat_n_files == 0:
        return tokenizer, None
    if dataset_args.dat_n_files is None:
        dataset_args.dat_n_files = len(files)

    dataset = ProcessDatasetPipeline(
        tokenizer,
        tokenizer_args.max_length,
        dataset_args.dat_path,
        dataset_args.num_proc,
        dataset_args.dat_use_saved,
        dataset_args.dat_overwrite,
    )(files[: dataset_args.dat_n_files])
    if dataset_args.dat_n_examples and dataset_args.dat_n_examples < dataset.num_rows:
        dataset = dataset.select(range(dataset_args.dat_n_examples))
    dataset = dataset.train_test_split(dataset_args.validation_split)
    print(f"{dataset=}")
    print(BR, flush=True)

    return tokenizer, dataset


def main():
    pprint({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
    print(BR, flush=True)

    parser = HfArgumentParser([TokenizerArguments, DatasetArguments])
    args = parser.parse_args_into_dataclasses()
    tokenizer_args = args[0]
    dataset_args = args[1]
    pprint(f"{tokenizer_args=}")
    pprint(f"{dataset_args=}")
    print(BR, flush=True)

    tokenizer, dataset = get_tokenizer_and_dataset(tokenizer_args, dataset_args)
    print(f"{tokenizer=}")
    print(f"{dataset=}")
    print(BR, flush=True)


def debug():
    ...


if __name__ == "__main__":
    print(f"{BR}\nSTART @{datetime.now()}\n{BR}", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        debug()
    else:
        main()
    print(f"{BR}\nFINISH @{datetime.now()}\n{BR}", flush=True)
