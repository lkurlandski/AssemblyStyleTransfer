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


def get_dataset(
    files: list[Path],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    dat_path: Path,
    num_proc: int = 1,
    use_saved: bool = True,
    overwrite: bool = False,
) -> DatasetDict:
    """
    If dat_path is given, will retrieve the dataset from there if it exists.
    If dat_path is given, will save the dataset there if it does not exist.
    If dat_path is not given, will prepare the dataset from scratch and not save.
    """

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

    if use_saved and overwrite:
        raise ValueError()

    dat_path = Path(dat_path)
    dat_path.mkdir(parents=True, exist_ok=True)
    raw_path = dat_path / "raw"
    split_path = dat_path / "split"
    tokenized_path = dat_path / "tokenized"
    grouped_path = dat_path / "grouped"

    if use_saved and raw_path.exists():
        raw_dataset = Dataset.load_from_disk(raw_path.as_posix())
    else:
        raw_dataset = get_raw_assembly_dataset(
            files,
            min_lines=4,
            num_proc=num_proc,
        )
    if overwrite or not raw_path.exists():
        shutil.rmtree(raw_path, ignore_errors=True)
        raw_dataset.save_to_disk(raw_path.as_posix())
    print(f"{raw_dataset=}")
    print(f"{raw_dataset[0]}")
    print(BR, flush=True)

    if use_saved and split_path.exists():
        split_dataset = Dataset.load_from_disk(split_path.as_posix())
    else:
        split_dataset = raw_dataset.map(
            split_instructions_fn,
            batched=False,
            remove_columns="text",
        )
    if overwrite or not split_path.exists():
        shutil.rmtree(split_path, ignore_errors=True)
        split_dataset.save_to_disk(split_path.as_posix())
    print(f"{split_dataset=}")
    print(f"{split_dataset[0]}")
    print(BR, flush=True)

    if use_saved and tokenized_path.exists():
        tokenized_dataset = Dataset.load_from_disk(tokenized_path.as_posix())
    else:
        tokenized_dataset = split_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns="text",
        )
    if overwrite or not tokenized_path.exists():
        shutil.rmtree(tokenized_path, ignore_errors=True)
        tokenized_dataset.save_to_disk(tokenized_path.as_posix())
    print(f"{tokenized_dataset=}")
    print(f"{tokenized_dataset[0]}")
    print(BR, flush=True)

    if use_saved and grouped_path.exists():
        grouped_dataset = Dataset.load_from_disk(grouped_path.as_posix())
    else:
        grouped_dataset = tokenized_dataset.map(group_fn, batched=True)
    if overwrite or not grouped_path.exists():
        shutil.rmtree(grouped_path, ignore_errors=True)
        grouped_dataset.save_to_disk(grouped_path.as_posix())
    print(f"{grouped_dataset=}")
    print(f"{grouped_dataset[0]}")
    print(f"{len(grouped_dataset[0]['input_ids'])=}")
    print(BR, flush=True)

    return grouped_dataset


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

    dataset = get_dataset(
        files[: dataset_args.dat_n_files],
        tokenizer,
        tokenizer_args.max_length,
        dataset_args.dat_path,
        dataset_args.num_proc,
        dataset_args.dat_use_saved,
        dataset_args.dat_overwrite,
    )
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
    if len(sys.argv) > 1 and sys.arv[1] == "--debug":
        debug()
    else:
        main()
    print(f"{BR}\nFINISH @{datetime.now()}\n{BR}", flush=True)
