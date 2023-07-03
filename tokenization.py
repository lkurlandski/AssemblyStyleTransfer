"""
Process data for learning, including tokenization and data processing (CPU-intensive).
"""

from argparse import ArgumentParser
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from itertools import cycle, islice, tee, zip_longest
import os
from pathlib import Path
import re
import sys
import typing as tp

from datasets import Dataset, DatasetDict
import tokenizers
from tokenizers import models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PretrainedConfig

import cfg
from utils import mem, OutputManager


MIN_LINES = 4


def remove_first_line(text):
    return re.sub(r'^.*?\n', '', text, count=1)


def get_raw_assembly_dataset(
    files: Collection[Path],
    num_proc: int = cfg.N_WORKERS,
    min_lines: int = float("-inf"),
    max_lines: int = float("inf"),
) -> Dataset:
    def gen():
        for f in files:
            functions = f.read_text().split('-' * 42)
            for func in functions:
                func = remove_first_line(func.lstrip())  # trim location
                if min_lines <= func.count("\n") <= max_lines:
                    yield {"text": func}

    dataset = Dataset.from_generator(gen, num_proc=num_proc)
    return dataset


def get_pre_normalizer() -> normalizers.Sequence:
    """Normalizer for functions produced by the radare2 PDR command."""
    rxs_and_rps = [
        (r"^┌.*\n", ""),
        (r"^├.*\n", ""),
        (r"└", "│"),
        (r"^\|.*(?:\n|$)", ""),
        (r";.*", ""),
        (r"│ ", ""),
        (r"\n{2,}", "\n"),
        (r"^.{31}", ""),
        ("\n\n", "\n"),
    ]
    norms = [normalizers.Replace(tokenizers.Regex(rx), rp) for rx, rp in rxs_and_rps]
    return normalizers.Sequence(norms)


def get_normalizer() -> normalizers.Sequence:
    rxs_and_rps = [
        ("\n", cfg.INS),
        (r"(?<=\s)\d+(?=\s)", cfg.NUM),
        (r"asm\.\S*", cfg.ASM),
        (r"int\.\S*", cfg.INT),
        (r"vtable\.\S*", cfg.VTABLE),
        (r"switch\.\S*", cfg.SWITCH),
        (r"str\.\S*", cfg.STR),
        (r"sym\.\S*", cfg.SYM),
        (r"fcn\.\S*", cfg.FCN),
        (r"sub\.\S*", cfg.SUB),
        (r"case\.\S*", cfg.CASE),
        (r"var_\S*", cfg.VAR),
        (r"arg_\S*", cfg.ARG),
        (r"ARG_\S*", cfg.ARG),
        (r"\b0x\w+\b", cfg.ADR),
        (r"^[a-f0-9]+$", cfg.INVALID),
        (r"^invalid$", cfg.INVALID),
    ]
    norms = [normalizers.Replace(tokenizers.Regex(rx), rp) for rx, rp in rxs_and_rps]
    return normalizers.Sequence(norms)


def get_pre_tokenizer() -> pre_tokenizers.Sequence:
    # pre_tokenizers.Whitespace splits on "<", ">"
    isolated = [",", "]", "[", "*", "/", "+", "-"] + cfg.SPECIALS + cfg.NONSPECIALS
    removed = [" ", "\t"]
    return pre_tokenizers.Sequence(
        [pre_tokenizers.Split(s, "removed") for s in removed] + [pre_tokenizers.Split(s, "isolated") for s in isolated]
    )


def get_model(model: str) -> Tokenizer:
    if model == "WordLevel":
        return models.WordLevel(unk_token=cfg.UNK)
    elif model == "WordPiece":
        return models.WordPiece(unk_token=cfg.UNK)
    elif model == "BPE":
        return models.BPE(unk_token=cfg.UNK)
    elif model == "Unigram":
        return models.Unigram()
    raise ValueError()


def get_trainer(model: models.Model, vocab_size: int = None) -> trainers.Trainer:
    if isinstance(model, models.WordLevel):
        return trainers.WordLevelTrainer(special_tokens=cfg.SPECIALS)
    elif isinstance(model, models.WordPiece):
        return trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=cfg.SPECIALS)
    elif isinstance(model, models.BPE):
        return trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=cfg.SPECIALS)
    elif isinstance(model, models.Unigram):
        return trainers.UnigramTrainer(vocab_size=vocab_size, unk_token=cfg.UNK, special_tokens=cfg.SPECIALS)
    raise ValueError()


def get_post_processor() -> processors.TemplateProcessing:
    return processors.TemplateProcessing(
        single=f"{cfg.BOS} $0 {cfg.EOS}",
        pair=f"{cfg.BOS} $A {cfg.EOS} {cfg.BOS} $B:1 {cfg.EOS}:1",
        special_tokens=[(t, i) for i, t in enumerate(cfg.SPECIALS)],
    )


def get_added_token(s: str) -> tokenizers.AddedToken:
    return tokenizers.AddedToken(s, lstrip=True, rstrip=True, normalized=False)


def batch_iterator(dataset: Dataset, batch_size: int = 512) -> tp.Generator[str, None, None]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


def get_tokenizer_file(tokenizers_path: Path, model: Path) -> Path:
    return (Path(tokenizers_path) / model).with_suffix(".json")


def get_tokenizer(
    tokenizers_path: Path,
    model: Path,
    files: Collection[Path] = None,
    batch_size: int = 512,
    vocab_size: int = None,
    use_cache: bool = True,
    overwrite: bool = False,
) -> Tokenizer:
    path = get_tokenizer_file(tokenizers_path, model)
    print(f"Tokenizer file: {path.as_posix()}", flush=True)
    if path.exists() and use_cache:
        print(f"Found cached tokenizer.", flush=True)
        return Tokenizer.from_file(path.as_posix())
    elif path.exists() and not overwrite:
        print(f"Will train a new tokenizer, but not overwrite the existing tokenizer.", flush=True)
    elif path.exists() and overwrite:
        print(f"Will train a new tokenizer and overwrite the existing tokenizer.", flush=True)

    print(f"Found {len(files)} files ({round(mem(files), 1)}G).")
    dataset = get_raw_assembly_dataset(files, min_lines=MIN_LINES)
    print(f"Built a dataset with {dataset.num_rows} examples.")
    model = get_model(model)
    trainer = get_trainer(model, vocab_size)
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = get_normalizer()
    tokenizer.pre_tokenizer = get_pre_tokenizer()
    tokenizer.post_processor = get_post_processor()
    tokenizer.add_special_tokens(cfg.SPECIALS)
    tokenizer.add_tokens([get_added_token(s) for s in cfg.NONSPECIALS])
    print(f"Training new tokenizer with {batch_size=} and {len(os.sched_getaffinity(0))} logical cores.")
    tokenizer.train_from_iterator(batch_iterator(dataset, batch_size), trainer, length=len(dataset))
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path.as_posix())
    return tokenizer


def get_fast_tokenizer(tokenizer: Tokenizer, **kwargs) -> PreTrainedTokenizerFast:
    """
    Suggested kwargs include: `model_max_length`
    """
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=tokenizers.AddedToken(cfg.UNK),
        mask_token=tokenizers.AddedToken(cfg.MSK),
        pad_token=tokenizers.AddedToken(cfg.PAD),
        sep_token=tokenizers.AddedToken(cfg.SEP),
        cls_token=tokenizers.AddedToken(cfg.CLS),
        bos_token=tokenizers.AddedToken(cfg.BOS),
        eos_token=tokenizers.AddedToken(cfg.EOS),
        additional_special_tokens=cfg.SPECIALS,
        **kwargs,
    )


def main(model: str, batch_size: int = 512, vocab_size: int = None, overwrite: bool = False, subset: int = None, **kwargs) -> None:
    om = OutputManager()
    files = list(om.merged.iterdir())[:subset]
    tokenizer = get_tokenizer(om.tokenizers, model, files, batch_size, vocab_size, overwrite)
    fast_tokenizer = get_fast_tokenizer(tokenizer=tokenizer, **kwargs)
    return fast_tokenizer


def debug() -> None:
    main("WordLevel", overwrite=True)


def cli() -> None:
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=str, choices=["WordLevel", "WordPiece", "BPE", "Unigram"], default="WordLevel")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--vocab_size", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.debug:
        debug()
    else:
        main(args.model, args.batch_size, int(args.vocab_size) if args.vocab_size else None, args.overwrite, args.subset)


if __name__ == "__main__":
    cli()
