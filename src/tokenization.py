"""
Process data for learning, including tokenization and data processing (CPU-intensive).
"""

from argparse import ArgumentParser
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
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

from src import cfg
from src.output_manager import OutputManager
from src.utils import disk_usage


MIN_LINES = 4


def remove_first_line(text):
    return re.sub(r'^.*?\n', '', text, count=1)


def get_raw_assembly_dataset(
    files: Collection[Path],
    num_proc: int = 1,
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
        (r"asm\.\S+", cfg.ASM),
        (r"int\.\S+", cfg.INT),
        (r"loc\.\S+", cfg.LOC),
        (r"vtable\.\S+", cfg.VTABLE),
        (r"switch\.\S+", cfg.SWITCH),
        (r"section\.\S+", cfg.SECTION),
        (r"str\.\S+", cfg.STR),
        (r"sym\.\S+", cfg.SYM),
        (r"fcn\.\S+", cfg.FCN),
        (r"sub\.\S+", cfg.SUB),
        (r"case\.\S+", cfg.CASE),
        (r"reloc\.\S+", cfg.RELOC),
        (r"var_\S+", cfg.VAR),
        (r"arg_\S+", cfg.ARG),
        (r"ARG_\S+", cfg.ARG),
        (r"std::\S+", cfg.STD),
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


def get_model(algorithm: str) -> Tokenizer:
    if algorithm == "WordLevel":
        return models.WordLevel(unk_token=cfg.UNK)
    elif algorithm == "WordPiece":
        return models.WordPiece()
    elif algorithm == "BPE":
        return models.BPE()
    elif algorithm == "Unigram":
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


def get_tokenizer_file(tokenizers_path: Path, algorithm: str) -> Path:
    return (tokenizers_path / algorithm).with_suffix(".json")


def get_tokenizer(
    tokenizers_path: Path,
    algorithm: str,
    files: Collection[Path] = None,
    batch_size: int = 512,
    vocab_size: int = None,
    use_cache: bool = True,
    overwrite: bool = False,
) -> Tokenizer:
    path = get_tokenizer_file(tokenizers_path, algorithm)
    if path.exists() and use_cache:
        print("Retrieving cached_tokenizer.", flush=True)
        return Tokenizer.from_file(path.as_posix())
    
    print("Training a new tokenzer.", flush=True)
    if path.exists() and overwrite:
        print(f"Will overwrite the existing tokenizer.", flush=True)
    elif path.exists() and not overwrite:
        print(f"Will not overwrite the existing tokenizer.", flush=True)

    dataset = get_raw_assembly_dataset(files, min_lines=MIN_LINES)
    model = get_model(algorithm)
    trainer = get_trainer(model, vocab_size)
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = get_normalizer()
    tokenizer.pre_tokenizer = get_pre_tokenizer()
    tokenizer.add_special_tokens(cfg.SPECIALS)
    tokenizer.add_tokens([get_added_token(s) for s in cfg.NONSPECIALS])
    tokenizer.train_from_iterator(
        batch_iterator(dataset, batch_size),
        trainer,
        length=len(dataset) // batch_size,
    )

    if not path.exists() or overwrite:
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
