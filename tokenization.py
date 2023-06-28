"""
Process data for learning, including tokenization and data processing (CPU-intensive).
"""

from argparse import ArgumentParser
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from itertools import cycle, islice, tee, zip_longest
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


def get_raw_assembly_dataset(files: Collection[Path]) -> Dataset:
    def gen():
        for f in files:
            snippet = f.read_text()
            yield {"text": snippet}

    dataset = Dataset.from_generator(gen)
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
    ]
    norms = [normalizers.Replace(tokenizers.Regex(rx), rp) for rx, rp in rxs_and_rps]
    return normalizers.Sequence(norms)


def get_normalizer() -> normalizers.Sequence:
    rxs_and_rps = [
        (r"(?<=\s)\d+(?=\s)", cfg.NUM),
        (r"asm\.\S*", cfg.ASM),
        (r"vtable\.\S*", cfg.VTABLE),
        (r"switch\.\S*", cfg.SWITCH),
        (r"str\.\S*", cfg.STR),
        (r"sym\.\S*", cfg.SYM),
        (r"fcn\.\S*", cfg.FCN),
        (r"sub\.\S*", cfg.SUB),
        (r"case\.\S*", cfg.CASE),
        (r"var_\S*", cfg.VAR),
        (r"arg_\S*", cfg.ARG),
        (r"\b0x\w+\b", cfg.ADR),
    ]
    norms = [normalizers.Replace(tokenizers.Regex(rx), rp) for rx, rp in rxs_and_rps]
    return normalizers.Sequence(norms)


def get_pre_tokenizer() -> pre_tokenizers.Sequence:
    # pre_tokenizers.Whitespace splits on "<", ">"
    isolated = [",", "]", "[", "*", "/", "+", "-"] + cfg.SPECIALS
    removed = [" ", "\t", "\n"]
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
        return trainers.WordLevelTrainer(vocab_size)  # for some reason, adding special_tokens here fucks everything up
    elif isinstance(model, models.WordPiece):
        return trainers.WordPieceTrainer(vocab_size, special_tokens=cfg.SPECIALS)
    elif isinstance(model, models.BPE):
        return trainers.BpeTrainer(vocab_size, special_tokens=cfg.SPECIALS)
    elif isinstance(model, models.Unigram):
        return trainers.UnigramTrainer(vocab_size, unk_token=cfg.UNK, special_tokens=cfg.SPECIALS)
    raise ValueError()


def get_post_processor() -> processors.TemplateProcessing:
    return processors.TemplateProcessing(
        single=f"{cfg.BOS} $0 {cfg.EOS}",
        pair=f"{cfg.BOS} $A {cfg.EOS} {cfg.BOS} $B:1 {cfg.EOS}:1",
        special_tokens=[(t, i) for i, t in enumerate(cfg.SPECIALS)],
    )


def get_added_special_token(s: str) -> tokenizers.AddedToken:
    return tokenizers.AddedToken(s, lstrip=True, rstrip=True, normalized=False)


def batch_iterator(dataset: Dataset, batch_size: int = 512) -> tp.Generator[str, None, None]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


def get_tokenizer_file(tokenizers_path: Path, model: Path) -> Path:
    return (Path(tokenizers_path) / model).with_suffix(".json")


def get_tokenizer(tokenizers_path: Path, model: Path, files: Collection[Path] = None) -> Tokenizer:
    path = get_tokenizer_file(tokenizers_path, model)
    print(f"Tokenizer file: {path.as_posix()}", flush=True)
    if path.exists():
        print(f"Found cached tokenizer.", flush=True)
        return Tokenizer.from_file(path.as_posix())

    print(f"Training new tokenizer from {len(files)} files ({round(mem(files), 1)}G).")
    dataset = get_raw_assembly_dataset(files)
    model = get_model(model)
    trainer = get_trainer(model)
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = get_normalizer()
    tokenizer.pre_tokenizer = get_pre_tokenizer()
    tokenizer.post_processor = get_post_processor()
    tokenizer.add_special_tokens([get_added_special_token(s) for s in cfg.SPECIALS])
    tokenizer.train_from_iterator(batch_iterator(dataset), trainer, length=len(dataset))
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
        additional_special_tokens=[tokenizers.AddedToken(s) for s in (cfg.ADR, cfg.STR)],
        **kwargs,
    )


def main(model: str) -> None:
    om = OutputManager()
    tokenizers_path = om.tokenizers
    files = list(om.pre_normalized.rglob("*.asm"))
    tokenizer = get_tokenizer(tokenizers_path, model, files)


def debug() -> None:
    main("WordLevel")


def cli() -> None:
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=str, choices=["WordLevel", "WordPiece", "BPE", "Unigram"])
    args = parser.parse_args()
    if args.debug:
        debug()
    else:
        main(args.model)


if __name__ == "__main__":
    cli()
