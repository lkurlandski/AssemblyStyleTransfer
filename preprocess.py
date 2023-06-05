"""
Process data for learning, including tokenization and data processing (CPU-intensive).
"""

from argparse import ArgumentParser
from collections.abc import Collection
from dataclasses import dataclass
from itertools import cycle, islice, tee, zip_longest
from pathlib import Path
import re
import typing as tp

from datasets import Dataset, DatasetDict
import tokenizers
from tokenizers import models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PretrainedConfig

import cfg
import prepare


def get_tokenizer_file(tokenizers_path: Path, model: Path) -> Path:
    if model not in ("WordLevel", "WordPiece", "BPE", "Unigram"):
        raise ValueError()
    return (Path(tokenizers_path) / model).with_suffix(".json")


def remove_word_before_tab(string: str | list[str]):
    def _remove_word_before_tab(s: str) -> str:
        return re.sub(r"\w+\t", "", s)

    if isinstance(string, str):
        return _remove_word_before_tab(string)
    return [_remove_word_before_tab(s) for s in string]


def batch_iterator_seq2seq(dataset: Dataset, batch_size: int) -> tp.Generator[str, None, None]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["mal"]
        yield dataset[i : i + batch_size]["ben"]


def batch_iterator_pretraining(dataset: Dataset, batch_size: int) -> tp.Generator[str, None, None]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


def get_tokenizer(
    path: Path = None,
    model: tp.Literal["WordLevel", "WordPiece", "BPE", "Unigram"] = "WordLevel",
    dataset: Dataset = None,
    cache: bool = True,
) -> Tokenizer:
    post_processor = processors.TemplateProcessing(
        single=f"{cfg.BOS} $0 {cfg.EOS}",
        pair=f"{cfg.BOS} $A {cfg.EOS} {cfg.BOS} $B:1 {cfg.EOS}:1",
        special_tokens=[(t, i) for i, t in enumerate(cfg.SPECIALS)],
    )

    if path is not None and path.exists() and cache:
        print(f"Using tokenizer from {path.as_posix()}")
        tokenizer = Tokenizer.from_file(path.as_posix())
        tokenizer.post_processor = post_processor
        return tokenizer

    if model == "WordLevel":
        model = models.WordLevel(unk_token=cfg.UNK)
        trainer = trainers.WordLevelTrainer(special_tokens=cfg.SPECIALS)
    elif model == "WordPiece":
        model = models.WordPiece(unk_token=cfg.UNK)
        trainer = trainers.WordPieceTrainer(special_tokens=cfg.SPECIALS)
    elif model == "BPE":
        model = models.BPE(unk_token=cfg.UNK)
        trainer = trainers.BpeTrainer(special_tokens=cfg.SPECIALS)
    elif model == "Unigram":
        model = models.Unigram()
        trainer = trainers.UnigramTrainer(unk_token=cfg.UNK, special_tokens=cfg.SPECIALS)
    else:
        raise ValueError(f"Unknown model: {model}")

    normalizer = normalizers.Sequence(
        [
            normalizers.Replace(tokenizers.Regex(r"\w+\t"), ""),  # Remove address before tab
            normalizers.Replace(tokenizers.Regex(r"\b0x\w+\b"), cfg.ADR),  # Replace addresses
        ]
    )
    pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.CharDelimiterSplit("\n"),
            pre_tokenizers.Punctuation("isolated"),
            pre_tokenizers.Split(tokenizers.Regex(r"\[|\]"), "isolated"),
            pre_tokenizers.Split(cfg.ADR, "isolated"),
        ]
    )
    tokenizer = Tokenizer(model)

    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = post_processor

    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator_pretraining(dataset, 1024), trainer, length=len(dataset))

    if path is not None:
        print(f"Saving tokenizer to {path.as_posix()}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(path.as_posix())

    return tokenizer


def get_pretrained_tokenizer(tokenizer: Tokenizer, **kwargs) -> PreTrainedTokenizerFast:
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


def get_pretrained_tokenizer_specific(
    tokenizer_file: Path,
    config: PretrainedConfig,
) -> PreTrainedTokenizerFast | PreTrainedTokenizer:
    if isinstance(config, transformers.BertConfig):
        return transformers.BertTokenizerFast(
            tokenizer_file=tokenizer_file,
            do_lower_case=False,
            unk_token=cfg.UNK,
            sep_token=cfg.SEP,
            pad_token=cfg.PAD,
            cls_token=cfg.CLS,
            mask_token=cfg.MSK,
            tokenize_chinese_chars=False,
            strip_accents=False,
            wordpieces_prefix="##",  # TODO
        )
    raise NotImplementedError()


def get_pretraining_dataset(files: Collection[Path]) -> Dataset:
    def gen():
        for f in files:
            snippet = f.read_text()
            yield {"text": snippet}

    dataset = Dataset.from_generator(gen)
    return dataset


def get_processed_pretraining_dataset(
    dataset: Dataset = None,
    tokenizer: PreTrainedTokenizerFast = None,
    load_from_cache_file: bool = True,
    n_instructions: int = 48,
    **kwargs,
) -> Dataset:
    def fn(batch):
        chunks = []
        for snippet in batch["text"]:
            instructions = snippet.split("\n")
            chunks += [
                "\n".join(instructions[i : i + n_instructions]) for i in range(0, len(instructions), n_instructions)
            ]
        return tokenizer(chunks, truncation=True, padding="longest", **kwargs)

    tokenized_dataset = dataset.map(
        fn, batched=True, remove_columns=dataset.column_names, load_from_cache_file=load_from_cache_file
    )
    return tokenized_dataset


def get_seq2seq_dataset(
    mal_files: Collection[Path],
    ben_files: Collection[Path],
    mode: tp.Literal["min", "repeat", "empty"],
) -> Dataset:
    FILL = ""

    def gen() -> tp.Generator[dict[str, str], None, None]:
        if mode == "min":
            itr = zip(mal_files, ben_files)
        elif mode == "repeat":
            mal_itrs, ben_itrs = tee(mal_files, 2), tee(ben_files, 2)
            l_mal = sum(1 for _ in mal_itrs[0])
            l_ben = sum(1 for _ in ben_itrs[0])
            _mal_files = islice(cycle(mal_itrs[1]), 0, max(l_mal, l_ben))
            _ben_files = islice(cycle(ben_itrs[1]), 0, max(l_mal, l_ben))
            itr = zip(_mal_files, _ben_files, strict=True)
        elif mode == "empty":
            itr = zip_longest(mal_files, ben_files, fillvalue=None)

        for m, b in itr:
            m_snippet = m.read_text() if m is not None else FILL
            b_snippet = b.read_text() if b is not None else FILL
            yield {"mal": m_snippet, "ben": b_snippet}

    dataset = Dataset.from_generator(gen)
    return dataset


def get_processed_seq2seq_dataset(
    dataset: Dataset = None,
    tokenizer: PreTrainedTokenizerFast = None,
    load_from_cache_file: bool = True,
    **kwargs,
) -> Dataset:
    """
    Suggested kwargs include: `max_length`, `padding`, `truncation`
    """

    def fn(batch):
        return tokenizer(text=batch["mal"], text_target=batch["ben"], **kwargs)

    tokenized_dataset = dataset.map(fn, batched=True, load_from_cache_file=load_from_cache_file)
    return tokenized_dataset


@dataclass
class PathArgs:
    root: Path
    disassemble: Path = prepare.PathArgs.disassemble
    tokenizers: Path = Path("tokenizers")

    def __post_init__(self) -> None:
        self.disassemble = self.root / self.disassemble
        self.tokenizers = self.root / self.tokenizers


@dataclass
class TokenizerArgs:
    model: str
    model_max_length: int = 256
    use_cache: bool = True
    clear_cache: bool = False


@dataclass
class DatasetArgs:
    test_size: int = 0.1
    use_cache: bool = True
    clear_cache: bool = False
    subsample: tp.Optional[int] = None


def main(
    paths: PathArgs,
    token_args: TokenizerArgs,
    data_args: TokenizerArgs,
) -> tuple[PreTrainedTokenizerFast, DatasetDict]:
    files = [p for p in paths.disassemble.iterdir() if p.suffix == ".asm"]
    mem = sum(f.stat().st_size for f in files) * 10e-9
    print(f"Found {round(mem, 1)}G of files in {paths.disassemble.as_posix()}", flush=True)
    print("Getting pretraining dataset...", flush=True)
    dataset = get_pretraining_dataset(files)
    if data_args.clear_cache:
        dataset.cleanup_cache_files()
    if data_args.subsample is not None:
        dataset = dataset.select(range(data_args.subsample))
    print(f"{dataset=}", flush=True)

    print("Getting tokenizer...", flush=True)
    paths.tokenizers.mkdir(exist_ok=True)
    tokenizer_file = get_tokenizer_file(paths.tokenizers, token_args.model)
    tokenizer = get_tokenizer(tokenizer_file, token_args.model, dataset, token_args.use_cache)
    if token_args.clear_cache:
        tokenizer_file.unlink()
    print(f"{tokenizer=}", flush=True)
    fast_tokenizer = get_pretrained_tokenizer(tokenizer, model_max_length=token_args.model_max_length)
    print(f"{fast_tokenizer=}", flush=True)

    print("Tokenizing pretraining dataset...", flush=True)
    tokenized_dataset = get_processed_pretraining_dataset(dataset, fast_tokenizer, data_args.use_cache)
    if data_args.clear_cache:
        tokenized_dataset.cleanup_cache_files()
    print(f"{tokenized_dataset=}", flush=True)
    split_tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=data_args.test_size, load_from_cache_file=data_args.use_cache
    )
    if data_args.clear_cache:
        split_tokenized_dataset.cleanup_cache_files()
    print(f"{split_tokenized_dataset=}", flush=True)

    return fast_tokenizer, split_tokenized_dataset


if __name__ == "__main__":
    parser = ArgumentParser(description="Your program description.")
    parser.add_argument("--root", type=Path, help="Path")
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_max_length", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--clear_cache", action="store_true")
    parser.add_argument("--subsample", type=int)
    args = parser.parse_args()

    main(
        PathArgs(args.root),
        TokenizerArgs(
            args.model,
            args.model_max_length,
            not args.no_cache,
            args.clear_cache,
        ),
        DatasetArgs(
            args.test_size,
            not args.no_cache,
            args.clear_cache,
            args.subsample,
        ),
    )
