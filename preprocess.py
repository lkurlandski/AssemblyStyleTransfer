"""
Pepare data for learning.
"""

from collections.abc import Collection
from itertools import cycle, islice, tee, zip_longest
from pathlib import Path
import re
import typing as tp

import datasets
import tokenizers
from tokenizers import models, normalizers, pre_tokenizers, processors, trainers
import transformers

UNK = "<UNK>"
MSK = "<MSK>"
PAD = "<PAD>"
SEP = "<SEP>"
CLS = "<CLS>"
BOS = "<BOS>"
EOS = "<EOS>"
ADR = "<ADR>"
STR = "<STR>"
SPECIALS = [UNK, MSK, PAD, SEP, CLS, BOS, EOS, ADR, STR]

FILL = ""


def remove_word_before_tab(string: str | list[str]):
    def _remove_word_before_tab(s: str) -> str:
        return re.sub(r"\w+\t", "", s)

    if isinstance(string, str):
        return _remove_word_before_tab(string)
    return [_remove_word_before_tab(s) for s in string]


def batch_iterator_seq2seq(dataset: datasets.Dataset, batch_size: int) -> tp.Generator[str, None, None]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["mal"]
        yield dataset[i : i + batch_size]["ben"]


def batch_iterator_pretraining(dataset: datasets.Dataset, batch_size: int) -> tp.Generator[str, None, None]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


def get_tokenizer(
    path: Path = None,
    model: tp.Literal["WordLevel", "WordPiece", "BPE", "Unigram"] = "WordLevel",
    dataset: datasets.Dataset = None,
    cache: bool = True,
) -> tokenizers.Tokenizer:

    post_processor = processors.TemplateProcessing(
        single=f"{BOS} $0 {EOS}",
        pair=f"{BOS} $A {EOS} {BOS} $B:1 {EOS}:1",
        special_tokens=[(t, i) for i, t in enumerate(SPECIALS)],
    )

    if path is not None and path.exists() and cache:
        tokenizer = tokenizers.Tokenizer.from_file(path.as_posix())
        tokenizer.post_processor = post_processor
        return tokenizer

    if model == "WordLevel":
        model = models.WordLevel(unk_token=UNK)
        trainer = trainers.WordLevelTrainer(special_tokens=SPECIALS)
    elif model == "WordPiece":
        model = models.WordPiece(unk_token=UNK)
        trainer = trainers.WordPieceTrainer(special_tokens=SPECIALS)
    elif model == "BPE":
        model = models.BPE(unk_token=UNK)
        trainer = trainers.BpeTrainer(special_tokens=SPECIALS)
    elif model == "Unigram":
        model = models.Unigram()
        trainer = trainers.UnigramTrainer(unk_token=UNK, special_tokens=SPECIALS)
    else:
        raise ValueError(f"Unknown model: {model}")

    normalizer = normalizers.Sequence(
        [
            normalizers.Replace(tokenizers.Regex(r"\w+\t"), ""),  # Remove address before tab
            normalizers.Replace(tokenizers.Regex(r"\b0x\w+\b"), ADR),  # Replace addresses
        ]
    )
    pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.CharDelimiterSplit("\n"),
            pre_tokenizers.Punctuation("isolated"),
            pre_tokenizers.Split(tokenizers.Regex(r"\[|\]"), "isolated"),
            pre_tokenizers.Split(ADR, "isolated"),
        ]
    )
    tokenizer = tokenizers.Tokenizer(model)

    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = post_processor

    tokenizer.train_from_iterator(batch_iterator_pretraining(dataset, 1024), trainer, length=len(dataset))

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(path.as_posix())

    return tokenizer


def get_pretrained_tokenizer(tokenizer: tokenizers.Tokenizer, **kwargs) -> transformers.PreTrainedTokenizerFast:
    """
    Suggested kwargs include: `model_max_length`
    """
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=tokenizers.AddedToken(UNK),
        mask_token=tokenizers.AddedToken(MSK),
        pad_token=tokenizers.AddedToken(PAD),
        sep_token=tokenizers.AddedToken(SEP),
        cls_token=tokenizers.AddedToken(CLS),
        bos_token=tokenizers.AddedToken(BOS),
        eos_token=tokenizers.AddedToken(EOS),
        additional_special_tokens=[tokenizers.AddedToken(s) for s in (ADR, STR)],
        **kwargs,
    )


def get_pretraining_dataset(files: Collection[Path]) -> datasets.Dataset:
    def gen():
        for f in files:
            snippet = f.read_text()
            yield {"text": snippet}

    dataset = datasets.Dataset.from_generator(gen)
    return dataset


def get_processed_pretraining_dataset(
    dataset: datasets.Dataset = None,
    tokenizer: transformers.PreTrainedTokenizerFast = None,
    load_from_cache_file: bool = True,
    n_instructions: int = 48,
    **kwargs,
) -> datasets.Dataset:
    def fn(batch):
        chunks = []
        for snippet in batch["text"]:
            instructions = snippet.split("\n")
            chunks += [
                "\n".join(instructions[i : i + n_instructions]) for i in range(0, len(instructions), n_instructions)
            ]
        return tokenizer(chunks, **kwargs)

    tokenized_dataset = dataset.map(
        fn, batched=True, remove_columns=dataset.column_names, load_from_cache_file=load_from_cache_file
    )
    return tokenized_dataset


def get_seq2seq_dataset(
    mal_files: Collection[Path],
    ben_files: Collection[Path],
    mode: tp.Literal["min", "repeat", "empty"],
) -> datasets.Dataset:
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

    dataset = datasets.Dataset.from_generator(gen)
    return dataset


def get_processed_seq2seq_dataset(
    dataset: datasets.Dataset = None,
    tokenizer: transformers.PreTrainedTokenizerFast = None,
    load_from_cache_file: bool = True,
    **kwargs,
) -> datasets.Dataset:
    """
    Suggested kwargs include: `max_length`, `padding`, `truncation`
    """

    def fn(batch):
        return tokenizer(text=batch["mal"], text_target=batch["ben"], **kwargs)

    tokenized_dataset = dataset.map(fn, batched=True, load_from_cache_file=load_from_cache_file)
    return tokenized_dataset
