"""
Training for seq2seq models.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil
import sys
import typing as tp
from typing import Optional

from torch import tensor

import datasets
from datasets import Dataset, DatasetDict
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# TODO: try to optimize the generation process
# from optimum.bettertransformer import BetterTransformer

import cfg
import preprocess
from utils import (
    get_highest_path,
    OutputManager,
)


PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1


class Direction(Enum):
    FORWARD = "fw"
    BACKWARD = "bw"


def opposite_direction(direction: Direction):
    return Direction.BACKWARD if direction == Direction.FORWARD else Direction.FORWARD


def build_seq2seq_model(
    encoder_pretrained_path: Path,
    decoder_pretrained_path: Path,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> PreTrainedModel:

    # This will cause warnings that can be ignored because:
    # - the encoder was pretrained, so its head is not needed, so its head is removed.
    # - the decoder will have a new head attached to it, which initally contains random weights.
    seq2seq = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_path.as_posix(), decoder_pretrained_path.as_posix()
    )
    seq2seq.config.decoder_start_token_id = bos_token_id
    seq2seq.config.forced_bos_token_id = True
    seq2seq.config.eos_token_id = eos_token_id
    seq2seq.config.forced_eos_token_id = True
    seq2seq.config.pad_token_id = pad_token_id

    return seq2seq


def train_unsupervised(
    fw_seq2seq: PreTrainedModel,
    bw_seq2seq: PreTrainedModel,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    overwrite_output_dir: bool = False,
    num_train_epochs: int = 1,
    generation_config: Optional[GenerationConfig] = None,
) -> PreTrainedModel:
    def get_data_collator(seq2seq: PreTrainedModel) -> DataCollatorForSeq2Seq:
        return DataCollatorForSeq2Seq(
            tokenizer,
            model=seq2seq,
            max_length=tokenizer.model_max_length,
            padding="max_length",
        )

    def get_training_args(
        output_dir: Path, epoch: int, direction: Direction
    ) -> Seq2SeqTrainingArguments:  # pylint: disable=unused-argument
        return Seq2SeqTrainingArguments(
            output_dir=output_dir.as_posix(),
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            save_total_limit=3,
            num_train_epochs=1,
            fp16=True,
            push_to_hub=False,
        )

    def get_trainer(
        seq2seq: PreTrainedModel,
        training_args: Seq2SeqTrainingArguments,
        data_collator: DataCollatorForSeq2Seq,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        epoch: int,
        direction: Direction,
    ) -> Seq2SeqTrainer:  # pylint: disable=unused-argument
        return Seq2SeqTrainer(
            model=seq2seq,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

    def get_directional_dataset(direction: Direction) -> Dataset:
        columns = [c for c in dataset["train"].column_names if c.startswith(direction.value)]
        columns = {c: c.lstrip(f"{direction.value}_") for c in columns}
        return dataset["train"].select_columns(list(columns.keys())).rename_columns(columns)

    def get_synthetic_dataset(generator: PreTrainedModel, direction: Direction) -> Dataset:
        print(f"{generator=}\n{generator.device=}")
        dataset_for_generator = get_directional_dataset(opposite_direction(direction))
        print(f"{dataset_for_generator=}")
        inputs_for_generator = tensor(dataset_for_generator["input_ids"], device=generator.device)
        print(f"{inputs_for_generator=}\n{inputs_for_generator.device=}")
        outputs = generator.generate(inputs_for_generator, generation_config)
        print(f"{outputs=}")
        labels = Dataset.from_dict({"labels": outputs})
        print(f"{labels=}")
        dataset_for_learner = get_directional_dataset(direction).remove_columns("labels")
        print(f"{dataset_for_learner=}")
        dataset_for_learner = datasets.concatenate_datasets([dataset_for_learner, labels], axis=1)
        print(f"{dataset_for_learner=}")
        return dataset_for_learner

    def backtranslate(
        learner: PreTrainedModel,
        generator: PreTrainedModel,
        output_dir: Path,
        epoch: int,
        direction: Direction,
    ):
        data_collator = get_data_collator(learner)
        training_args = get_training_args(output_dir, epoch, direction)
        dataset = get_synthetic_dataset(generator, direction)
        dataset = dataset.train_test_split(test_size=0.1, load_from_cache_file=False)
        trainer = get_trainer(
            learner,
            training_args,
            data_collator,
            dataset["train"],
            dataset["test"],
            epoch,
            direction,
        )
        trainer.train()

    fw_output_dir = output_dir / Direction.FORWARD.value
    bw_output_dir = output_dir / Direction.BACKWARD.value
    if overwrite_output_dir:
        shutil.rmtree(fw_output_dir, ignore_errors=True)
        shutil.rmtree(bw_output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    fw_output_dir.mkdir(exist_ok=True)
    bw_output_dir.mkdir(exist_ok=True)

    fw_seq2seq = fw_seq2seq.to(cfg.DEVICE)  # FIXME: distributed bug
    bw_seq2seq = fw_seq2seq.to(cfg.DEVICE)

    print("Unsupervised Training...")
    for epoch in range(num_train_epochs):
        backtranslate(fw_seq2seq, bw_seq2seq, fw_output_dir, epoch, Direction.FORWARD)
        backtranslate(bw_seq2seq, fw_seq2seq, bw_output_dir, epoch, Direction.BACKWARD)


def train_supervised(
    seq2seq: PreTrainedModel,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    overwrite_output_dir: bool = False,
    num_train_epochs: int = 1,
) -> PreTrainedModel:
    output_dir.mkdir(exist_ok=True, parents=True)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=seq2seq,
        max_length=tokenizer.model_max_length,
        padding="max_length",
    )

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir.as_posix(),
        overwrite_output_dir=overwrite_output_dir,
        do_train=True,
        do_eval=True,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        fp16=True,
        push_to_hub=False,
    )

    trainer = transformers.Seq2SeqTrainer(
        model=seq2seq,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    return seq2seq


def main(
    paths: OutputManager,
    token_args: preprocess.TokenizerArgs,
    supervised: bool = False,
    unsupervised: bool = False,
) -> None:
    if not supervised and not unsupervised:
        raise ValueError("Either supervised or unsupervised must be True.")
    if supervised and unsupervised:
        raise ValueError("Only one of supervised or unsupervised can be True.")

    tokenizer, dataset = preprocess.main(
        paths,
        token_args,
        preprocess.DatasetArgs(),
        pseudosupervised=supervised,
        unsupervised=unsupervised,
    )
    print(f"{tokenizer=}", flush=True)
    print(f"{dataset=}", flush=True)

    def seq2seq() -> PreTrainedModel:
        return build_seq2seq_model(
            get_highest_path(paths.encoder, lstrip="checkpoint-"),
            get_highest_path(paths.decoder, lstrip="checkpoint-"),
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )

    if supervised:
        return train_supervised(seq2seq(), dataset, tokenizer, paths.pseudo_supervised)
    if unsupervised:
        return train_unsupervised(seq2seq(), seq2seq(), dataset, tokenizer, paths.unsupervised)

    raise Exception("This should never happen.")


def debug() -> None:
    main(
        OutputManager(),
        preprocess.TokenizerArgs(),
        True,
        False,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Your program description.")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--unsupervised", action="store_true")
    parser.add_argument("--model", default=preprocess.TokenizerArgs.model, type=str)
    parser.add_argument("--mode", type=str, default=preprocess.DatasetArgs.mode)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        debug()
        sys.exit(0)

    main(
        OutputManager(),
        preprocess.TokenizerArgs(args.model),
        args.supervised,
        args.unsupervised,
    )
