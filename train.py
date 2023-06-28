"""
Training for seq2seq models.

# TODO: try to optimize the generation process
    - from optimum.bettertransformer import BetterTransformer
"""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import os
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
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import cfg
import preprocess
from utils import estimate_memory_needs, get_highest_path, get_num_workers, message, one_and_only_one, OutputManager


@dataclass
class TrainArgs:
    tokenizer: str = preprocess.TokenizerArgs.model
    patience: int = 5
    threshold: int = 0
    mode: str = preprocess.DatasetArgs.mode
    tr_supervised: bool = field(default=False, metadata={"help": "Whether to train the encoder."})
    tr_unsupervised: bool = field(default=False, metadata={"help": "Whether to train the decoder."})


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
    training_args: Seq2SeqTrainingArguments,
    generation_config: Optional[GenerationConfig] = None,
) -> PreTrainedModel:
    def get_single_run_data_collator(
        seq2seq: PreTrainedModel, epoch: int, direction: Direction
    ) -> DataCollatorForSeq2Seq:  # pylint: disable=unused-argument
        return DataCollatorForSeq2Seq(
            tokenizer, model=seq2seq, max_length=tokenizer.model_max_length, padding="max_length"
        )

    def get_single_run_training_args(
        output_dir: Path, epoch: int, direction: Direction
    ) -> Seq2SeqTrainingArguments:  # pylint: disable=unused-argument
        training_args_ = deepcopy(training_args)
        training_args_.num_train_epochs = 1
        training_args_.output_dir = output_dir
        return training_args_

    def get_single_run_trainer(
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
        # print(f"{generator=}\n{generator.device=}", flush=True)
        dataset_for_generator = get_directional_dataset(opposite_direction(direction))
        # print(f"{dataset_for_generator=}", flush=True)
        inputs_for_generator = tensor(dataset_for_generator["input_ids"], device=generator.device)
        # print(f"{inputs_for_generator=}\n{inputs_for_generator.device=}", flush=True)
        outputs = generator.generate(inputs_for_generator, generation_config)
        # print(f"{outputs=}", flush=True)
        labels = Dataset.from_dict({"labels": outputs})
        # print(f"{labels=}", flush=True)
        dataset_for_learner = get_directional_dataset(direction).remove_columns("labels")
        # print(f"{dataset_for_learner=}", flush=True)
        dataset_for_learner = datasets.concatenate_datasets([dataset_for_learner, labels], axis=1)
        # print(f"{dataset_for_learner=}", flush=True)
        return dataset_for_learner

    def backtranslate(
        learner: PreTrainedModel, generator: PreTrainedModel, output_dir: Path, epoch: int, direction: Direction
    ):
        data_collator = get_single_run_data_collator(learner, epoch, direction)
        training_args = get_single_run_training_args(output_dir, epoch, direction)
        dataset = get_synthetic_dataset(generator, direction)
        dataset = dataset.train_test_split(test_size=0.1, load_from_cache_file=False)
        trainer = get_single_run_trainer(
            learner, training_args, data_collator, dataset["train"], dataset["test"], epoch, direction
        )
        trainer.train()

    fw_output_dir = training_args.output_dir / Direction.FORWARD.value
    bw_output_dir = training_args.output_dir / Direction.BACKWARD.value
    if training_args.overwrite_output_dir:
        shutil.rmtree(fw_output_dir, ignore_errors=True)
        shutil.rmtree(bw_output_dir, ignore_errors=True)
    training_args.output_dir.mkdir(exist_ok=True, parents=True)
    fw_output_dir.mkdir(exist_ok=True)
    bw_output_dir.mkdir(exist_ok=True)

    fw_seq2seq = fw_seq2seq.to(cfg.DEVICE)  # FIXME: this will cause a bug when doing distributed training
    bw_seq2seq = fw_seq2seq.to(cfg.DEVICE)

    print("Unsupervised Training...", flush=True)
    for epoch in range(int(training_args.num_train_epochs)):
        backtranslate(fw_seq2seq, bw_seq2seq, fw_output_dir, epoch, Direction.FORWARD)
        backtranslate(bw_seq2seq, fw_seq2seq, bw_output_dir, epoch, Direction.BACKWARD)
        # TODO: implement early stopping

    return fw_seq2seq


def train_supervised(
    seq2seq: PreTrainedModel,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    training_args: Seq2SeqTrainingArguments,
) -> PreTrainedModel:
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, model=seq2seq, max_length=tokenizer.model_max_length, padding="max_length"
    )
    trainer = transformers.Seq2SeqTrainer(
        model=seq2seq,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    print("Unsupervised Training...", flush=True)
    trainer.train()

    return seq2seq


def main(
    paths: OutputManager,
    token_args: preprocess.TokenizerArgs,
    train_args: TrainArgs,
    training_args: Seq2SeqTrainingArguments,
) -> PreTrainedModel:
    if not one_and_only_one(train_args.tr_supervised, train_args.tr_unsupervised):
        raise ValueError("Either supervised or unsupervised must be True.")

    tokenizer, dataset = preprocess.main(
        paths,
        token_args,
        preprocess.DatasetArgs(),
        pseudosupervised=train_args.tr_supervised,
        unsupervised=train_args.tr_unsupervised,
    )
    print(f"{tokenizer=}", flush=True)
    print(f"{dataset=}", flush=True)

    # We preprocessed data so this removes some pesky warnings that are not relevant
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    print(f"{os.environ['TOKENIZERS_PARALLELISM']=}", flush=True)

    def seq2seq() -> PreTrainedModel:
        return build_seq2seq_model(
            get_highest_path(paths.encoder, lstrip="checkpoint-"),
            get_highest_path(paths.decoder, lstrip="checkpoint-"),
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Assuming dataset already tokenized, so settting TOKENIZERS_PARALLELISM=false")
    if train_args.tr_supervised:
        model = train_supervised(seq2seq(), dataset, tokenizer, training_args)
    if train_args.tr_unsupervised:
        model = train_unsupervised(seq2seq(), seq2seq(), dataset, tokenizer, training_args)

    print(f"{model=}", flush=True)
    return model


def debug() -> None:
    main(OutputManager(), preprocess.TokenizerArgs(), True, False)


def cli() -> None:
    print(message(True, __file__), flush=True)

    parser = HfArgumentParser((TrainArgs, Seq2SeqTrainingArguments))
    args = parser.parse_args()
    train_args, training_args = parser.parse_args_into_dataclasses()

    # if args.debug:
    #     debug()
    #     sys.exit(0)

    om = OutputManager()
    if train_args.tr_supervised:
        training_args.output_dir = om.pseudo_supervised
    elif train_args.tr_unsupervised:
        training_args.output_dir = om.unsupervised

    main(om, preprocess.TokenizerArgs(args.tokenizer), train_args, training_args)

    print(message(False, __file__), flush=True)


if __name__ == "__main__":
    cli()
