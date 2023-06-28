"""
Pretrain encoder and decoder separately on language modeling tasks.
"""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys

from datasets import DatasetDict
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
)

import preprocess
from preprocess import DatasetArgs, TokenizerArgs
from utils import get_num_workers, message, OutputManager


@dataclass
class PretrainArgs:
    tokenizer: str = preprocess.TokenizerArgs.model
    downsize: int = 1
    patience: int = 5
    threshold: int = 0
    tr_encoder: bool = field(default=False, metadata={"help": "Whether to train the encoder."})
    tr_decoder: bool = field(default=False, metadata={"help": "Whether to train the decoder."})


def train_mlm_encoder(
    dataset: DatasetDict, tokenizer: PreTrainedTokenizer, pretrain_args: PretrainArgs, training_args: TrainingArguments
) -> PreTrainedModel:
    config = transformers.BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768 // pretrain_args.downsize,
        num_hidden_layers=12 // pretrain_args.downsize,
        num_attention_heads=12 // pretrain_args.downsize,
        intermediate_size=3072 // pretrain_args.downsize,
        max_position_embeddings=tokenizer.model_max_length,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=0.2,
    )
    encoder = AutoModelForMaskedLM.from_config(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer)
    trainer = Trainer(
        model=encoder,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(pretrain_args.patience, pretrain_args.threshold)],
    )
    print("Training encoder...", flush=True)
    trainer.train()

    return encoder


def train_clm_decoder(
    dataset: DatasetDict, tokenizer: PreTrainedTokenizer, pretrain_args: PretrainArgs, training_args: TrainingArguments
) -> PreTrainedModel:
    config = transformers.GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=tokenizer.model_max_length,
        n_embd=768 // pretrain_args.downsize,
        n_layer=12 // pretrain_args.downsize,
        n_head=12 // pretrain_args.downsize,
        n_inner=None,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoder = AutoModelForCausalLM.from_config(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=decoder,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(pretrain_args.patience, pretrain_args.threshold)],
    )
    print("Training decoder...", flush=True)
    trainer.train()

    return decoder


def main(
    om: OutputManager, token_args: TokenizerArgs, pretrain_args: PretrainArgs, training_args: TrainingArguments
) -> None:
    print(f"{os.path.basename(__file__)}.main", flush=True)
    print(f"pretrain_args=\n{pretrain_args}", flush=True)
    print(f"training_args=\n{training_args}", flush=True)
    print(f"{get_num_workers()=}", flush=True)

    tokenizer, dataset = preprocess.main(om, token_args, DatasetArgs(), pretrain=True)
    print(f"{tokenizer=}", flush=True)
    print(f"{dataset=}", flush=True)

    # We preprocessed data so this removes some pesky warnings that are not relevant
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    print(f"{os.environ['TOKENIZERS_PARALLELISM']=}", flush=True)

    if pretrain_args.tr_encoder:
        encoder = train_mlm_encoder(dataset, tokenizer, pretrain_args, training_args)
        print(f"encoder=\n{encoder}", flush=True)
    if pretrain_args.tr_decoder:
        decoder = train_clm_decoder(dataset, tokenizer, pretrain_args, training_args)
        print(f"decoder=\n{decoder}", flush=True)


def cli() -> None:
    print(message(True, __file__), flush=True)

    parser = HfArgumentParser((PretrainArgs, TrainingArguments))
    args = parser.parse_args()
    pretrain_args, training_args = parser.parse_args_into_dataclasses()

    om = OutputManager()
    if pretrain_args.tr_encoder:
        training_args.output_dir = om.encoder
    elif pretrain_args.tr_decoder:
        training_args.output_dir = om.decoder

    main(om, TokenizerArgs(args.tokenizer), pretrain_args, training_args)

    print(message(False, __file__), flush=True)


if __name__ == "__main__":
    cli()
