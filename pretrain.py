"""
Pretrain encoder and decoder separately on language modeling tasks.
"""

from argparse import ArgumentParser
from pathlib import Path

from datasets import DatasetDict
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
)

import preprocess
from preprocess import DatasetArgs, TokenizerArgs
from utils import OutputManager


PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1


def train_mlm_encoder(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    overwrite_output_dir: bool = False,
) -> PreTrainedModel:
    config = transformers.BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=tokenizer.model_max_length,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
    )
    encoder = AutoModelForMaskedLM.from_config(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer)
    train_args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        overwrite_output_dir=overwrite_output_dir,
        do_train=True,
        do_eval=True,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        save_total_limit=3,
        num_train_epochs=2,
        fp16=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=encoder,
        args=train_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    print("Training encoder...", flush=True)
    trainer.train()

    return encoder


def train_clm_decoder(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    overwrite_output_dir: bool = False,
) -> PreTrainedModel:
    config = transformers.GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=tokenizer.model_max_length,
        n_embd=64,
        n_layer=4,
        n_head=4,
        n_inner=1024,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoder = AutoModelForCausalLM.from_config(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=output_dir.as_posix(),
        overwrite_output_dir=overwrite_output_dir,
        do_train=True,
        do_eval=True,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        save_total_limit=3,
        num_train_epochs=2,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=decoder,
        args=train_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    print("Training decoder...", flush=True)
    trainer.train()

    return decoder


def main(
    paths: OutputManager,
    token_args: TokenizerArgs,
    tr_encoder: bool,
    tr_decoder: bool,
) -> None:
    fast_tokenizer, split_tokenized_dataset = preprocess.main(paths, token_args, DatasetArgs(), pretrain=True)
    print(f"{fast_tokenizer=}", flush=True)
    print(f"{split_tokenized_dataset=}", flush=True)

    if tr_encoder:
        encoder = train_mlm_encoder(split_tokenized_dataset, fast_tokenizer, paths.encoder, True)
        print(f"{encoder=}", flush=True)
    if tr_decoder:
        decoder = train_clm_decoder(split_tokenized_dataset, fast_tokenizer, paths.decoder, True)
        print(f"{decoder=}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Your program description.")
    parser.add_argument("--model", default=preprocess.TokenizerArgs.model, type=str)
    parser.add_argument("--tr_encoder", action="store_true")
    parser.add_argument("--tr_decoder", action="store_true")
    args = parser.parse_args()

    main(
        OutputManager(),
        TokenizerArgs(args.model),
        args.tr_encoder,
        args.tr_decoder,
    )
