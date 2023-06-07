"""
"""

from argparse import ArgumentParser
from pathlib import Path

from datasets import DatasetDict
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

# from backtranslation import BacktranslationTrainer, BacktranslationTrainingArguments, DataCollatorForBacktranslation
import preprocess
import pretrain
import utils


SEQ2SEQ_PATH = Path("./models/seq2seq")


def train_seq2seq(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    encoder_pretrained_path: Path = None,
    decoder_pretrained_path: Path = None,
    output_dir: Path = SEQ2SEQ_PATH,
    overwrite_output_dir: bool = False,
) -> PreTrainedModel:

    if encoder_pretrained_path is None:
        encoder_pretrained_path = utils.get_highest_path(pretrain.ENCODER_PATH, lstrip="checkpoint-")
    if decoder_pretrained_path is None:
        decoder_pretrained_path = utils.get_highest_path(pretrain.DECODER_PATH, lstrip="checkpoint-")

    # This will cause warnings that can be ignored.
    # The encoder was pretrained, so its head is not needed, so its head is removed.
    # The decoder will have a new head attached to it, which initally contains random weights.
    seq2seq = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_path.as_posix(), decoder_pretrained_path.as_posix()
    )
    seq2seq.config.decoder_start_token_id = tokenizer.bos_token_id
    seq2seq.config.forced_bos_token_id = True
    seq2seq.config.eos_token_id = tokenizer.eos_token_id
    seq2seq.config.forced_eos_token_id = True
    seq2seq.config.pad_token_id = tokenizer.pad_token_id

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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=3,
        num_train_epochs=2,
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


def main(paths: preprocess.PathArgs, token_args: preprocess.TokenizerArgs) -> None:
    tokenizer, dataset = preprocess.main(paths, token_args, preprocess.DatasetArgs(), seq2seq=True)
    print(f"{tokenizer=}", flush=True)
    print(f"{dataset=}", flush=True)
    train_seq2seq(dataset, tokenizer)


if __name__ == "__main__":
    parser = ArgumentParser(description="Your program description.")
    parser.add_argument("--root", type=Path, help="Path")
    parser.add_argument("--model", default=preprocess.TokenizerArgs.model, type=str)
    parser.add_argument("--mode", type=str, default=preprocess.DatasetArgs.mode)
    args = parser.parse_args()

    main(
        preprocess.PathArgs(args.root),
        preprocess.TokenizerArgs(args.model),
    )
