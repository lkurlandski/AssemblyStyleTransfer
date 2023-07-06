"""

Source:
    https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_bart_dlm_flax.py
"""

from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pprint import pprint
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import evaluate
import numpy as np
import torch
from torch import tensor
from transformers import (
    BartConfig,
    BatchEncoding,
    BartForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.bart.modeling_bart import shift_tokens_right

from src.cfg import *
from src.pretrain.arg_classes import (
    TokenizerArguments,
    DatasetArguments,
    BARTArguments,
    DAEArguments,
)
from src.pretrain.prep import get_tokenizer_and_dataset
from src.pretrain.utils import get_callbacks
from src.utilization import status
from src.utils import count_parameters


@dataclass
class DataCollatorForDAE:
    max_length: int
    padding = "max_length"
    pad_to_multiple_of = None
    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0
    permute_sentence_ratio: float = 1.0
    device: int = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token or eos token token which is necessary for denoising"
                " language modeling. "
            )

    def __call__(self, examples: list[dict[str, list[int]]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        encodings = [
            BatchEncoding({"input_ids": examples[i]["input_ids"]}) for i in range(len(examples))
        ]
        batch = self.tokenizer.pad(
            encodings,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        self.device = batch["input_ids"].device
        print(status(self.device))
        batch["labels"] = torch.clone(batch["input_ids"])
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )
        # permuting sentences
        do_permute = False
        if self.permute_sentence_ratio > 0.0:
            batch["input_ids"] = tensor(
                self.permute_sentences(batch["input_ids"].cpu().detach().numpy())
            )
            do_permute = True

        print(status(self.device))

        # masking span of tokens (text infilling in the paper)
        if self.mask_ratio:
            tmp_input_ids, tmp_labels = self.span_mask_tokens(
                batch["input_ids"].cpu().detach().numpy(),
                batch["labels"].cpu().detach().numpy(),
                do_permute,
            )
            batch["input_ids"], batch["labels"] = tensor(tmp_input_ids), tensor(tmp_labels)
        print(status(self.device))
        # ignore pad tokens
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).int()
        batch["decoder_attention_mask"] = (
            batch["decoder_input_ids"] != self.tokenizer.pad_token_id
        ).int()
        return batch

    def permute_sentences(self, input_ids):
        """
        Shuffle sentences in each document.
        """
        results = input_ids.copy()

        # find end locations of sentences
        end_sentence_mask = input_ids == self.tokenizer.pad_token_id
        sentence_ends = np.argwhere(end_sentence_mask)
        sentence_ends[:, 1] += 1
        example_has_multiple_sentences, num_sentences = np.unique(
            sentence_ends[:, 0], return_counts=True
        )
        num_sentences_map = dict(zip(example_has_multiple_sentences, num_sentences))

        num_to_permute = np.ceil(num_sentences * self.permute_sentence_ratio).astype(int)
        num_to_permute_map = dict(zip(example_has_multiple_sentences, num_to_permute))

        sentence_ends = np.split(
            sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:]
        )
        sentence_ends_map = dict(zip(example_has_multiple_sentences, sentence_ends))

        for i in range(input_ids.shape[0]):
            if i not in example_has_multiple_sentences:
                continue
            substitutions = np.random.permutation(num_sentences_map[i])[: num_to_permute_map[i]]
            ordering = np.arange(0, num_sentences_map[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute_map[i])]

            # write shuffled sentences into results
            index = 0
            for j in ordering:
                sentence = input_ids[
                    i, (sentence_ends_map[i][j - 1] if j > 0 else 0) : sentence_ends_map[i][j]
                ]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def span_mask_tokens(self, input_ids, labels, do_permute):
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """

        special_tokens_mask_labels = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask_inputs = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in input_ids.tolist()
        ]
        special_tokens_mask_labels = np.array(special_tokens_mask_labels, dtype=bool)
        special_tokens_mask_inputs = np.array(special_tokens_mask_inputs, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token_mask = ~(input_ids == self.tokenizer.pad_token_id) & ~special_tokens_mask_inputs
        num_tokens_to_mask = int(math.ceil(is_token_mask.astype(float).sum() * self.mask_ratio))
        if num_tokens_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        span_lengths = np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))
        while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
            span_lengths = np.concatenate(
                [
                    span_lengths,
                    np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,)),
                ]
            )

        # remove all spans of length 0
        # note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        span_lengths = span_lengths[span_lengths > 0]

        # trim to about num_tokens_to_mask tokens
        cutoff_idx = np.argmin(np.abs(np.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
        span_lengths = span_lengths[:cutoff_idx]

        # randomly choose starting positions for masking
        token_indices = np.argwhere(is_token_mask == 1)
        span_starts = np.random.permutation(token_indices.shape[0])[: span_lengths.shape[0]]
        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(input_ids, fill_value=False)

        # mask starting positions
        for mi in masked_indices:
            mask[tuple(mi)] = True
        span_lengths -= 1

        # fill up spans
        max_index = input_ids.shape[1] - 1
        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            span_lengths -= 1
            remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask_inputs)] = False
        input_ids[np.where(mask)] = self.tokenizer.mask_token_id
        if not do_permute:
            labels[np.where(mask == 0)] = -100
        else:
            labels[np.where(special_tokens_mask_labels)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_input_ids = np.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(input_ids):
            new_example = example[~to_remove[i]]
            new_input_ids[i, : new_example.shape[0]] = new_example

        return new_input_ids, labels


def get_compute_metric_fn(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def main():
    pprint({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
    print(BR, flush=True)

    parser = HfArgumentParser(
        [
            TokenizerArguments,
            DatasetArguments,
            BARTArguments,
            DAEArguments,
            Seq2SeqTrainingArguments,
        ]
    )
    args = parser.parse_args_into_dataclasses()
    tokenizer_args = args[0]
    dataset_args = args[1]
    model_args = args[2]
    dae_args = args[3]
    training_args = args[4]
    pprint(dataset_args)
    pprint(tokenizer_args)
    pprint(model_args)
    pprint(dae_args)
    pprint(training_args)
    print(BR, flush=True)

    tokenizer, dataset = get_tokenizer_and_dataset(tokenizer_args, dataset_args)

    config = BartConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=tokenizer_args.max_length,
        encoder_layers=model_args.encoder_layers,
        encoder_ffn_dim=model_args.encoder_ffn_dim,
        encoder_attention_heads=model_args.encoder_attention_heads,
        decoder_layers=model_args.decoder_layers,
        decoder_ffn_dim=model_args.decoder_ffn_dim,
        decoder_attention_heads=model_args.decoder_attention_heads,
        d_model=model_args.d_model,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
    )
    model = BartForConditionalGeneration(config)
    print(f"{model=}")
    print(f"{count_parameters(model)=}")
    print(BR, flush=True)

    data_collator = DataCollatorForDAE(
        max_length=tokenizer_args.max_length,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        mask_ratio=dae_args.mlm_probability,
        poisson_lambda=dae_args.poisson_lambda,
        permute_sentence_ratio=dae_args.permute_sentence_ratio,
    )
    callbacks = get_callbacks()
    print(f"{data_collator=}")
    print(f"{callbacks=}")
    print(BR, flush=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=get_compute_metric_fn(tokenizer),
        callbacks=callbacks,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    print(f"{BR}\nSTART @{datetime.now()}\n{BR}", flush=True)
    main()
    print(f"{BR}\nFINISH @{datetime.now()}\n{BR}", flush=True)
