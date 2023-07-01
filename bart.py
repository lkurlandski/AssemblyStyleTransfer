#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for denoising language modeling on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=bart
"""
# You can also adapt this script on your own denoising language modeling task. Pointers for this are left as comments.

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
import torch
from torch import tensor
from tqdm import tqdm

from datasets import Dataset, DatasetDict
from transformers import (
    CONFIG_MAPPING,
    AutoTokenizer,
    BartConfig,
    BatchEncoding,
    BartForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    is_tensorboard_available,
    set_seed,
)
import transformers
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.utils import get_full_repo_name, send_example_telemetry

import cfg
import tokenization
from utils import OutputManager


# MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

MODEL_TYPES = (
    ("albert", "FlaxAlbertForMaskedLM"),
    ("bart", "FlaxBartForConditionalGeneration"),
    ("bert", "FlaxBertForMaskedLM"),
    ("big_bird", "FlaxBigBirdForMaskedLM"),
    ("distilbert", "FlaxDistilBertForMaskedLM"),
    ("electra", "FlaxElectraForMaskedLM"),
    ("mbart", "FlaxMBartForConditionalGeneration"),
    ("roberta", "FlaxRobertaForMaskedLM"),
    ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForMaskedLM"),
    ("roformer", "FlaxRoFormerForMaskedLM"),
    ("xlm-roberta", "FlaxXLMRobertaForMaskedLM"),
)
MODEL_TYPES = [i for i, _ in MODEL_TYPES]


import evaluate
rouge = evaluate.load("rouge")

import numpy as np

DATASET_PATH = Path("output/datasets/bart")


def get_compute_metric_fn(tokenizer):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    return compute_metrics


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.3, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    permute_sentence_ratio: float = field(
        default=1.0, metadata={"help": "Ratio of sentences to be permuted in each document"}
    )
    poisson_lambda: float = field(
        default=3.0, metadata={"help": "Mean of Poisson distribution used to generate span-lengths to be masked"}
    )

@dataclass
class DataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling. The code is largely copied from
    `<https://github.com/morganmcg1/rotobart/blob/main/data_collator.py#L223>`__.
    For more information on how BART denoising language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.13461.pdf>`__
    or the `official code for preprocessing <https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/denoising_dataset.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
        mask_ratio (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input
        poisson_lambda (:obj:`float`):
            Mean parameter of Poisson distribution used to generate span-lengths to be masked
        permute_sentence_ratio (:obj:`float`):
            Ratio of sentences to be permuted in each document
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    max_length: int
    padding = "max_length"
    pad_to_multiple_of = None
    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0
    permute_sentence_ratio: float = 1.0

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token or eos token token which is necessary for denoising"
                " language modeling. "
            )

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        encodings = [BatchEncoding({"input_ids": examples[i]["input_ids"]}) for i in range(len(examples))]
        batch = self.tokenizer.pad(
            encodings,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # batch = BatchEncoding(
        #     {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        # )
        batch["labels"] = torch.clone(batch["input_ids"])
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )
        # permuting sentences
        do_permute = False
        if self.permute_sentence_ratio > 0.0:
            batch["input_ids"] = tensor(self.permute_sentences(batch["input_ids"].cpu().detach().numpy()))
            do_permute = True

        # masking span of tokens (text infilling in the paper)
        if self.mask_ratio:
            tmp_input_ids, tmp_labels = self.span_mask_tokens(
                batch["input_ids"].cpu().detach().numpy(),
                batch["labels"].cpu().detach().numpy(),
                do_permute,
            )
            batch["input_ids"], batch["labels"] = tensor(tmp_input_ids), tensor(tmp_labels)

        # ignore pad tokens
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).int()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).int()
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
        example_has_multiple_sentences, num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)
        num_sentences_map = dict(zip(example_has_multiple_sentences, num_sentences))

        num_to_permute = np.ceil(num_sentences * self.permute_sentence_ratio).astype(int)
        num_to_permute_map = dict(zip(example_has_multiple_sentences, num_to_permute))

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])
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
                sentence = input_ids[i, (sentence_ends_map[i][j - 1] if j > 0 else 0) : sentence_ends_map[i][j]]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def span_mask_tokens(self, input_ids, labels, do_permute):
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """
        
        special_tokens_mask_labels = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask_inputs = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
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
                [span_lengths, np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))]
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


def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_bart_dlm", model_args, data_args)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    
    SUBSET = 1000
    DOWNSCALE = 4
    MAX_SEQ_LENGTH = 1024 // DOWNSCALE

    om = OutputManager()
    try:
        print("Loading from disk")
        datasets = DatasetDict.load_from_disk((DATASET_PATH / "raw").as_posix())
    except FileNotFoundError:
        print("Generating dataset")
        datasets = tokenization.get_raw_assembly_dataset(list(om.pre_normalized.rglob("*.asm")), min_lines=4).train_test_split()
        datasets.save_to_disk((DATASET_PATH / "raw").as_posix())
    tokenizer = tokenization.get_tokenizer(om.tokenizers, "WordLevel")
    tokenizer = tokenization.get_fast_tokenizer(
        tokenizer,
        model_max_length=MAX_SEQ_LENGTH,
        padding_side="right",
    )
    datasets["train"] = datasets["train"].select(range(SUBSET))
    datasets["test"] = datasets["test"].select(range(SUBSET))
    
    print(datasets)
    print(tokenizer)

    config = BartConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=MAX_SEQ_LENGTH,
        encoder_layers=12 // DOWNSCALE,
        encoder_ffn_dim=4096 // DOWNSCALE,
        encoder_attention_heads=16 // DOWNSCALE,
        decoder_layers=12 // DOWNSCALE,
        decoder_ffn_dim=4096 // DOWNSCALE,
        d_model=1024 // DOWNSCALE,
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["test"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Use Punkt Sentence Tokenizer to divide a document into a list of sentences
    # nltk.download("punkt")
    # sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    # split_datasets = datasets
    def sentence_split_function(example):
        sents = example["text"].split(cfg.INS)
        # use pad token as end of sentence indicator
        new_text = tokenizer.bos_token + f"{tokenizer.pad_token}".join(sents) + tokenizer.eos_token
        return {"text": new_text}
    
    p = (DATASET_PATH / "split")
    if p.exists():
        split_datasets = DatasetDict.load_from_disk(p.as_posix())
    else:
        split_datasets = datasets.map(
            sentence_split_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        split_datasets.save_to_disk(p.as_posix())

    # Tokenize every text, then concatenate them together before splitting them in smaller parts.
    # Since we make sure that all sequences are of the same length, no attention_mask is needed.
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], add_special_tokens=False, return_attention_mask=False)

    p = (DATASET_PATH / "tokenized")
    if p.exists():
        tokenized_datasets = DatasetDict.load_from_disk(p.as_posix())
    else:
        tokenized_datasets = split_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=text_column_name,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        tokenized_datasets.save_to_disk(p.as_posix())

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= MAX_SEQ_LENGTH:
            total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    try:
        tokenized_datasets = DatasetDict.load_from_disk((DATASET_PATH / "grouped").as_posix())
    except FileNotFoundError:
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        tokenized_datasets.save_to_disk((DATASET_PATH / "grouped").as_posix())

    # Initialize our training
    # rng = jax.random.PRNGKey(training_args.seed)
    # dropout_rngs = jax.random.split(rng, jax.local_device_count())

    config.vocab_size = len(tokenizer)
    model = BartForConditionalGeneration(
        config,
        # seed=training_args.seed,
        # dtype=getattr(jnp, model_args.dtype),
    )
    
    # tokenizer.set_truncation_and_padding(
    #     padding_strategy=transformers.utils.PaddingStrategy("max_length"),
    #     truncation_strategy=transformers.tokenization_utils_base.TruncationStrategy("longest_first"),
    #     max_length=config.max_position_embeddings,
    #     stride=0,
    #     pad_to_multiple_of=None,
    # )
    
    print(tokenized_datasets)
    print(len(tokenized_datasets["train"][0]["input_ids"]))
    print(tokenizer)
    
    print(f"PARAMTERS: {sum(p.numel() for p in model.parameters())}")

    # Data collator
    # This one will take care of randomly masking the tokens and permuting the sentences.
    data_collator = DataCollatorForBartDenoisingLM(
        max_length=config.max_position_embeddings,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        mask_ratio=data_args.mlm_probability,
        poisson_lambda=data_args.poisson_lambda,
        permute_sentence_ratio=data_args.permute_sentence_ratio,
    )
    
    training_args = Seq2SeqTrainingArguments(
        overwrite_output_dir=True,
        output_dir=training_args.output_dir,
        fp16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=100,
        load_best_model_at_end=True,
        dataloader_num_workers=len(os.sched_getaffinity(0)),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=get_compute_metric_fn(tokenizer),
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()

if __name__ == "__main__":
    main()
