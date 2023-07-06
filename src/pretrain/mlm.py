"""
Pretrain encoder and decoder separately on language modeling tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
import os
from pprint import pprint
import sys
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import transformers
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)

from src.cfg import *
from src.pretrain.prep import get_tokenizer_and_dataset
from src.pretrain.arg_classes import (
    TokenizerArguments,
    DatasetArguments,
    BERTArguments,
)
from src.pretrain.utils import get_callbacks
from src.utils import count_parameters
from src.utilization import status


def main():
    pprint({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
    print(BR, flush=True)

    parser = HfArgumentParser(
        [
            TokenizerArguments,
            DatasetArguments,
            BERTArguments,
            TrainingArguments,
        ]
    )
    args = parser.parse_args_into_dataclasses()
    tokenizer_args = args[0]
    dataset_args = args[1]
    model_args = args[2]
    training_args = args[3]
    pprint(dataset_args)
    pprint(tokenizer_args)
    pprint(model_args)
    pprint(training_args)
    print(BR, flush=True)

    tokenizer, dataset = get_tokenizer_and_dataset(tokenizer_args, dataset_args)
    config = transformers.BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_position_embeddings=tokenizer.model_max_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = AutoModelForMaskedLM.from_config(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer)
    callbacks = get_callbacks()
    print(f"{config=}")
    print(f"{model=}")
    print(f"{count_parameters(model)=}")
    print(f"{data_collator=}")
    print(f"{callbacks=}")
    print(BR, flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    print(f"{BR}\nSTART @{datetime.now()}\n{BR}", flush=True)
    main()
    print(f"{BR}\nFINISH @{datetime.now()}\n{BR}", flush=True)
