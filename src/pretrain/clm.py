"""
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
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)

from src.cfg import *
from src.pretrain.prep import (
    get_callbacks,
    get_tokenizer_and_dataset,
    TokenizerArguments,
    DatasetArguments,
)
from src.utils import count_parameters


@dataclass
class ModelArguments:
    n_embd: Optional[int] = field(default=768, metadata={"help": ""})
    n_layer: Optional[int] = field(default=12, metadata={"help": ""})
    n_head: Optional[int] = field(default=12, metadata={"help": ""})
    downsize: Optional[int] = field(default=None, metadata={"help": ""})

    def __post_init__(self):
        if self.downsize:
            self.n_embd = self.n_embd // self.downsize
            self.n_layer = self.n_layer // self.downsize
            self.n_head = self.n_head // self.downsize


def main():
    pprint({k: v for k, v in os.environ.items() if k.startswith("SLURM")})
    print(BR, flush=True)

    parser = HfArgumentParser(
        [
            TokenizerArguments,
            DatasetArguments,
            ModelArguments,
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
    config = transformers.OpenAIGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=tokenizer.model_max_length,
        n_embd=model_args.downsize,
        n_layer=model_args.downsize,
        n_head=model_args.downsize,
    )
    model = AutoModelForCausalLM.from_config(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
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
