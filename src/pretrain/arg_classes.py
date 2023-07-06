"""

"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenizerArguments:
    max_length: Optional[int] = field(default=128, metadata={"help": ""})
    vocab_size: Optional[int] = field(default=1024, metadata={"help": ""})
    tok_algorithm: Optional[str] = field(default="WordLevel", metadata={"help": ""})
    tok_use_saved: Optional[bool] = field(default=True, metadata={"help": ""})
    tok_overwrite: Optional[bool] = field(default=False, metadata={"help": ""})
    tok_batch_size: Optional[int] = field(default=512, metadata={"help": ""})
    tok_n_files: Optional[int] = field(default=None, metadata={"help": ""})


@dataclass
class DatasetArguments:
    dat_path: str = field(metadata={"help": ""})
    dat_n_examples: Optional[int] = field(default=None, metadata={"help": ""})
    dat_n_files: Optional[int] = field(default=None, metadata={"help": ""})
    num_proc: Optional[int] = field(default=None, metadata={"help": ""})
    validation_split: Optional[float] = field(default=0.1, metadata={"help": ""})
    dat_use_saved: Optional[bool] = field(default=True, metadata={"help": ""})
    dat_overwrite: Optional[bool] = field(default=False, metadata={"help": ""})


@dataclass
class BERTArguments:
    hidden_size: Optional[int] = field(default=768, metadata={"help": ""})
    num_hidden_layers: Optional[int] = field(default=12, metadata={"help": ""})
    num_attention_heads: Optional[int] = field(default=12, metadata={"help": ""})
    intermediate_size: Optional[int] = field(default=3072, metadata={"help": ""})
    downsize: Optional[int] = field(default=None, metadata={"help": ""})

    def __post_init__(self):
        if self.downsize:
            self.hidden_size = self.hidden_size // self.downsize
            self.num_hidden_layers = self.num_hidden_layers // self.downsize
            self.num_attention_heads = self.num_attention_heads // self.downsize
            self.intermediate_size = self.intermediate_size // self.downsize


@dataclass
class GPTArguments:
    n_embd: Optional[int] = field(default=768, metadata={"help": ""})
    n_layer: Optional[int] = field(default=12, metadata={"help": ""})
    n_head: Optional[int] = field(default=12, metadata={"help": ""})
    downsize: Optional[int] = field(default=None, metadata={"help": ""})

    def __post_init__(self):
        if self.downsize:
            self.n_embd = self.n_embd // self.downsize
            self.n_layer = self.n_layer // self.downsize
            self.n_head = self.n_head // self.downsize


@dataclass
class BARTArguments:
    encoder_layers: int = field(default=12, metadata={"help": ""})
    encoder_ffn_dim: int = field(default=4096, metadata={"help": ""})
    encoder_attention_heads: int = field(default=16, metadata={"help": ""})
    decoder_layers: int = field(default=12, metadata={"help": ""})
    decoder_ffn_dim: int = field(default=4096, metadata={"help": ""})
    decoder_attention_heads: int = field(default=16, metadata={"help": ""})
    d_model: int = field(default=1024, metadata={"help": ""})
    downsize: int = field(default=None, metadata={"help": "1 -> 370M; 2 -> 50.M; 4 -> 8.7M"})

    def __post_init__(self):
        if self.downsize:
            self.encoder_layers = self.encoder_layers // self.downsize
            self.encoder_ffn_dim = self.encoder_ffn_dim // self.downsize
            self.encoder_attention_heads = self.encoder_attention_heads // self.downsize
            self.decoder_layers = self.decoder_layers // self.downsize
            self.decoder_ffn_dim = self.decoder_ffn_dim // self.downsize
            self.decoder_attention_heads = self.decoder_attention_heads // self.downsize
            self.d_model = self.d_model // self.downsize


@dataclass
class DAEArguments:
    mlm_probability: float = field(default=0.3, metadata={"help": ""})
    permute_sentence_ratio: float = field(default=1.0, metadata={"help": ""})
    poisson_lambda: float = field(default=3.0, metadata={"help": ""})
