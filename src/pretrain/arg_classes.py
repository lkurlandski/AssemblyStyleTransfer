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
    scale: Optional[float] = field(
        default=None,
        metadata={
            "help": ".5 -> 690M; .75 -> 210M; 1 -> 89M; 1.5 -> 27M; 2 -> 12.4M; 3 -> 4.3M, 4 -> 2.2M"
        },
    )

    def __post_init__(self):
        if self.scale:
            self.hidden_size = int(self.hidden_size / self.scale)
            self.num_hidden_layers = int(self.num_hidden_layers / self.scale)
            self.num_attention_heads = int(self.num_attention_heads / self.scale)
            self.intermediate_size = int(self.intermediate_size / self.scale)


@dataclass
class GPTArguments:
    n_embd: Optional[int] = field(default=768, metadata={"help": ""})
    n_layer: Optional[int] = field(default=12, metadata={"help": ""})
    n_head: Optional[int] = field(default=12, metadata={"help": ""})
    scale: Optional[float] = field(
        default=None,
        metadata={"help": "0.5: ; 0.75: 200M, 1: 88M ; 2: 12M ; 3: 4.2M, 4: 2.1M"},
    )

    def __post_init__(self):
        if self.scale:
            self.n_embd = int(self.n_embd / self.scale)
            self.n_layer = int(self.n_layer / self.scale)
            self.n_head = int(self.n_head / self.scale)


@dataclass
class BARTArguments:
    encoder_layers: int = field(default=12, metadata={"help": ""})
    encoder_ffn_dim: int = field(default=4096, metadata={"help": ""})
    encoder_attention_heads: int = field(default=16, metadata={"help": ""})
    decoder_layers: int = field(default=12, metadata={"help": ""})
    decoder_ffn_dim: int = field(default=4096, metadata={"help": ""})
    decoder_attention_heads: int = field(default=16, metadata={"help": ""})
    d_model: int = field(default=1024, metadata={"help": ""})
    scale: Optional[float] = field(
        default=None,
        metadata={"help": "0.5: 2.8B, 0.75: 850M; 1 -> 360M; 2 -> 46.M; 4 -> 6.6M"},
    )

    def __post_init__(self):
        if self.scale:
            self.encoder_layers = int(self.encoder_layers / self.scale)
            self.encoder_ffn_dim = int(self.encoder_ffn_dim / self.scale)
            self.encoder_attention_heads = int(self.encoder_attention_heads / self.scale)
            self.decoder_layers = int(self.decoder_layers / self.scale)
            self.decoder_ffn_dim = int(self.decoder_ffn_dim / self.scale)
            self.decoder_attention_heads = int(self.decoder_attention_heads / self.scale)
            self.d_model = int(self.d_model / self.scale)


@dataclass
class DAEArguments:
    mlm_probability: float = field(default=0.3, metadata={"help": ""})
    permute_sentence_ratio: float = field(default=1.0, metadata={"help": ""})
    poisson_lambda: float = field(default=3.0, metadata={"help": ""})
