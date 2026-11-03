"""Self-contained PyTorch model and tokenizer for SYNTRA."""

from .model import LocalTransformer, TransformerConfig, select_device
from .tokenizer import BytePairTokenizer

__all__ = ["LocalTransformer", "TransformerConfig", "select_device", "BytePairTokenizer"]
