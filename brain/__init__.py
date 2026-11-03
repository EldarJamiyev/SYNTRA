"""Self-contained PyTorch model and tokenizer for SYNTRA."""

from .tokenizer import BytePairTokenizer

try:
    from .model import LocalTransformer, TransformerConfig, select_device
    __all__ = ["LocalTransformer", "TransformerConfig", "select_device", "BytePairTokenizer"]
except ImportError:
    LocalTransformer = None  # type: ignore[assignment]
    TransformerConfig = None  # type: ignore[assignment]
    select_device = None  # type: ignore[assignment]
    __all__ = ["BytePairTokenizer"]

