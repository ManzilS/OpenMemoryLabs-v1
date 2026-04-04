"""
T5-Small Document Summarizer
=============================
Fast, local summarization using T5-Small (60M params).
Automatically uses GPU when available for ~5-10x speedup.
"""

import logging

from oml.utils.device import resolve_device

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_model_device: str = "cpu"   # tracks which device the cached model lives on


def _load_model(device: str = "auto") -> tuple:
    """
    Lazy-load T5-Small model and tokenizer (cached globally).
    Re-loads if the requested device differs from the cached device.
    """
    global _model, _tokenizer, _model_device

    target = resolve_device(device)

    # Return cached model if device matches
    if _model is not None and _model_device == target:
        return _model, _tokenizer

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"Loading T5-Small summarization model on {target}...")
        _tokenizer = AutoTokenizer.from_pretrained("t5-small")
        _model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        _model = _model.to(target)
        _model.eval()
        _model_device = target
        logger.info(f"T5-Small loaded on {target}.")
        return _model, _tokenizer
    except Exception as e:
        logger.error(f"Failed to load T5-Small: {e}")
        raise


class T5Summarizer:
    """Summarizes documents using T5-Small locally. Uses GPU automatically."""

    def __init__(self, max_input_tokens: int = 512, max_output_tokens: int = 150,
                 device: str = "auto"):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.device = resolve_device(device)
        # Trigger lazy load on init so failures surface early
        _load_model(self.device)

    def summarize(self, text: str) -> str:
        """Summarize a single piece of text."""
        if not text or not text.strip():
            return ""

        import torch
        model, tokenizer = _load_model(self.device)

        # T5 expects "summarize: " prefix
        input_text = "summarize: " + text.strip()

        # Tokenize; use BatchEncoding.to() to move all tensors to the target device
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
        )
        inputs = inputs.to(self.device)  # BatchEncoding supports .to(device) natively

        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,  # attribute access works with BatchEncoding & mocks
                max_length=self.max_output_tokens,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary.strip()

    def summarize_document(self, doc) -> str:
        """Summarize an OML Document object (drop-in replacement for Summarizer)."""
        text = getattr(doc, "clean_text", "") or ""
        if not text:
            return ""
        # For long documents, summarize the first chunk
        max_chars = 2000  # T5-Small works best with shorter input
        if len(text) > max_chars:
            text = text[:max_chars]
        return self.summarize(text)

    def summarize_batch(self, texts: list[str]) -> list[str]:
        """Summarize multiple texts efficiently."""
        return [self.summarize(t) for t in texts]
