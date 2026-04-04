"""
REBEL Triplet Extractor
========================
Extract (subject, relation, object) triples using Babelscape/rebel-large.
A BART-based seq2seq model purpose-built for relation extraction.
200+ relation types, single forward pass per chunk, no JSON parsing needed.
Automatically uses GPU when available.
"""

import logging
from typing import List, Tuple

from oml.utils.device import resolve_device

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_model_device: str = "cpu"   # tracks which device the cached model lives on


def _load_model(device: str = "auto") -> tuple:
    """
    Lazy-load REBEL model and tokenizer (cached globally).
    Re-loads if the requested device differs from the cached device.
    """
    global _model, _tokenizer, _model_device

    target = resolve_device(device)

    if _model is not None and _model_device == target:
        return _model, _tokenizer

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"Loading REBEL relation extraction model on {target}...")
        _tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        _model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        _model = _model.to(target)
        _model.eval()
        _model_device = target
        logger.info(f"REBEL model loaded on {target}.")
        return _model, _tokenizer
    except Exception as e:
        logger.error(f"Failed to load REBEL model: {e}")
        raise


def _parse_rebel_output(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse REBEL's special token output format into structured triples.
    
    REBEL outputs text like:
    <triplet> subject <subj> relation <obj> object <triplet> ...
    """
    triples = []
    # Split by <triplet> token
    parts = text.split("<triplet>")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        try:
            # Format: "subject <subj> relation <obj> object"
            if "<subj>" in part and "<obj>" in part:
                subj_split = part.split("<subj>")
                subject = subj_split[0].strip()

                rest = subj_split[1] if len(subj_split) > 1 else ""
                obj_split = rest.split("<obj>")
                relation = obj_split[0].strip()
                obj = obj_split[1].strip() if len(obj_split) > 1 else ""

                if subject and relation and obj:
                    triples.append((subject, relation, obj))
        except (IndexError, ValueError) as e:
            logger.debug(f"Failed to parse REBEL output part '{part}': {e}")
            continue

    return triples


def extract_triples_rebel(
    text: str, max_length: int = 512, device: str = "auto"
) -> List[Tuple[str, str, str]]:
    """
    Extract knowledge graph triples from text using REBEL.

    Args:
        text: Input text to extract triples from
        max_length: Maximum input token length
        device: ``"auto"`` (default), ``"cuda"``, or ``"cpu"``

    Returns:
        List of (subject, predicate, object) tuples
    """
    if not text or not text.strip():
        return []

    try:
        model, tokenizer = _load_model(device)
    except Exception:
        return []

    import torch
    target = resolve_device(device)

    # Tokenize; use BatchEncoding.to() to move all tensors to the target device
    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    inputs = inputs.to(target)  # BatchEncoding supports .to(device) natively

    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,  # attribute access works with BatchEncoding & mocks
            max_length=256,
            num_beams=3,
            length_penalty=1.0,
        )

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    decoded = decoded.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()

    triples = _parse_rebel_output(decoded)
    return triples


def extract_triples_rebel_batch(
    texts: List[str], max_length: int = 512, batch_size: int = 8,
    device: str = "auto"
) -> List[List[Tuple[str, str, str]]]:
    """
    Batch extract triples from multiple texts.

    Returns:
        List of triple lists, one per input text
    """
    all_triples = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            triples = extract_triples_rebel(text, max_length, device=device)
            all_triples.append(triples)
    return all_triples
