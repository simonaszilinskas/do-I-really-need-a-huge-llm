# my_text_classifier.py

from transformers import AutoTokenizer, ModernBertForSequenceClassification
import torch
from typing import Optional
from pathlib import Path

# Replace this path with the location of your fine-tuned ModernBERT model
_THIS_DIR = Path(__file__).resolve().parent
_MODEL_DIR = str(_THIS_DIR.parent / "model/ecorouter-modernbert")

# Load tokenizer and model at import time so we don’t pay the I/O cost on every call
try:
    _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_DIR)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer from {_MODEL_DIR}: {e}")

try:
    _MODEL = ModernBertForSequenceClassification.from_pretrained(_MODEL_DIR)
    _MODEL.eval()  # evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load model from {_MODEL_DIR}: {e}")


def predict_label(
    text: str,
    tokenizer: Optional[AutoTokenizer] = None,
    model: Optional[ModernBertForSequenceClassification] = None
) -> str:
    """
    Classify a single string and return the predicted label.
    
    Args:
        text: The input text to classify.
        tokenizer: (optional) tokenizer instance; defaults to the module-level tokenizer.
        model: (optional) model instance; defaults to the module-level model.
    
    Returns:
        The name of the predicted class (as per `model.config.id2label`).
    
    Raises:
        ValueError: if the model’s config has no `id2label` mapping.
    """
    tok = tokenizer or _TOKENIZER
    mdl = model or _MODEL

    # Tokenize
    inputs = tok([text], padding=True, truncation=True, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = mdl(**inputs)

    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=-1).item()

    # Map to label name
    if not hasattr(mdl.config, "id2label") or len(mdl.config.id2label) == 0:
        raise ValueError("Model config does not contain an id2label mapping.")
    return mdl.config.id2label[pred_id]

"""
# Example usage
if __name__ == "__main__":
    sample = "This is an example sentence to classify."
    label = predict_label(sample)
    print(f"Input: {sample}\nPredicted label: {label}")

"""