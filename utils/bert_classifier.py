"""U3 BERT sequence-classification inference helper."""
from pathlib import Path
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bert_prompt_classifier"

def bert_model_available():
    return MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()

def predict_prompt_quality(text):
    if not bert_model_available():
        raise RuntimeError(
            "BERT model is not trained yet. Run: python scripts/train_bert.py"
        )
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install the ML dependencies: pip install -r requirements-ml.txt") from exc
    classifier = pipeline("text-classification", model=str(MODEL_DIR), tokenizer=str(MODEL_DIR))
    result = classifier(text, truncation=True)[0]
    label = result["label"]
    label_map = {"LABEL_0": "Low", "LABEL_1": "Medium", "LABEL_2": "High"}
    return label_map.get(label, label), float(result["score"])
