
"""Train a small BERT classifier using a synthetic teaching dataset.
This is a demonstration baseline, not a research-grade dataset.
"""

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

MODEL_NAME = "distilbert-base-uncased"

OUT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "bert_prompt_classifier"
)

# Training examples
samples = [
    ("help me", 0),
    ("write something", 0),
    ("tell me about AI", 0),

    ("explain python in simple words", 1),
    ("write a short email about leave", 1),
    ("summarize this article in five points", 1),

    ("Act as a Python teacher. Explain loops to beginners using numbered steps, with one code example and under 150 words.", 2),

    ("You are a career mentor. Compare data analyst and data scientist roles for final-year students in a table, include skills and salary factors, and keep it under 300 words.", 2),

    ("Act as a nutrition educator. Explain balanced meals for college students in simple language, use headings and bullet points, avoid medical claims, and keep it under 200 words.", 2),
]

# Repeat examples for demonstration only
samples = samples * 20

dataset = Dataset.from_dict({
    "text": [x[0] for x in samples],
    "label": [x[1] for x in samples]
})

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

encoded = dataset.map(tokenize, batched=True)

# Load classification model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# Evaluation metrics
def metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro")
    }

# Training settings
args = TrainingArguments(
    output_dir=str(OUT.parent / "bert_training"),
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=5,
    report_to="none"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    compute_metrics=metrics
)

# Train model
print("\nStarting BERT training...\n")

trainer.train()

# Evaluate model
print("\nModel Evaluation:")
print(trainer.evaluate())

# Save trained model
OUT.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(OUT))
tokenizer.save_pretrained(str(OUT))

print("\nBERT training completed successfully!")
print(f"Model saved at: {OUT}")
