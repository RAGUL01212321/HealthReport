import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np

dataset = load_dataset("TimSchopf/medical_abstracts")

# Inspect one sample
print(dataset["train"][0])

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    # Shift labels from 1..5 → 0..4
    labels = []
    for label in batch["condition_label"]:
        labels.append(label - 1)
    
    batch["labels"] = labels

    # Tokenize the abstract text
    tokenized = tokenizer(
        batch["medical_abstract"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # Add labels to the tokenized batch
    tokenized["labels"] = batch["labels"]

    return tokenized

tokenized = dataset.map(preprocess, batched=True)

num_labels = 5
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",  # <- replace
    eval_steps=500,               # <- how often you want eval
    save_strategy="steps",
    save_steps=500
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(results)

trainer.save_model("./bert-medical-abstracts")
tokenizer.save_pretrained("./bert-medical-abstracts")
