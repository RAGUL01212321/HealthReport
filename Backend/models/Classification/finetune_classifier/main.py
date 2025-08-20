import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np

if torch.cuda.is_available():
    print("CUDA is available")
    print(torch.cuda.get_device_name(0))
else:
    print("No CUDA") 

#---------------------------------------------------------------------------------------------#
dataset = load_dataset("TimSchopf/medical_abstracts")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = 5
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
#---------------------------------------------------------------------------------------------#

def preprocess(batch):
    # Convert condition_label to zero-based index
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
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

training_args = TrainingArguments(
    output_dir="Backend/models/Classification/finetune_classifier/Output_Model/results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="Backend/models/Classification/finetune_classifier/Output_Model/logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
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

trainer.save_model("Backend/models/Classification/finetune_classifier/Output_Model/bert-medical-abstracts")
tokenizer.save_pretrained("Backend/models/Classification/finetune_classifier/Output_Model/bert-medical-abstracts")
