import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#---------------------------------------CUDA Check------------------------------------------#
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
        max_length=500
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

#---------------------------------------------------------------------------------------------#

def plot_metrics(trainer, metric="loss"):
    """
    Plot training and evaluation metrics from the trainer logs.
    
    Args:
        trainer: Hugging Face Trainer object
        metric: str, which metric to plot ("loss", "accuracy", "f1", etc.)
    """
    logs = trainer.state.log_history
    
    # Collect values
    steps = []
    train_values = []
    eval_values = []

    for entry in logs:
        if "loss" in entry and metric == "loss":
            steps.append(entry["step"])
            train_values.append(entry["loss"])
        if "eval_" + metric in entry:
            eval_values.append(entry["eval_" + metric])
    
    # Plot
    plt.figure(figsize=(8, 6))
    if train_values:
        plt.plot(steps[:len(train_values)], train_values, label=f"train_{metric}")
    if eval_values:
        plt.plot(steps[:len(eval_values)], eval_values, label=f"eval_{metric}")
    
    plt.xlabel("Steps")
    plt.ylabel(metric.capitalize())
    plt.title(f"Training and Evaluation {metric.capitalize()} Over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(trainer, dataset, labels=None, normalize=False, title="Confusion Matrix"):
    """
    Plots the confusion matrix for a Hugging Face Trainer on a given dataset.

    Args:
        trainer: Hugging Face Trainer object
        dataset: Dataset to evaluate (e.g., tokenized["test"])
        labels: List of label names (optional)
        normalize: If True, shows percentages instead of raw counts
        title: Title of the plot
    """
    # Run predictions
    predictions = trainer.predict(dataset)

    # Extract true and predicted labels
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.show()
#---------------------------------------------------------------------------------------------#


plot_metrics(trainer, metric="loss")
plot_confusion_matrix(trainer, tokenized["test"], labels=["Neoplasms", "Digestive system diseases", "Nervous system diseases", "Cardiovascular diseases", "General pathological conditions"])