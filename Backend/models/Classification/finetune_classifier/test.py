from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Path to your saved model
model_path = r"Backend\models\Classification\finetune_classifier\Output_Model\bert-medical-abstracts"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Create pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example text
text = ""

# Get prediction
prediction = classifier(text)
print(prediction)
