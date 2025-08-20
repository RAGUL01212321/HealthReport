from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

'''
LABEL_0 : Neoplasms
LABEL_1 : Digestive system diseases
LABEL_2 : Nervous system diseases
LABEL_3 : Cardiovascular diseases
LABEL_4 : General pathological conditions
'''

# Path to your saved model
model_path = r"Backend\models\Classification\finetune_classifier\Output_Model\bert-medical-abstracts"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Create pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example text
text = "Unfading acral microlivedo. A discrete marker of thrombotic skin disease associated with antiphospholipid antibody syndrome. "

# Get prediction
prediction = classifier(text)
print(prediction)