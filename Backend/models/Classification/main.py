import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if torch.cuda.is_available():
    print("CUDA is available. Using GPU for inference.")
    device="cuda"
else:
    print("CUDA is not available. Using CPU for inference.")


model_name = "FacebookAI/roberta-base"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Get he input from the JSON file
'''
# Load input text
json_path = r"Data\Mid_Process\extracted_text.json"
with open(json_path, 'r') as json_file:
    extracted_text = json.load(json_file)

text = extracted_text["extracted_text"]
'''

text = "The patient was diagnosed with stage III non-small cell lung cancer (NSCLC) after presenting with a persistent cough and unintentional weight loss. A CT scan revealed a 4.2 cm mass in the upper lobe of the right lung along with mediastinal lymphadenopathy. A biopsy confirmed adenocarcinoma, and molecular testing identified an EGFR mutation. The treatment plan includes targeted therapy with an EGFR inhibitor followed by concurrent chemoradiotherapy. Regular follow-ups are scheduled to monitor tumor response and manage potential side effects."


inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Make prediction
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class = torch.argmax(logits, dim=-1).item()

# Label map
labels = ["Cardiology", "Oncology"]
print("Predicted Domain:", labels[predicted_class])
