from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
text = "what do you think about hitler."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

print(output)