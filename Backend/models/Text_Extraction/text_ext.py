from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance
import torch


'''
Models already used 
    - microsoft/trocr-base-stage1
    - stepfun-ai/GOT-OCR2_0
'''


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for inference.")


model_name = "microsoft/trocr-base-stage1"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

image_in=Image.open(r"C:\Users\uragu\College\Sem 3\NLP\Data\sample\rec1.jpg")
image_in = image_in.convert("L") 
enhancer = ImageEnhance.Contrast(image_in)
image_in = enhancer.enhance(2)
image_in = image_in.convert("RGB")
image_in = image_in.resize((1280, 720))  # Resize to a more manageable size

pixel_values = processor(images=image_in, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

# Decode the generated text from the model output
extracted_text = processor.decode(generated_ids[0], skip_special_tokens=True)
print(extracted_text)