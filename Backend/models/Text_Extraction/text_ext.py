from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for inference.")


# Load the pretrained TrOCR model and processor
model_name = "microsoft/trocr-base-stage1"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Load the image containing the report

image_in=Image.open(r"C:\Users\uragu\College\Sem 3\NLP\Data\sample\rec1.jpg")

# Convert image to grayscale and enhance contrast
image_in = image_in.convert("L")  # Convert to grayscale
enhancer = ImageEnhance.Contrast(image_in)
image_in = enhancer.enhance(2)  # Increase contrast for better text clarity
image_in = image_in.convert("RGB")  # Convert back to RGB if necessary

# Optionally, resize the image (if necessary)
image_in = image_in.resize((1280, 720))  # Resize to a more manageable size

# Preprocess the image and pass it through the model
pixel_values = processor(images=image_in, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

# Decode the generated text from the model output
extracted_text = processor.decode(generated_ids[0], skip_special_tokens=True)

# Print the extracted text
print(extracted_text)
