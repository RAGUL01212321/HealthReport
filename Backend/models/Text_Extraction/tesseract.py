import cv2
import pytesseract
import json

json_path="Data\Mid_Process\extracted_text.json"
img_path = "Data/sample/image.png"

img = cv2.imread(img_path)
# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply threshold to convert to binary image
threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# Pass the image through pytesseract
text = pytesseract.image_to_string(threshold_img)

data={
    "filename" : img_path,
    "extracted_text" : text
}

with open(json_path, 'w') as json_file:
    json.dump(data,json_file,indent=4)

print("Extracted text")