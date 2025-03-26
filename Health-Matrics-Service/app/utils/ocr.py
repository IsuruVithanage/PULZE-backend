from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image
import os

def convert_pdf_to_img(pdf_file: str):
    return convert_from_path(pdf_file)

def convert_image_to_text(image_file):
    return image_to_string(image_file)

def is_image(file_path: str) -> bool:
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    return os.path.splitext(file_path)[1].lower() in image_extensions

def get_text_from_any_file(file_path: str) -> str:
    final_text = ""
    if file_path.lower().endswith('.pdf'):
        images = convert_pdf_to_img(file_path)
        for img in images:
            final_text += convert_image_to_text(img)
    elif is_image(file_path):
        with Image.open(file_path) as img:
            final_text = convert_image_to_text(img)
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or an image.")
    return final_text
