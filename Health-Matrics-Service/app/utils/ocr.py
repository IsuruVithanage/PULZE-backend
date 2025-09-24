"""
Contains utility functions for performing Optical Character Recognition (OCR)
on various file types using the Tesseract engine.
"""

import os
from PIL import Image
from pdf2image import convert_from_path
from pytesseract import image_to_string


def _convert_pdf_to_images(pdf_file: str) -> list:
    """Converts a PDF file into a list of PIL Image objects."""
    return convert_from_path(pdf_file)


def _convert_image_to_text(image_file) -> str:
    """Extracts text from a single PIL Image object using Tesseract."""
    # Tesseract configuration for better layout detection
    custom_config = r'--oem 3 --psm 6'
    return image_to_string(image_file, config=custom_config)


def _is_image(file_path: str) -> bool:
    """Checks if a file path has a common image extension."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    return os.path.splitext(file_path)[1].lower() in image_extensions


def get_text_from_any_file(file_path: str) -> str:
    """
    Extracts raw text from a given file, supporting both PDF and image formats.

    Args:
        file_path (str): The local path to the file.

    Raises:
        ValueError: If the file type is not a supported PDF or image format.

    Returns:
        str: The extracted text content from the file.
    """
    final_text = ""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        images = _convert_pdf_to_images(file_path)
        for img in images:
            final_text += _convert_image_to_text(img) + "\n"
    elif _is_image(file_path):
        with Image.open(file_path) as img:
            final_text = _convert_image_to_text(img)
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or an image.")

    return final_text