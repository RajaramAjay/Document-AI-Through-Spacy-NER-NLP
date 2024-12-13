import os, sys
from abc import ABC, abstractmethod
import pytesseract
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
from src.logger import setup_logger
logger = setup_logger()

class TextRecognizer(ABC):
    @abstractmethod
    def recognize_text_in_boxes(self, image, text_boxes):
        pass

class PytesseractTextRecognition(TextRecognizer):
    def __init__(self):
        pass

    def recognize_text_in_boxes(self, image, text_boxes):
        logger.info("Starting text recognition in boxes.")
        recognized_text = []
        
        try:
            # Sort the boxes by Y-coordinate and then by X-coordinate to maintain reading order
            text_boxes = sorted(text_boxes, key=lambda box: (box[1], box[0]))

            # Function to process a single text box
            def process_box(box):
                startX, startY, endX, endY = box
                roi = image[startY:endY, startX:endX]
                return pytesseract.image_to_string(roi, config='--psm 6').strip().replace('\n', ' ')

            # Run OCR in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_box, text_boxes))

            # Filter out empty results
            recognized_text = [text for text in results if text]

            logger.info(f"Text recognition completed. Recognized {len(recognized_text)} texts.")
        
        except Exception as e:
            logger.error(f"Error during text recognition: {e}")

        return recognized_text