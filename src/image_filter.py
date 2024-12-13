import cv2
import numpy as np
from abc import ABC, abstractmethod
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.logger import setup_logger
import time

logger = setup_logger()

class ImageFilter(ABC):
    @abstractmethod
    def apply_filter(self, img):
        """Abstract method for applying a filter to an image."""
        pass

class LinesFilter(ImageFilter):
    def __init__(self, vertical_scale=25, horizontal_scale=25, dilation_iter=1):
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.dilation_iter = dilation_iter

    def detect_lines(self, img):
        # logger.debug("Detecting lines in the image.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        rows = bw.shape[0]
        vertical_size = int(rows / self.vertical_scale)
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

        vertical_lines = cv2.erode(bw, vertical_structure)
        vertical_lines = cv2.dilate(vertical_lines, vertical_structure, iterations=self.dilation_iter)

        cols = bw.shape[1]
        horizontal_size = int(cols / self.horizontal_scale)
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

        horizontal_lines = cv2.erode(bw, horizontal_structure)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_structure, iterations=self.dilation_iter)

        lines = cv2.add(vertical_lines, horizontal_lines)
        logger.debug("Line detection complete.")
        return bw, lines

    def remove_lines_inpaint(self, img, lines):
        # logger.debug("Removing lines using inpainting.")
        mask = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    def apply_filter(self, img):
        try:
            # logger.info("Applying filter to image.")
            start_time = time.time()  # Start time for performance monitoring
            
            bw, lines = self.detect_lines(img)
            processed_img = self.remove_lines_inpaint(cv2.bitwise_not(bw), lines)

            elapsed_time = time.time() - start_time  # Calculate elapsed time
            logger.info(f"Filter applied successfully in {elapsed_time:.2f} seconds.")
            return processed_img
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return None