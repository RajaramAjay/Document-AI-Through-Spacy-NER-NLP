import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import setup_logger
from src.image_filter import *
from src.textdetector import *
from src.text_recognition import *
from src.key_value_extractor import *
logger = setup_logger()

class ImageProcessor:
    def __init__(self, nlp, punc, image_filter: ImageFilter, text_detector: TextDetector, text_recognizer: TextRecognizer, key_value_extractor: KeyValueExtractor):
        self.nlp = nlp
        self.punc = punc
        self.image_filter = image_filter
        self.text_detector = text_detector
        self.text_recognizer = text_recognizer
        self.key_value_extractor = key_value_extractor

    def process_single_image(self, img_path):
        logger.info(f"Processing image at path: {img_path}")
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Image at path {img_path} could not be loaded.")
                return {}

            filtered_img = self.image_filter.apply_filter(img)
            logger.debug("Image filtering completed.")

            result_image, text_boxes = self.text_detector.detect_text_areas(filtered_img)
            logger.debug(f"Text detection completed. Found {len(text_boxes)} text boxes.")

            recognized_text = self.text_recognizer.recognize_text_in_boxes(result_image, text_boxes)
            logger.debug(f"Text recognition completed. Recognized {len(recognized_text)} text segments.")

            key_value_pairs = self.key_value_extractor.extract_key_value_pairs(recognized_text)
            logger.info(f"Key-value extraction completed. Extracted {len(key_value_pairs)} pairs.")

            return key_value_pairs
            
        except Exception as e:
            logger.error(f"Error during image processing: {e}")
            return {}

# # Factory for creating ImageProcessor instances
class ImageProcessorFactory:
    @staticmethod
    def create(nlp, punc):
        return ImageProcessor(
            nlp=nlp,
            punc=punc,
            image_filter=LinesFilter(),
            text_detector=EASTTextDetection(),
            text_recognizer=PytesseractTextRecognition(),
            key_value_extractor=NLPKeyValueExtraction(nlp, punc)
        )
    
# Updated ImageListProcessor and DirectoryProcessor
class ImageListProcessor:
    def __init__(self, nlp, text_noise):
        self.image_processor = ImageProcessorFactory.create(nlp, text_noise)

    def process_image_list(self, image_paths):
        all_folders_results = [{"folder_name": "Provided Images", "images": []}]
        for img_path in image_paths:
            if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_result = self.process_single_image(img_path)
                all_folders_results[0]["images"].append({
                    "image_filename": os.path.basename(img_path),
                    "key_value_pairs": image_result
                })
            else:
                raise ValueError(f"Invalid image path: {img_path}")
        return all_folders_results

    def process_single_image(self, img_path):
        return self.image_processor.process_single_image(img_path)
    
class DirectoryProcessor:
    def __init__(self, nlp, text_noise):
        self.image_processor = ImageProcessorFactory.create(nlp, text_noise)

    def process_directory(self, path):
        all_folders_results = []
        for root, _, files in os.walk(path):
            folder_name = os.path.basename(root)
            folder_result = {"folder_name": folder_name, "images": []}
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    img_path = os.path.join(root, filename)
                    image_result = self.process_single_image(img_path)
                    folder_result["images"].append({
                        "image_filename": filename,
                        "key_value_pairs": image_result
                    })
            if folder_result["images"]:
                all_folders_results.append(folder_result)
        return all_folders_results

    def process_single_image(self, img_path):
        return self.image_processor.process_single_image(img_path)