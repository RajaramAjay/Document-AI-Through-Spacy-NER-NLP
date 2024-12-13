import os, sys
from abc import ABC, abstractmethod
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import setup_logger
from src.json_cleaning import *
from src.doc_identify import *
logger = setup_logger()

class TextNoiseRemover(ABC):
    @abstractmethod
    def remove_text_noise(self, text):
        pass

class KeyValueExtractor(ABC):
    @abstractmethod
    def extract_key_value_pairs(self, recognized_text):
        pass


class CleanText(TextNoiseRemover):
    def __init__(self, text_noise):
        self.text_noise = text_noise

    def remove_text_noise(self, text):
        # logger.info("Removing text noise from recognized text.")
        for ele in self.text_noise:
            text = text.replace(ele, "")
        return text

class NLPKeyValueExtraction(KeyValueExtractor):
    def __init__(self, nlp, text_noise):
        self.nlp = nlp
        self.text_noise = text_noise
        self.text_noise_remover = CleanText(self.text_noise)

    
    def extract_key_value_pairs(self, recognized_text):
        # logger.info("Starting key-value extraction from recognized text.")
        key_value_pairs = {}
        
        try:
            document_type, document_name = identify_document_type(recognized_text)
            #print(document_type, document_name)
            for text in recognized_text:
                text = self.text_noise_remover.remove_text_noise(text)
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in key_value_pairs:
                        if ent.text not in key_value_pairs[ent.label_]:
                            key_value_pairs[ent.label_] += ", " + ent.text
                    else:
                        key_value_pairs[ent.label_] = ent.text

            logger.info(f"Cleaning text & Key-value extraction completed. Extracted {len(key_value_pairs)} pairs.")
        except Exception as e:
            logger.error(f"Error during key-value extraction: {e}")
        
        #print(key_value_pairs)

        cleaned_key_value_pairs = clean_ocr_json(key_value_pairs)
        return cleaned_key_value_pairs
    
    