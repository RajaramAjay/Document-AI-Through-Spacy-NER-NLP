import sys
import os
import toml
import cv2, numpy as np
from abc import ABC, abstractmethod
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.logger import setup_logger
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'config', 'config.toml'))
config = toml.load(config_path)
east_model_path = config['paths']['east_model_path']


logger = setup_logger()

class TextDetector(ABC):
    @abstractmethod
    def detect_text_areas(self, img):
        pass

class EASTTextDetection(TextDetector):
    def __init__(self, min_confidence=0.5, padding=5):
        try:
            self.net = cv2.dnn.readNet(east_model_path)
            self.layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"
            ]
            self.min_confidence = min_confidence
            self.padding = padding
            logger.info("EASTTextDetection initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing EASTTextDetection: {e}")
            raise

    def non_max_suppression(self, boxes, probs=None, overlapThresh=0.3):
        # logger.debug("Starting non-max suppression.")
        if len(boxes) == 0:
            logger.warning("No boxes provided for non-max suppression.")
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y1 - y2 + 1)
        idxs = y2

        if probs is not None:
            idxs = probs

        idxs = np.argsort(idxs)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        logger.debug("Non-max suppression completed.")
        return boxes[pick].astype("int")

    def preprocess_image(self, image):
        try:
            # logger.debug("Preprocessing image for text detection.")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            (H, W) = image.shape[:2]
            (newW, newH) = (int(W / 32) * 32, int(H / 32) * 32)
            rW = W / float(newW)
            rH = H / float(newH)

            image = cv2.resize(image, (newW, newH))
            # logger.debug("Image preprocessing complete.")
            return image, (H, W), (newW, newH), rW, rH
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None, None, None, None, None

    def detect_text(self, image):
        try:
            # logger.debug("Starting text detection.")
            blob = cv2.dnn.blobFromImage(image, 1.0, (image.shape[1], image.shape[0]),
                                         (123.68, 116.78, 103.94), swapRB=True, crop=False)
            self.net.setInput(blob)
            (scores, geometry) = self.net.forward(self.layerNames)

            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []

            for y in range(0, numRows):
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                for x in range(0, numCols):
                    if scoresData[x] < self.min_confidence:
                        continue

                    (offsetX, offsetY) = (x * 4.0, y * 4.0)
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])

            logger.debug("Text detection complete.")
            return rects, confidences
        except Exception as e:
            logger.error(f"Error detecting text: {e}")
            return [], []

    def box_process(self, boxes, newW, newH):
        try:
            # logger.debug("Processing bounding boxes.")
            mask = np.zeros((newH, newW), dtype=np.uint8)
            for (startX, startY, endX, endY) in boxes:
                cv2.rectangle(mask, (startX, startY), (endX, endY), 255, -1)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 4))
            dilated = cv2.dilate(mask, kernel, iterations=10)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            text_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x = max(0, x - self.padding)
                y = max(0, y - self.padding)
                w = min(newW - x, w + 2 * self.padding)
                h = min(newH - y, h + 2 * self.padding)
                text_boxes.append((x, y, w, h))

            # logger.debug("Bounding box processing complete.")
            return text_boxes
        except Exception as e:
            logger.error(f"Error processing boxes: {e}")
            return []

    def detect_text_areas(self, image):
        try:
            # logger.info("Detecting text areas in the image.")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            orig = image.copy()
            image, (H, W), (newW, newH), rW, rH = self.preprocess_image(image)

            rects, confidences = self.detect_text(image)

            boxes = self.non_max_suppression(np.array(rects), probs=confidences)

            text_boxes = self.box_process(boxes, newW, newH)

            final_boxes = []
            for (x, y, w, h) in text_boxes:
                startX, startY = int(x * rW), int(y * rH)
                endX, endY = int((x + w) * rW), int((y + h) * rH)

                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
                final_boxes.append((startX, startY, endX, endY))

            logger.info("Text area detection complete.")
            return orig, final_boxes
        except Exception as e:
            logger.error(f"Error detecting text areas: {e}")
            return None, []