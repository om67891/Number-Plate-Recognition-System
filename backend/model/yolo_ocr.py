from pathlib import Path
import re

import cv2
import easyocr
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "runs" / "detect" / "number_plate_detector" / "weights" / "best.pt"

# Load YOLO model
model = YOLO(str(MODEL_PATH))

# Load OCR
reader = easyocr.Reader(["en"], gpu=False)


def clean_plate(text):
    text = text.upper()

    # remove unwanted characters
    text = re.sub(r"[^A-Z0-9 ]", "", text)

    return text


def detect_and_read(image_path):
    results = model(image_path)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            return "NO PLATE DETECTED"

        # Take first detected plate
        x1, y1, x2, y2 = map(int, boxes[0])

        img = cv2.imread(image_path)
        plate = img[y1:y2, x1:x2]

        # OCR
        ocr_result = reader.readtext(plate)

        if not ocr_result:
            return "UNREADABLE"

        raw_text = ocr_result[0][1]
        cleaned_text = clean_plate(raw_text)

        return cleaned_text

    return "UNREADABLE"
