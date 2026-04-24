import os
import torch
import cv2
import numpy as np
import string

from backend.model.ocr_model import CRNN

# Characters
characters = string.ascii_uppercase + string.digits
idx_to_char = {i+1: c for i, c in enumerate(characters)}

device = torch.device("cpu")

# Correct path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "m1", "saved_model", "model.pt")

num_classes = len(characters) + 1
model = CRNN(num_classes)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
else:
    model = None


def preprocess_plate(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.equalizeHist(image)
    image = cv2.resize(image, (128, 32))
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    image = image / 255.0

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    return torch.tensor(image, dtype=torch.float32)


def decode_output(output):
    output = torch.softmax(output, dim=2)
    output = output.permute(1, 0, 2)

    pred = torch.argmax(output, dim=2)

    result = []
    prev = -1

    for i in pred[0]:
        if i != prev and i != 0:
            result.append(idx_to_char.get(i.item(), ''))
        prev = i

    text = ''.join(result)

    if not text or len(text) < 4:
        return "UNREADABLE"

    return text


def predict_text(image):
    if model is None:
        return "UNREADABLE"

    img = preprocess_plate(image).to(device)

    with torch.no_grad():
        output = model(img)

    return decode_output(output)
