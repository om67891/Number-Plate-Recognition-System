import os
import cv2
import xml.etree.ElementTree as ET

IMG_DIR = "m1/dataset/raw/images"
ANN_DIR = "m1/dataset/raw/annotations"
OUT_DIR = "m1/dataset/cropped"

os.makedirs(OUT_DIR, exist_ok=True)

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def crop_all():
    for file in os.listdir(ANN_DIR):
        if not file.endswith(".xml"):
            continue

        xml_path = os.path.join(ANN_DIR, file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_name = root.find("filename").text
        img_path = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"❌ Missing image: {img_name}")
            continue

        img = cv2.imread(img_path)

        boxes = parse_xml(xml_path)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop = img[y1:y2, x1:x2]

            save_name = f"{os.path.splitext(img_name)[0]}_{i}.jpg"
            save_path = os.path.join(OUT_DIR, save_name)

            cv2.imwrite(save_path, crop)

            print(f"✅ Saved: {save_name}")

if __name__ == "__main__":
    crop_all()
