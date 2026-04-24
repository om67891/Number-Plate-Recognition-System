from ultralytics import YOLO

# Load model (pretrained)
model = YOLO("yolov8n.pt")  # nano = fast

# Train
model.train(
    data="m1/dataset/yolo/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="number_plate_detector"
)