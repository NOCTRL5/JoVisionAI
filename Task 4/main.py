import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import os

img_directory = os.path.join(os.curdir,"Chess_dataset")
img_files = [f for f in os.listdir(img_directory)if f.endswith('.jpg')]
img_files.sort(key=lambda x: int(os.path.splitext(x
.replace('.jpg', '')
.replace('(', '')
.replace(')', ''))[0]))
print('Loading model...')
model = YOLO('yolo11n.pt')
print('model loaded.')
print("Model classes:", model.names)
transform = transforms.Compose([
    transforms.ToTensor(),
])
for image in img_files:
    file = os.path.join(img_directory, image)
    img = Image.open(file).convert('RGB')

    results = model(img, imgsz=640)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls.item())  # Class index
            conf = float(box.conf.item())  # Confidence
            xyxy = box.xyxy[0].tolist()  # Bounding box

            class_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
            print(f"Image: {image} | Class: {class_names[cls]} | Confidence: {conf:.2f} | Box: {xyxy}")