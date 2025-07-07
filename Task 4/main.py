import torch
from PIL import Image
import torchvision.transforms as transforms
import os


image_directory = f"{os.curdir}/Task4/Chess_dataset/"
image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
image_files.sort(key=lambda x: int(os.path.splitext(x
                                                    .replace('.jpg', '')
                                                    .replace('(', '')
                                                    .replace(')', ''))[0]))

print('Loading model...')
model = torch.hub.load('ultralytics/yolo11n.pt', 'yolo11n')
print('model loaded.')
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
for image in image_files:
    file = os.path.join(image_directory, image)
    img = Image.open(file).convert('RGB')
    img_transformed = transform(img).unsqueeze(0)
    results = model(img_transformed)