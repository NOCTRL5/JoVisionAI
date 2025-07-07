import torch
from PIL import Image
import torchvision.transforms as transforms
import os

img_directory = f"{os.curdir}/Task 4/Chess_dataset/"
img_files = [f for f in os.listdir(img_directory)]
img_files.sort(key=lambda x: int(os.path.splitext(x
.replace('.jpg', '')
.replace('(', '')
.replace(')', ''))[0]))
print('Loading model...')
model = torch.load('/yolo11n.pt', 'yolo11n')
print('model loaded.')
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
for image in img_files:
    file = os.path.join(img_directory, image)
    img = Image.open(file).convert('RGB')
    img_transformed = transform(img).unsqueeze(0)
    results = model(img_transformed)