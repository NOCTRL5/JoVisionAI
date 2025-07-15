import torch
from torchvision import transforms, models
from PIL import Image
import os

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model architecture
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6 classes
model.load_state_dict(torch.load("model.pth", map_location=device))  # use correct path
model = model.to(device)
model.eval()

# Class names from your training set
class_names = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']  # or whatever yours are

# Evaluate on random images
random_dir = 'random'

for filename in os.listdir(random_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(random_dir, filename)
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = class_names[pred_idx]
        print(f"{filename}: {pred_label}")
