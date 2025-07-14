import json
import os
from PIL import Image

json_file = "labels.json"
image_folder = "Chess_dataset" 
output_labels = "labels"
os.makedirs(output_labels, exist_ok=True)

with open(json_file) as f:
    coco = json.load(f)

category_map = {cat["id"]: i for i, cat in enumerate(coco["categories"])}  # YOLO uses 0-indexed

image_info = {}
for img in coco["images"]:
    image_info[img["id"]] = {
        "filename": img["file_name"],
        "width": img["width"],
        "height": img["height"]
    }

for img in image_info.values():
    open(os.path.join(output_labels, os.path.splitext(img["filename"])[0] + ".txt"), 'w').close()

for ann in coco["annotations"]:
    image_id = ann["image_id"]
    cat_id = ann["category_id"]
    bbox = ann["bbox"] 
    
    x, y, w, h = bbox
    img_w, img_h = image_info[image_id]["width"], image_info[image_id]["height"]
    
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    
    class_id = category_map[cat_id]
    
    label_file = os.path.join(output_labels, os.path.splitext(image_info[image_id]["filename"])[0] + ".txt")
    with open(label_file, 'a') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
