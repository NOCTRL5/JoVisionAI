import sys
from PIL import Image
import pytesseract
import os
import numpy as np
import pandas as pnd


data = []
img_directory = f"{os.curdir}/task3images/"
img_files = [f for f in os.listdir(img_directory)]
img_files.sort(key=lambda x: int(os.path.splitext(x.replace('.jpg', ''))[0]))

for file_name in img_files:
    try:
        file = os.path.join(img_directory, file_name)
        img = Image.open(file).convert('L')
        width, height = img.size
        
        right_half = img.crop((width // 2, 0, width, height - 8)).rotate(-90)
        width, height = right_half.size
        
        img_np = np.array(right_half)
        min_color_val = np.min(img_np[img_np >= 30]) if np.any(img_np >= 30) else 255

        thumb_end = int(width * .40)
        remaining_width = width - thumb_end
        finger_width = remaining_width // 4
        
        finger_section = {
            "Thumb": (0, height - 56, thumb_end, height - 24),
            "Index": (thumb_end, 0, thumb_end + finger_width, height // 2),
            "Middle": (int(thumb_end + finger_width), 0, int(thumb_end + 2 * finger_width), height // 2),
            "Ring": (int(thumb_end + 2 * finger_width), 0, int(thumb_end + 3 * finger_width), height // 2),
            "Pinky": (int(thumb_end + 3 * finger_width), 0, int(thumb_end + 4 * finger_width), height // 2),
        }
        fingers = {finger: right_half.crop(box) for finger, box in finger_section.items()}
        
        finger_pressure = {}
        for finger, cropped_img in fingers.items():
            x = (np.mean(np.array(cropped_img)) + np.std(np.array(cropped_img))) / 2
            finger_pressure[finger] = 1 if x >= min_color_val else 0

        data.append({
            "Image": file_name.replace('.jpg', ''),
            "Thumb": finger_pressure["Thumb"],
            "Index": finger_pressure["Index"],
            "Middle": finger_pressure["Middle"],
            "Ring": finger_pressure["Ring"],
            "Pinky": finger_pressure["Pinky"]
        })

        print(f'{file_name}: {finger_pressure}\n')

    except Exception as e:
        raise RuntimeError(f'Error processing {file_name}: {e}')
    finally:
        right_half.close()

df = pnd.DataFrame(data)

output_file = f"{os.curdir}/finger_pressure.xlsx"
df.to_excel(output_file, index=False)
print(f"Data has been saved to '{output_file}'.")