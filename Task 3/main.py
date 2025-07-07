import sys
from PIL import Image
import pytesseract
import os


date =[]
img_directory = f"{os.curdir}/Task 3/task3images/"
image_files = [f for f in os.listdir(img_directory) if f.endswith('.jpg')]
image_files.sort(key=lambda x: int(os.path.splitext(x.replace('.jpg', ''))[0]))

