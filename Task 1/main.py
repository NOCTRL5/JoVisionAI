import sys
from PIL import Image
import pytesseract

def img_text(img_path):
    try:
        img = Image.open(img_path)
        img.verify()
        img = Image.open(img_path)
        print(f"Image {img_path} is valid.")
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")
    
if __name__ == "__main__":
    assert(len(sys.argv) == 2)

    image_path = sys.argv[1].strip("\"")
    print(img_text(image_path))