import sys
from PIL import Image
import pytesseract

def make_black(img_path):
    try:
        img = Image.open(img_path)
        result = img.copy()
        #img.verify()
        pixels = result.load()
        width, height = result.size
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                #Luminosity method
                gray = int(.299 * r + .587 * g + .114 * b)
                #Average method
                #gray = int((r + g + b) / 3)
                #Lightness method
                #gray = int((max(r, g, b) + min (r, g, b)) / 2)
                pixels[x, y] = (gray, gray, gray)
        return result
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")
    
if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    image_path = sys.argv[1].strip("\"")
    answer = make_black(image_path)
    answer.save("output.png", "PNG")
    answer.save("grayscale_{}".format(image_path.split("\\")[-1]))