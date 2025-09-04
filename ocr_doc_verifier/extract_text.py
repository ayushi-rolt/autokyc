from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\rushi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(opencv_image):
    pil_img = Image.fromarray(opencv_image)
   # config = r'--oem 3 --psm 6'
  #  text = pytesseract.image_to_string(pil_img, config=config)
    text = pytesseract.image_to_string(pil_img)
    return text
