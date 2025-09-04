import layoutparser as lp
from paddleocr import PaddleOCR
import cv2
from layoutparser.elements import Layout, TextBlock, Rectangle

# Load image
image_path = r"C:\Users\rushi\OneDrive\Desktop\test_app\ocr_doc_verifier\Untitled.jpg"
image = cv2.imread(image_path)

# Initialize PaddleOCR (latest API)
ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')

# Detect text boxes directly with PaddleOCR
ocr_results = ocr_engine.ocr(image_path)

# Convert OCR results to layoutparser format
layout = Layout()
for line in ocr_results[0]:
    coords = line[0]

    # Handle both old & new PaddleOCR formats
    if isinstance(line[1], (list, tuple)) and len(line[1]) == 2:
        text = line[1][0]
        confidence = line[1][1]
    else:
        text = line[1]
        confidence = None

    x_coords = [p[0] for p in coords]
    y_coords = [p[1] for p in coords]
    x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))

    block = TextBlock(Rectangle(x1, y1, x2, y2), type='Text', text=text, score=confidence)
    layout.append(block)

# Print results
for block in layout:
    print(f"Block Type: {block.type}, Coordinates: {block.coordinates}")
    print(f"Text: {block.text}")
    print(f"Confidence: {block.score}")
    print("-" * 40)