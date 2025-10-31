#update venv flask google-cloud-vision


from flask import Flask, request, jsonify, render_template_string
from google.cloud import vision
from google.cloud.vision import types
import io, re

app = Flask(__name__)
client = vision.ImageAnnotatorClient()

def extract_text_from_image(file):
    content = file.read()
    # image = vision.Image(content=content)
    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def extract_fields(text):
    data = {}
    pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text)
    if pan_match:
        data['pan_number'] = pan_match.group()
    aadhaar_match = re.search(r'\d{4}\s\d{4}\s\d{4}', text)
    if aadhaar_match:
        data['aadhaar_number'] = aadhaar_match.group()
    name_match = re.search(r'^To\n[^\n]+\n([A-Za-z ]+)', text, re.MULTILINE)
    # name_match =  re.search(r'^(?:.*\n){0,1}([^\n]+)', text.strip(), re.MULTILINE)
    if name_match:
        data['name'] = name_match.group(1).strip()
    dob_match = re.search(r'DOB\s*:\s*(\d{2}/\d{2}/\d{4})', text)
    if dob_match:
        data['dob'] = dob_match.group(1)
    return data

# Home route with upload form
@app.route('/')
def home():
    return render_template_string('''
        <h2>PAN/Aadhaar Verification</h2>
        <form method="POST" action="/verify" enctype="multipart/form-data">
            <input type="file" name="document" required>
            <input type="submit" value="Upload & Verify">
        </form>
    ''')

@app.route('/verify', methods=['POST'])
def verify_document():
    file = request.files.get('document')
    if not file:
        return jsonify({"error": "No document uploaded"}), 400
    text = extract_text_from_image(file)
    data = extract_fields(text)
    return jsonify({"verified": True if data else False, "fields": data})

if __name__ == '__main__':
    app.run(debug=True)
