import re
import spacy
from run_pipeline import get_extracted_text

from typing import Optional
entities: dict[str, Optional[str]] = {
    "NAME": None,
    "DOB": None,
    "MOBILE": None,
    "ADDRESS": None
}


# Get text from pipeline
image_path = r"D:\internship\clg_internship\automated-video-KYC\ocr_doc_verifier\test_imgs\hd_1.jpeg"
text = get_extracted_text(image_path)

nlp = spacy.load("en_core_web_sm")

entities = {"NAME": None, "DOB": None, "MOBILE": None, "ADDRESS": None}

doc = nlp(text)
for ent in doc.ents:
    if ent.label_ == "PERSON" and not entities["NAME"]:
        entities["NAME"] = ent.text
    elif ent.label_ in ["GPE", "LOC"]:
        if entities["ADDRESS"]:
            entities["ADDRESS"] += ", " + ent.text
        else:
            entities["ADDRESS"] = ent.text

dob_pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
mobile_pattern = r"\b(\+91[-\s]?\d{10}|\d{10})\b"

dob_match = re.search(dob_pattern, text)
mobile_match = re.search(mobile_pattern, text)

if dob_match:
    entities["DOB"] = dob_match.group(1)
if mobile_match:
    entities["MOBILE"] = mobile_match.group(1)

print("\nExtracted Entities:")
for key, value in entities.items():
    print(f"{key}: {value}")
