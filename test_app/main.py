from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import numpy as np
import cv2
import io, re
from motor.motor_asyncio import AsyncIOMotorClient
from bson import Binary
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------
# OCR & Vision Setup
# ---------------------------
# Try to import Google Vision
try:
    from google.cloud import vision
    from google.cloud.vision_v1 import types
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    print("WARNING: Google Cloud Vision not available. OCR features will be disabled.")

# Try to import Tesseract (Fallback - Optional)
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# ---------------------------
# MongoDB Connection Setup
# ---------------------------
# Ensure you have MONGO_URI in your .env file
MONGODB_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGODB_URL)
db = client["forms"]
users_collection = db["users"]
videos_collection = db["videos"]
documents_collection = db["documents"]


# ---------------------------
# Initialize FastAPI App
# ---------------------------
app = FastAPI(
    title="Face & Document Verification API",
    description="API for face recognition, age/gender prediction, and document verification"
)

# Allow CORS (Crucial for React/Next.js frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to ["http://localhost:3000"] for better security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# HELPER FUNCTIONS (OCR Logic)
# ---------------------------
if GOOGLE_VISION_AVAILABLE:
    google_client = vision.ImageAnnotatorClient()
else:
    google_client = None 

def extract_text_from_image(file_bytes):
    """
    Sends image bytes to Google Cloud Vision and returns raw text.
    """
    if not google_client:
        return ""
    try:
        image = types.Image(content=file_bytes)
        response = google_client.text_detection(image=image)
        texts = response.text_annotations
        return texts[0].description if texts else ""
    except Exception as e:
        print(f"Google Vision API Error: {e}")
        return ""

def extract_fields(text):
    """
    Extract fields using Regex.
    """
    data = {}
    
    # 🔍 Debug: Print text to console
    print(f"\n--- EXTRACTING FROM TEXT ---\n{text}\n----------------------------\n")

    # Your PAN regex
    pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text)
    if pan_match:
        data['pan_number'] = pan_match.group() 
    
    # Your Aadhaar regex
    aadhaar_match = re.search(r'\d{4}\s\d{4}\s\d{4}', text)
    if aadhaar_match:
        data['aadhaar_number'] = aadhaar_match.group()
    
    # Your Name regex
    name_match = re.search(r'^To\n[^\n]+\n([A-Za-z ]+)', text, re.MULTILINE)
    if name_match:
        data['name'] = name_match.group(1).strip()
    
    # Your DOB regex
    dob_match = re.search(r'DOB\s*:\s*(\d{2}/\d{2}/\d{4})', text)
    if dob_match:
        data['dob'] = dob_match.group(1)
        
    return data

# ---------------------------
# ROOT UI
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <html>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>🎯 KYC Verification API is Running</h1>
            <p>FastAPI is connected to MongoDB and ready to process requests.</p>
            <a href="/docs" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View Swagger Docs</a>
        </body>
    </html>
    """

# ---------------------------
# KYC ENDPOINTS
# ---------------------------

@app.post("/api/kyc/submit-user-data")
async def submit_user_data(
    fullName: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    photograph: UploadFile = File(...)
):
    try:
        user_id = str(uuid.uuid4())
        photo_content = await photograph.read()
        
        user_data = {
            "user_id": user_id,
            "fullName": fullName,
            "email": email,
            "phone": phone,
            "photograph": Binary(photo_content),
            "photograph_filename": photograph.filename,
            "created_at": datetime.utcnow(),
            "status": "pending"
        }
        
        await users_collection.insert_one(user_data)
        
        return JSONResponse({
            "user_id": user_id,
            "message": "User data stored successfully"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/kyc/submit-video")
async def submit_video(
    video: UploadFile = File(...),
    user_id: str = Form(...)
):
    try:
        video_content = await video.read()
        
        video_data = {
            "user_id": user_id,
            "video": Binary(video_content),
            "video_filename": video.filename,
            "uploaded_at": datetime.utcnow()
        }
        
        result = await videos_collection.insert_one(video_data)
        
        # Update user status
        await users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"video_uploaded": True}}
        )
        
        return JSONResponse({
            "message": "Video stored successfully",
            "video_id": str(result.inserted_id)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------
# PREVIEW ENDPOINT (For React Frontend)
# ---------------------------
@app.post("/verify-document/")
async def verify_document_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for frontend to preview/verify document BEFORE final submission.
    """
    try:
        # 1. Read file
        content = await file.read()
        
        # 2. Extract Data
        text = extract_text_from_image(content)
        data = extract_fields(text)
        
        # 3. Return extracted fields to frontend
        return {
            "verified": True if data else False,
            "fields": data,
            "raw_text": text[:200] 
        }
    except Exception as e:
        print(f"Verification Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------
# SUBMIT DOCUMENTS (Updated for Optional Files)
# ---------------------------
@app.post("/api/kyc/submit-documents")
async def submit_documents(
    aadhaar: UploadFile = File(None), # 👈 Changed to Optional
    pan: UploadFile = File(None),     # 👈 Changed to Optional
    user_id: str = Form(...)
):
    """
    1. Accepts Either Aadhaar OR Pan (or both).
    2. Runs OCR on uploaded files.
    3. Saves specific files to MongoDB.
    """
    try:
        # 1. Validation: Ensure at least one is present
        if not aadhaar and not pan:
             return JSONResponse(status_code=400, content={"error": "Please upload at least one document (Aadhaar or PAN)."})

        # 2. Initialize Data Structure
        documents_data = {
            "user_id": user_id,
            "uploaded_at": datetime.utcnow(),
            "ocr_results": {}
        }
        
        aadhaar_extracted_data = {}
        pan_extracted_data = {}
        aadhaar_text_raw = ""
        pan_text_raw = ""

        # 3. Process Aadhaar (Only if user uploaded it)
        if aadhaar:
            aadhaar_content = await aadhaar.read()
            # Store in DB object
            documents_data["aadhaar_blob"] = Binary(aadhaar_content)
            documents_data["aadhaar_filename"] = aadhaar.filename
            
            # Run OCR
            if GOOGLE_VISION_AVAILABLE and google_client:
                try:
                    aadhaar_text_raw = extract_text_from_image(aadhaar_content)
                    aadhaar_extracted_data = extract_fields(aadhaar_text_raw)
                    documents_data["ocr_results"]["aadhaar"] = aadhaar_extracted_data
                    documents_data["ocr_results"]["raw_text_aadhaar_preview"] = aadhaar_text_raw[:200]
                except Exception as e:
                    print(f"Aadhaar OCR Error: {e}")

        # 4. Process PAN (Only if user uploaded it)
        if pan:
            pan_content = await pan.read()
            # Store in DB object
            documents_data["pan_blob"] = Binary(pan_content)
            documents_data["pan_filename"] = pan.filename
            
            # Run OCR
            if GOOGLE_VISION_AVAILABLE and google_client:
                try:
                    pan_text_raw = extract_text_from_image(pan_content)
                    pan_extracted_data = extract_fields(pan_text_raw)
                    documents_data["ocr_results"]["pan"] = pan_extracted_data
                    documents_data["ocr_results"]["raw_text_pan_preview"] = pan_text_raw[:200]
                except Exception as e:
                    print(f"PAN OCR Error: {e}")

        # 5. Insert into MongoDB
        result = await documents_collection.insert_one(documents_data)
        
        # 6. Determine verification status
        # Success if AT LEAST ONE document returned extracted data
        is_verified = (aadhaar_extracted_data) or (pan_extracted_data)
        
        new_status = "verified_by_ocr" if is_verified else "documents_uploaded"
            
        await users_collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "documents_uploaded": True, 
                "status": new_status,
                "verification_data": documents_data["ocr_results"]
            }}
        )
        
        return JSONResponse({
            "message": "Documents processed and stored",
            "document_id": str(result.inserted_id),
            "verification_status": new_status,
            "extracted_data": {
                "aadhaar": aadhaar_extracted_data,
                "pan": pan_extracted_data
            }
        })

    except Exception as e:
        print(f"Error in submit_documents: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/kyc/user/{user_id}")
async def get_user_data(user_id: str):
    """ Retrieves User Data """
    try:
        user = await users_collection.find_one({"user_id": user_id})
        if not user:
            return JSONResponse(status_code=404, content={"error": "User not found"})
        
        user["_id"] = str(user["_id"])
        if "created_at" in user: user["created_at"] = str(user["created_at"])
        if "photograph" in user: del user["photograph"]
        
        return JSONResponse(user)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# TO RUN: uvicorn main:app --reload --port 8000
# ---------------------------