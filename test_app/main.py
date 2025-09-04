from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import numpy as np
import cv2


app = FastAPI(title="Face Recognition API", description="API for face capture, embedding, and verification")

# Allow all CORS for testing purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { font-weight: bold; color: #007bff; }
            .url { font-family: monospace; background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
            .docs-link { text-align: center; margin-top: 30px; }
            .docs-link a { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Face Recognition API</h1>
            <p>Welcome to the Face Recognition API! This API provides endpoints for face capture, embedding generation, and face verification.</p>
            <h2>Available Endpoints:</n2>
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/capture-selfie/</div>
                <p>Upload a selfie image</p>
            </div>
            <div class="endpoint">
                <div class="method">GET</div>
                <div class="url">/embedding-from-selfie/</div>
                <p>Generate face embedding from the saved selfie</p>
            </div>
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/verify-face/</div>
                <p>Upload a reference image and verify it against the saved selfie</p>
            </div>
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/predict-age-gender/</div>
                <p>Predict age and gender from an image</p>
            </div>
            <div class="docs-link">
                <a href="/docs">📖 View Interactive API Documentation</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.post("/capture-selfie/")
async def capture_selfie(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(os.getcwd(), "selfie.jpg")
        with open(file_location, "wb") as f:
            f.write(await file.read())
        return {
            "message": "Selfie uploaded successfully",
            "filename": file.filename,
            "saved_as": "selfie.jpg",
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/embedding-from-selfie/")
def get_embedding():
    if not os.path.exists("selfie.jpg"):
        return JSONResponse(status_code=404, content={"error": "No selfie found. Please capture a selfie first."})
    try:
        # Placeholder: replace with real embedding logic
        dummy_embedding = np.random.rand(128).tolist()
        return {"embedding": dummy_embedding, "message": "Face embedding generated successfully"}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/verify-face/")
async def verify_face(reference_img: UploadFile = File(...)):
    try:
        with open("reference.jpg", "wb") as buffer:
            shutil.copyfileobj(reference_img.file, buffer)

        if not os.path.exists("selfie.jpg"):
            return JSONResponse(status_code=404, content={"error": "No selfie found. Please capture a selfie first."})

        # Placeholder matching
        similarity = float(np.random.uniform(0.3, 0.9))
        result = "Face Verified: Match" if similarity > 0.6 else "Face Mismatch: Verification Failed"
        return {
            "similarity": round(similarity, 4),
            "result": result,
            "message": "Face verification completed",
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        if os.path.exists("reference.jpg"):
            os.remove("reference.jpg")


# ------- Optional: Age/Gender prediction with Caffe models (only if files exist) -------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "age_gender", "models")

AGE_PROTO = os.path.join(MODELS_DIR, "deploy_age.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODELS_DIR, "deploy_gender.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


def models_available() -> bool:
    return all(os.path.exists(p) for p in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL])


if models_available():
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

    @app.post("/predict-age-gender/")
    async def predict_age_gender(file: UploadFile = File(...)):
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        blob = cv2.dnn.blobFromImage(
            img, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False
        )

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[int(gender_preds[0].argmax())]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[int(age_preds[0].argmax())]

        return {"age": age, "gender": gender}
else:
    @app.post("/predict-age-gender/")
    async def predict_age_gender_unavailable(_: UploadFile = File(...)):
        return JSONResponse(
            status_code=503,
            content={
                "error": "Age/Gender models not found.",
                "expected_paths": [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL],
            },
        )


