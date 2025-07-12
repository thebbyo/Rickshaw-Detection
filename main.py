from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
from typing import List
from rickshaw_detection import RickshawDetector

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

detector = RickshawDetector()
model_path = os.path.join(detector.data_dir, 'runs', 'rickshaw_detection5', 'weights', 'best.pt')
if os.path.exists(model_path):
    detector.load_model(model_path)
else:
    detector = None

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]

class DetectResponse(BaseModel):
    results: List[DetectionResult]
    image_url: str

@app.post("/detect", response_model=DetectResponse)
async def detect_rickshaw(file: UploadFile = File(...)):
    if detector is None:
        return {"results": [], "image_url": ""}
    try:
        filename = file.filename
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Run detection
        result_filename = f"result_{filename}"
        output_path = os.path.join(UPLOAD_DIR, result_filename)
        detections = detector.visualize_detection(file_path, save_path=output_path, conf_threshold=0.2)
        results = [
            DetectionResult(
                class_name=det["class"],
                confidence=det["confidence"],
                bbox=list(det["bbox"]),
            ) for det in detections
        ]
        # Return a URL that the frontend can access
        image_url = f"/uploads/{result_filename}"
        return {"results": results, "image_url": image_url}
    except Exception as e:
        print(f"Detection error: {e}")
        return {"results": [], "image_url": ""}

@app.get("/frontend")
def get_frontend():
    return FileResponse("frontend.html")

@app.get("/")
def root():
    return {"message": "Rickshaw Detection API is running."}
