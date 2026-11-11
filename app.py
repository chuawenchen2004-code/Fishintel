from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import shutil, os

app = FastAPI(title="Fishintel - UMT Fish Species Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Try load custom fish model, fallback to general
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
model = YOLO(MODEL_PATH)

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.post("/detect")
async def detect_fish(file: UploadFile = File(...)):
    temp_path = "temp.jpg"
    result_path = "result.jpg"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results = model(temp_path)
    
    detections = []
    for r in results:
        r.save(filename=result_path)
        for box in r.boxes:
            cls_id = int(box.cls[0])
            species = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"species": species, "confidence": round(conf, 2)})

    return {
        "model_used": MODEL_PATH,
        "fish_detected": len(detections),
        "detections": detections,
        "result_image_url": "/result-image"
    }

@app.get("/result-image")
def get_result_image():
    return FileResponse("result.jpg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)