from fastapi import FastAPI, File, UploadFile
import cv2, numpy as np
from ultralytics import YOLO

app = FastAPI(title="FishIntel - Aznan Model")
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"status": "Aznan model ready"}

@app.post("/detect")
async def detect_fish(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    count = len(results[0].boxes)
    return {"fish_count": count}
s