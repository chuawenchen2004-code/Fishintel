from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2, shutil, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
model = YOLO("yolov8n.pt")

def estimate_size(bbox, img_w, img_h):
    x1,y1,x2,y2 = bbox
    area = (x2-x1)*(y2-y1)
    ratio = area/(img_w*img_h)
    if ratio < 0.05: return "Small (~15cm)"
    elif ratio < 0.15: return "Medium (~30cm)"
    else: return "Large (~50cm+)"

@app.post("/detect")
async def detect_fish(file: UploadFile = File(...)):
    temp_path="temp.jpg"; result_path="result.jpg"
    with open(temp_path,"wb") as b: shutil.copyfileobj(file.file,b)
    results=model(temp_path,conf=0.25,verbose=False)
    img=cv2.imread(temp_path); h,w=img.shape[:2]
    detections=[]; fish_count=0
    for r in results:
        for box in r.boxes:
            fish_count+=1; conf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0])
            species="Fish"; size_label=estimate_size((x1,y1,x2,y2),w,h)
            detections.append({"label":species,"species":species,"confidence":conf,"size":size_label,"bbox":[x1,y1,x2,y2]})
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,220,255),3)
            cv2.putText(img,f"{species} | {size_label} {conf:.0%}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,255),2)
    cv2.imwrite(result_path,img)
    return {"fish_detected":fish_count,"total_count":fish_count,"status":"Success","result_image":result_path,"detections":detections,"model_used":"Fish Model","message":f"{fish_count} fish"}

@app.get("/result-image")
def get_result_image():
    if os.path.exists("result.jpg"): return FileResponse("result.jpg")
    return {"error":"no result yet"}

@app.get("/")
def home():
    if os.path.exists("index.html"): return FileResponse("index.html")
    return {"ok":True}