from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os
import torch

app = Flask(__name__)

MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)
SPECIES = list(model.names.values())
print(f"✅ Model loaded: {len(SPECIES)} species -> {SPECIES}")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === FISH FACTS DATABASE ===
FISH_FACTS = {
    'Baung': {'scientific': 'Mystus nemurus', 'habitat': 'Rivers & lowland streams in Malaysia', 'diet': 'Carnivore - small fish, shrimp, insects', 'length': 'Up to 60 cm', 'fun': 'Has sharp pectoral spines that can cause painful sting!'},
    'Bujuk': {'scientific': 'Channa lucius', 'habitat': 'Swamps, peat forests & slow blackwater streams', 'diet': 'Carnivore - fish & crustaceans', 'length': 'Up to 40 cm', 'fun': 'Shyest snakehead, known as the forest snakehead.'},
    'Haruan': {'scientific': 'Channa striata', 'habitat': 'Paddy fields, swamps, ponds & rivers', 'diet': 'Carnivore - fish, frogs, insects', 'length': 'Up to 90 cm', 'fun': 'Used in traditional medicine - its extract helps wound healing!'},
    'Keli': {'scientific': 'Clarias batrachus', 'habitat': 'Muddy ponds, swamps & rice fields, can breathe air', 'diet': 'Omnivore - detritus, insects, small fish', 'length': 'Up to 50 cm', 'fun': 'Can walk on land using pectoral fins to find new ponds!'},
    'Kerandang': {'scientific': 'Channa marulioides', 'habitat': 'Forest streams & clear jungle rivers', 'diet': 'Carnivore - aggressive predator', 'length': 'Up to 65 cm', 'fun': 'One of the most beautiful snakeheads with orange spots!'},
    'Lampam': {'scientific': 'Barbonymus schwanenfeldii', 'habitat': 'Large rivers & reservoirs', 'diet': 'Omnivore - plants, algae, insects', 'length': 'Up to 34 cm', 'fun': 'Diamond-shaped body with red fins, very popular sport fish!'},
    'Patin': {'scientific': 'Pangasius sp.', 'habitat': 'Large rivers like Pahang & Mekong', 'diet': 'Omnivore - fish, fruits, plants', 'length': 'Up to 130 cm', 'fun': 'Can grow huge! Temerloh is famous as Bandar Ikan Patin.'},
    'Puyu': {'scientific': 'Anabas testudineus', 'habitat': 'Swamps, ditches & paddy fields, survives drought', 'diet': 'Omnivore - insects, larvae, vegetation', 'length': 'Up to 25 cm', 'fun': 'Climbing perch - can climb trees and live out of water for 6 days!'},
    'Sebarau': {'scientific': 'Hampala macrolepidota', 'habitat': 'Fast-flowing clear rivers & rapids', 'diet': 'Carnivore - aggressive hunter', 'length': 'Up to 70 cm', 'fun': 'Called Jungle Perch - famous for explosive jumps when hooked!'},
    'Siakap': {'scientific': 'Lates calcarifer', 'habitat': 'Estuaries, mangroves & coastal rivers', 'diet': 'Carnivore - fish & shrimp', 'length': 'Up to 200 cm', 'fun': 'Changes sex! Starts as male, then becomes female when bigger.'},
    'Toman': {'scientific': 'Channa micropeltes', 'habitat': 'Large rivers, lakes & reservoirs', 'diet': 'Carnivore - apex predator', 'length': 'Up to 130 cm', 'fun': 'Largest snakehead in Malaysia - mother guards babies in red ball!'}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None
    top3 = None
    fish_info = None
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            image_path = filepath

            results = model(filepath)
            probs = results[0].probs
            
            pred_idx = int(probs.top1)
            pred_name = model.names[pred_idx]
            conf = float(probs.top1conf) * 100

            top3_idx = torch.topk(probs.data, 3).indices.tolist()
            top3_conf = torch.topk(probs.data, 3).values.tolist()
            top3_list = [{'name': model.names[idx], 'conf': float(c)*100} for idx, c in zip(top3_idx, top3_conf)]

            prediction = pred_name
            confidence = conf
            top3 = top3_list
            fish_info = FISH_FACTS.get(pred_name)

    return render_template('index.html', prediction=prediction, confidence=confidence, image_path=image_path, top3=top3, fish_info=fish_info, species_list=SPECIES, num_species=len(SPECIES), model_path=MODEL_PATH)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)