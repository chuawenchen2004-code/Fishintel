# UMT Fish Detector - 11 Species Classification
**StudentID_Topic: [YourID]_FishClassification**
**UMT - Fish Species Classification using YOLOv8n-cls**

### Project Overview
An AI web application that classifies 11 Malaysian freshwater fish species using deep learning. Built with YOLOv8n-cls (transfer learning) and Flask.

**Species (11):** Baung, Bujuk, Haruan, Keli, Kerandang, Lampam, Patin, Puyu, Sebarau, Siakap, Toman

### Model Performance
- **Model:** YOLOv8n-cls (pretrained on ImageNet, transfer learning)
- **Training:** 660 images (20 train / 20 val / 20 test per class)
- **Accuracy:** 62.27% Top-1, 89.1% Top-5
- **Best classes:** Sebarau & Siakap (90% accuracy)
- **Hardest:** Keli (20%) confused with Baung - catfish family similarity
- **Epochs:** 100, Image size: 224, Optimizer: Adam, Batch: 16

### Preprocessing Steps
1. Images resized to 224x224
2. Normalized (ImageNet mean/std)
3. Auto-augmentation by YOLO (rotation, flip, color jitter)
4. Folder structure: dataset/train/{class}, dataset/val/{class}, dataset/test/{class}

### Project Structure