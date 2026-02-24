# 🐘 Elephant Detector Model

A YOLOv11 model trained to detect elephants and other African wildlife.

## Model Performance
- Elephant detection accuracy: **94.6% mAP50**
- Trained on 1,049 images with 748 elephant instances

## Files
- `elephant_detector_final.pt` - The trained model
- `webcam_elephant_detector.py` - Real-time webcam detection script

## Usage
```bash
# Run webcam detection
python3 webcam_elephant_detector.py
