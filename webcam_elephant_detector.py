#!/usr/bin/env python3
"""
Webcam Elephant Detector
"""

from ultralytics import YOLO
import cv2
import os

# Load model
model_path = "elephant_detector_final.pt"
if not os.path.exists(model_path):
    print(f"❌ Model not found! Please make sure {model_path} is in this folder")
    exit(1)

print("🐘 Loading elephant detector...")
model = YOLO(model_path)
print("✅ Model loaded! Opening webcam...")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit(1)

print("\n🎯 Press 'q' to quit, 's' to save screenshot")
print("=" * 40)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame)
    annotated = results[0].plot()
    
    # Show FPS
    cv2.putText(annotated, "Press Q to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Elephant Detector', annotated)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("elephant_capture.jpg", annotated)
        print("📸 Screenshot saved!")

cap.release()
cv2.destroyAllWindows()
print("👋 Done!")
