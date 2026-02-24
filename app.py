"""
🐘 Elephant Detector - Streamlit Web App
Complete working version with RGBA image fix
"""

import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import torch

# Page configuration
st.set_page_config(
    page_title="🐘 Elephant Detector",
    page_icon="🐘",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🐘 Elephant Detector</h1>
        <p>Detects elephants and African wildlife using YOLOv11</p>
    </div>
""", unsafe_allow_html=True)

# Set Ultralytics config directory to temp (fixes permission issues)
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.makedirs("/tmp/ultralytics", exist_ok=True)

@st.cache_resource
def load_model():
    """Load YOLO model with error handling"""
    model_path = "elephant_detector_final.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Files in directory: " + ", ".join(os.listdir('.')))
        return None
    
    # Check file size
    file_size = os.path.getsize(model_path) / (1024*1024)
    
    try:
        with st.spinner("🔄 Loading model..."):
            # Load model
            model = YOLO(model_path)
            
            # Move to CPU if CUDA not available
            if not torch.cuda.is_available():
                model.to('cpu')
            
            # Test with dummy image to verify model works
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_result = model(dummy_img, verbose=False)
            
        st.success(f"✅ Model loaded successfully! (Size: {file_size:.2f} MB)")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Confidence threshold slider
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Class selection
    class_names = ['buffalo', 'elephant', 'rhino', 'zebra']
    selected_classes = st.multiselect(
        "Animals to detect",
        options=class_names,
        default=['elephant'],
        help="Select which animal classes to detect"
    )
    
    # Convert selected class names to indices
    if selected_classes:
        class_indices = [class_names.index(c) for c in selected_classes]
    else:
        class_indices = None
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info(f"""
    - 🐘 Elephant accuracy: 94.6%
    - 🎯 Total classes: 4
    - 📸 Training images: 1,049
    - 📁 Model size: {os.path.getsize('elephant_detector_final.pt')/(1024*1024):.2f} MB
    """)
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Upload an image (JPG, PNG)
    2. Adjust confidence threshold
    3. Select animals to detect
    4. View results
    """)

# Main content
st.markdown("### 📸 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image containing elephants or other wildlife"
)

if uploaded_file is not None:
    # Display file info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / 1024:.1f} KB"
    }
    st.write(file_details)
    
    try:
        # Read and process image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # ========== FIXED: Handle RGBA images (4 channels) ==========
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA image
                # Convert RGBA to RGB by removing alpha channel
                img_array = img_array[..., :3]
            elif img_array.shape[2] != 3:
                st.error(f"❌ Unsupported number of channels: {img_array.shape[2]}. Expected 3 (RGB) or 4 (RGBA).")
                st.stop()
        elif len(img_array.shape) == 2:  # Grayscale image
            # Convert grayscale to RGB by duplicating channels
            img_array = np.stack([img_array] * 3, axis=-1)
        else:
            st.error(f"❌ Unexpected image shape: {img_array.shape}")
            st.stop()
        
        # Convert RGB to BGR for OpenCV (YOLO expects BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # ========== OPTIONAL: Resize if image is too small ==========
        h, w = img_array.shape[:2]
        min_dim = 320  # YOLO works with various sizes, but very small images might need resizing
        if h < min_dim or w < min_dim:
            scale = max(min_dim/h, min_dim/w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_array = cv2.resize(img_array, (new_w, new_h))
        
        # Run detection
        with st.spinner("🔍 Detecting animals..."):
            results = model(
                img_array, 
                conf=confidence,
                classes=class_indices if class_indices else None,
                verbose=False
            )
        
        # Display results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            # Convert BGR back to RGB for display
            display_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            st.image(display_img, use_column_width=True)
        
        with col2:
            st.subheader("🎯 Detected Animals")
            if len(results) > 0 and len(results[0].boxes) > 0:
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True)
            else:
                st.info("No animals detected")
                display_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(display_img, use_column_width=True)
        
        # Show detection details
        if len(results) > 0 and len(results[0].boxes) > 0:
            st.markdown("### 📊 Detection Results")
            
            # Create a nice table of detections
            detection_data = []
            elephant_count = 0
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection_data.append({
                    "Animal": class_name,
                    "Confidence": f"{conf:.1%}",
                    "Location": f"({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})"
                })
                
                if class_name == "elephant":
                    elephant_count += 1
            
            # Display as dataframe
            st.dataframe(detection_data, use_container_width=True)
            
            # Celebration for elephants
            if elephant_count > 0:
                st.balloons()
                st.markdown(f"""
                    <div class="success-box">
                        <h3>🐘 Found {elephant_count} elephant(s)!</h3>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="info-box">
                        <p>No elephants detected in this image</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No animals detected in this image")
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; padding: 1rem;">
        <p>🐘 Built with Streamlit and Ultralytics YOLOv11 | 
        <a href="https://github.com/PRASHYA20/team-212" target="_blank">GitHub Repository</a></p>
    </div>
""", unsafe_allow_html=True)
