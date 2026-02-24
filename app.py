"""
🐘 Elephant Detector - Streamlit Web App
Complete working version for deployment
"""

import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np

# Set Ultralytics config directory to temp (fixes permission issues)
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.makedirs("/tmp/ultralytics", exist_ok=True)

# Import YOLO after setting config
from ultralytics import YOLO
import cv2

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
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
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

@st.cache_resource
def load_model():
    """Load YOLO model with error handling"""
    model_path = "elephant_detector_final.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please ensure the model file is in the correct location")
        return None
    
    try:
        with st.spinner("🔄 Loading model..."):
            model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
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
    st.info("""
    - 🐘 Elephant accuracy: 94.6%
    - 🎯 Total classes: 4
    - 📸 Training images: 1,049
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
    
    # Read and process image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV (YOLO expects BGR)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run detection
    with st.spinner("🔍 Detecting animals..."):
        results = model(
            img_array, 
            conf=confidence,
            classes=class_indices if class_indices else None
        )
    
    # Display results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        # FIXED: use_column_width instead of use_container_width
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎯 Detected Animals")
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        # FIXED: use_column_width instead of use_container_width
        st.image(annotated_rgb, use_column_width=True)
    
    # Show detection details
    if len(results[0].boxes) > 0:
        st.markdown("### 📊 Detection Results")
        
        # Create a nice table of detections
        detection_data = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detection_data.append({
                "Animal": class_name,
                "Confidence": f"{conf:.1%}",
                "Location": f"({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})",
                "🐘": "✅" if class_name == "elephant" else "❌"
            })
        
        # Display as dataframe
        st.dataframe(detection_data, use_container_width=True)
        
        # Count elephants
        elephant_count = sum(1 for d in detection_data if d["Animal"] == "elephant")
        if elephant_count > 0:
            st.balloons()
            st.success(f"🎉 Found {elephant_count} elephant(s)!")
        else:
            st.info("No elephants detected in this image")
    else:
        st.warning("⚠️ No animals detected. Try lowering the confidence threshold.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; padding: 1rem;">
        <p>🐘 Built with Streamlit and Ultralytics YOLOv11 | 
        <a href="https://github.com/PRASHYA20/team-212" target="_blank">GitHub Repository</a></p>
    </div>
""", unsafe_allow_html=True)
