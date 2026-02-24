"""
🐘 Elephant Detector - Streamlit Web App
Debug version to identify model issues
"""

import streamlit as st
import os
import sys
import traceback
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="🐘 Elephant Detector",
    page_icon="🐘",
    layout="wide"
)

st.title("🐘 Elephant Detector - Debug Mode")
st.write("Python version:", sys.version)

# Check if torch is available
try:
    import torch
    st.write("✅ PyTorch version:", torch.__version__)
    st.write("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        st.write("CUDA device:", torch.cuda.get_device_name(0))
except Exception as e:
    st.error(f"❌ PyTorch import error: {e}")

# Check if ultralytics is available
try:
    from ultralytics import YOLO
    st.write("✅ Ultralytics imported successfully")
except Exception as e:
    st.error(f"❌ Ultralytics import error: {e}")
    st.stop()

# Set environment variable for config
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.makedirs("/tmp/ultralytics", exist_ok=True)

@st.cache_resource
def load_model():
    """Load YOLO model with extensive debugging"""
    model_path = "elephant_detector_final.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Files in directory: " + ", ".join(os.listdir('.')))
        return None
    
    # Check file size
    file_size = os.path.getsize(model_path) / (1024*1024)
    st.write(f"📁 Model file size: {file_size:.2f} MB")
    
    try:
        with st.spinner("🔄 Loading model..."):
            # Try loading with different options
            st.write("Attempting to load model...")
            
            # Method 1: Standard loading
            try:
                model = YOLO(model_path)
                st.write("✅ Model loaded with standard method")
                
                # Test with dummy image
                st.write("Testing model with dummy image...")
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                test_result = model(dummy_img, verbose=False)
                st.write(f"✅ Test successful! Model output type: {type(test_result)}")
                return model
            except Exception as e1:
                st.error(f"❌ Standard loading failed: {e1}")
                
                # Method 2: Load with specific task
                try:
                    st.write("Attempting to load with task='detect'...")
                    model = YOLO(model_path, task='detect')
                    
                    # Test
                    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    test_result = model(dummy_img, verbose=False)
                    st.write("✅ Model loaded with task='detect'")
                    return model
                except Exception as e2:
                    st.error(f"❌ Task-specific loading failed: {e2}")
                    
                    # Method 3: Load with CPU only
                    try:
                        st.write("Attempting to load with CPU...")
                        import torch
                        model = YOLO(model_path)
                        model.to('cpu')
                        
                        # Test
                        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                        test_result = model(dummy_img, verbose=False)
                        st.write("✅ Model loaded on CPU")
                        return model
                    except Exception as e3:
                        st.error(f"❌ CPU loading failed: {e3}")
                        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.code(traceback.format_exc())
        return None

# Load model
model = load_model()

if model is None:
    st.error("❌ Could not load model. Please check the model file.")
    st.stop()

st.success("✅ Model loaded and ready!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)
    
    class_names = ['buffalo', 'elephant', 'rhino', 'zebra']
    selected_classes = st.multiselect("Animals to detect", class_names, default=['elephant'])
    class_indices = [class_names.index(c) for c in selected_classes] if selected_classes else None
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info(f"""
    - 🐘 Elephant accuracy: 94.6%
    - 🎯 Total classes: 4
    - 📸 Training images: 1,049
    - 📁 Model size: {os.path.getsize('elephant_detector_final.pt')/(1024*1024):.2f} MB
    """)

# Main content
st.markdown("### 📸 Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

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
        st.write(f"📷 Image shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Convert RGB to BGR for OpenCV (YOLO expects BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            import cv2
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            st.write("✅ Converted RGB to BGR")
        
        # Run detection with error catching
        st.write("🔍 Running detection...")
        try:
            results = model(
                img_array, 
                conf=confidence,
                classes=class_indices if class_indices else None,
                verbose=True  # Enable verbose output
            )
            st.write(f"✅ Detection complete! Results type: {type(results)}")
            st.write(f"Number of results: {len(results)}")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🎯 Detected Animals")
                if len(results) > 0 and len(results[0].boxes) > 0:
                    annotated = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_column_width=True)
                    
                    # Show detection details
                    st.write(f"Found {len(results[0].boxes)} objects:")
                    for i, box in enumerate(results[0].boxes):
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                        st.write(f"{i+1}. **{class_name}** - {conf:.1%} confidence")
                else:
                    st.info("No objects detected")
                    st.image(image, use_column_width=True)
        except Exception as e:
            st.error(f"❌ Detection error: {e}")
            st.code(traceback.format_exc())
            
    except Exception as e:
        st.error(f"❌ Image processing error: {e}")
        st.code(traceback.format_exc())

st.markdown("---")
st.write("Debug session complete")
