import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# sys.path handled locally in functions or before imports if needed
# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==================== CONSTANTS ====================
IMAGE_SIZE = (224, 224)
MODEL_PATH = Path(__file__).parent.parent / "saved_models" / "vgg16_brain_tumor_4class.h5"

# Class configuration
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": ("Glioma Tumor", "#E53935", True),
    "meningioma": ("Meningioma Tumor", "#FB8C00", True),
    "notumor": ("No Tumor", "#43A047", False),
    "pituitary": ("Pituitary Tumor", "#1E88E5", True)
}


# ==================== BRAIN CROPPING ====================
def crop_brain_region(image: np.ndarray) -> tuple:
    """
    Extract brain region from MRI image using CV (using centralized package logic).
    """
    try:
        from src.preprocessing.brain_cropper import BrainCropper
        cropper = BrainCropper(target_size=IMAGE_SIZE)
        return cropper.crop_brain_region(image)
    except Exception as e:
        st.error(f"Import Error in crop_brain_region: {e}")
        return cv2.resize(image, IMAGE_SIZE), False


# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    try:
        import tensorflow as tf
        if MODEL_PATH.exists():
            model = tf.keras.models.load_model(str(MODEL_PATH))
            return model, None
        else:
            return None, f"Model not found at: {MODEL_PATH}"
    except Exception as e:
        return None, str(e)


# ==================== MAIN APP ====================
def main():
    # Ensure current directory is in path for src imports
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
        
    # Header - Using h1 with inline style for maximum size
    st.markdown('''
        <h1 style="font-size: 4rem; font-weight: bold; text-align: center; color: #1E88E5; margin-bottom: 0.5rem;">
            🧠 Brain Tumor Detection
        </h1>
    ''', unsafe_allow_html=True)
    st.markdown('''
        <p style="font-size: 1.5rem; text-align: center; color: #666; margin-bottom: 2rem;">
            AI-powered MRI analysis using VGG-16 Deep Learning
        </p>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        # st.success("✅ Model: Trained & Fine-Tuned")
        st.markdown("""
        This tool uses a **VGG-16** deep learning model trained on brain MRI images 
        to detect the presence of tumors.
        
        **How it works:**
        1. Upload an MRI scan image
        2. The system extracts the brain region
        3. AI analyzes the image
        4. Prediction with confidence is displayed
        
        **Model Details:**
        - Architecture: VGG-16 (Transfer Learning)
        - Input Size: 224×224 pixels
        - Training: Two-phase (frozen + fine-tuned)
        """)
        
        st.header("📊 Options")
        show_gradcam = st.checkbox("Show Model Explainability (Grad-CAM)", value=True, help="Visualizes which areas of the brain the AI is focusing on.")
        
        st.header("📊 Sample Results")
        st.info("Upload an MRI image to see analysis results.")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload MRI Image")
        
        uploaded_file = st.file_uploader(
            "Choose an MRI scan image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a brain MRI scan in JPG, PNG, or BMP format"
        )
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("❌ Could not read the uploaded image. Please try another file.")
                return
            
            # Display original image
            st.subheader("Original Image")
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Uploaded MRI Scan",
                use_container_width=True
            )
            
            # Process image
            with st.spinner("🔍 Extracting brain region..."):
                cropped, success = crop_brain_region(image)
            
            if success:
                st.success("✅ Brain region extracted successfully!")
            else:
                st.warning("⚠️ Could not detect brain contour. Using original image.")
    
    with col2:
        if uploaded_file is not None:
            st.header("🔬 Analysis Results")
            
            # Display cropped image
            st.subheader("Processed Brain Region")
            st.image(
                cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                caption="Extracted Brain Region (224×224)",
                use_container_width=True
            )
            
            # Load model and predict
            model, error = load_model()
            
            if error:
                st.error(f"❌ Model Error: {error}")
                st.info("Please ensure the model file exists at: " + str(MODEL_PATH))
                return
            
            if model is not None:
                with st.spinner("🧠 Analyzing with AI..."):
                    from tensorflow.keras.applications.vgg16 import preprocess_input
                    # Preprocess for model
                    img_array = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array.astype(np.float32))
                    
                    # Predict (4-class softmax output)
                    predictions = model.predict(img_array, verbose=0)[0]
                
                # Get prediction results
                predicted_idx = int(np.argmax(predictions))
                predicted_class = CLASSES[predicted_idx]
                confidence = float(predictions[predicted_idx])
                
                display_name, color, is_tumor = CLASS_DISPLAY[predicted_class]
                
                # Display result
                st.subheader("Prediction")
                
                bg_color = "#FFEBEE" if is_tumor else "#E8F5E9"
                st.markdown(f'''
                <div style="padding: 2rem; border-radius: 10px; text-align: center; 
                            background-color: {bg_color}; border: 2px solid {color};">
                    <h2 style="color: {color}; margin: 0;">{display_name}</h2>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                        Confidence: <strong>{confidence * 100:.1f}%</strong>
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Confidence bar
                st.progress(confidence)
                
                # Show all class probabilities
                st.subheader("All Probabilities")
                for i, cls in enumerate(CLASSES):
                    prob = float(predictions[i])
                    display, clr, _ = CLASS_DISPLAY[cls]
                    st.markdown(f"**{display.replace('⚠️ ', '').replace('✅ ', '')}**: {prob*100:.1f}%")
                    st.progress(prob)
                
                # Grad-CAM Visualization
                if show_gradcam:
                    st.markdown("---")
                    st.subheader("💡 Model Explainability (Grad-CAM)")
                    with st.spinner("Generating heatmap..."):
                        try:
                            from src.utils.explainability import generate_gradcam, overlay_heatmap
                            # Generate heatmap
                            heatmap = generate_gradcam(model, img_array)
                            
                            # Original image for overlay (RGB)
                            img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                            
                            # Create overlay
                            gradcam_img = overlay_heatmap(img_rgb, heatmap, alpha=0.5)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(heatmap, caption="Activation Heatmap", use_container_width=True, clamp=True)
                            with col_b:
                                st.image(gradcam_img, caption="Focus Area Overlay", use_container_width=True)
                                
                            st.info("""
                            **What does this show?**
                            The red/yellow areas indicate where the AI model is focusing most of its 
                            attention to make the prediction. In a medical context, this should 
                            ideally align with the location of the tumor.
                            """)
                        except Exception as e:
                            st.error(f"Could not generate Grad-CAM: {e}")
                            st.info("Grad-CAM requires the model to have convolutional layers matching 'block5_conv3'.")
            else:
                st.warning("⚠️ Model not loaded. Train the model first.")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Built by <a href="https://github.com/moiz-mansoori">Moiz Mansoori</a> ❤️ using TensorFlow, OpenCV, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Final footer spacing
    st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
