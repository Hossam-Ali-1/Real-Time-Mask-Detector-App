import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- Constants ---
IMG_SIZE = 124
LABELS = {0: 'Mask', 1: 'No Mask'}
COLORS = {'Mask': (0, 255, 0), 'No Mask': (0, 0, 255)}
PREDICTION_THRESHOLD = 0.5

# --- Default Detection Settings ---
DEFAULT_SETTINGS = {
    'GAMMA': 2.5,
    'CONFIDENCE_THRESHOLD': 0.15,
    'BLOB_SIZE': 400
}

# --- Live Camera Detection Settings ---
LIVE_SETTINGS = {
    'GAMMA': 0.8,  # Reduced gamma for better visibility
    'CONFIDENCE_THRESHOLD': 0.5,  # Increased confidence threshold
    'BLOB_SIZE': 300  # Adjusted blob size
}

# --- Load Model (Cached) ---
@st.cache_resource
def load_keras_model():
    try:
        model = load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_keras_model()

# --- Face Detection Setup ---
prototxt_path = "architecture.txt"
caffemodel_path = "weights.caffemodel"
cvNet = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# --- Image Processing Functions ---
def safe_crop(image, startX, startY, endX, endY):
    
    h, w = image.shape[:2]
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(w, endX)
    endY = min(h, endY)
    return image[startY:endY, startX:endX]

def adjust_gamma(image, gamma=1.0):
    
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_face(face):
    
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face_normalized = face_resized.astype("float32") / 255.0
    return face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 3)

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide"
)

# --- Dark Theme Customization ---
st.markdown("""
<style>
    :root {
        --primary-color: #1E88E5;
        --background-color: #121212;
        --secondary-background: #1E1E1E;
        --text-color: #FFFFFF;
        --border-radius: 10px;
    }
    
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        font-weight: bold;
        border-radius: var(--border-radius);
    }
    
    .stRadio>div {
        background-color: var(--secondary-background);
        border-radius: var(--border-radius);
        padding: 10px;
    }
    
    .title {
        color: var(--primary-color);
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    
    .result-box {
        border-radius: var(--border-radius);
        padding: 20px;
        margin: 10px 0;
        background-color: var(--secondary-background);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stImage>img {
        border-radius: var(--border-radius);
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown('<h1 class="title">😷 Real-Time Mask Detector</h1>', unsafe_allow_html=True)

# --- Input Options ---
option = st.radio(
    "Select input method:",
    ("Upload Image", "Live Camera"),
    horizontal=True
)

def detect_masks(image_np, is_live=False):
    
    settings = LIVE_SETTINGS if is_live else DEFAULT_SETTINGS

    image_np = adjust_gamma(image_np, gamma=settings['GAMMA'])
    orig = image_np.copy()
    (h, w) = image_np.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (settings['BLOB_SIZE'], settings['BLOB_SIZE'])), 
                               1.0, 
                               (settings['BLOB_SIZE'], settings['BLOB_SIZE']), 
                               (104.0, 177.0, 123.0))
    
    cvNet.setInput(blob)
    detections = cvNet.forward()
    
    results = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > settings['CONFIDENCE_THRESHOLD']:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            if (endX - startX) < 30 or (endY - startY) < 30:
                continue
                
            face = safe_crop(orig, startX, startY, endX, endY)
            if face.size == 0:
                continue
                
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            if gray.std() < 15:
                continue
            
            try:
                face_input = preprocess_face(face)
                prediction = model.predict(face_input, verbose=0)
                
                # binary classification
                label_Y = int(prediction[0][0] > PREDICTION_THRESHOLD)
                
                label_text = LABELS[label_Y]
                color = COLORS[label_text]
                
                cv2.rectangle(image_np, (startX, startY), (endX, endY), color, 2)
                cv2.putText(image_np, label_text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                
                results.append({
                    'label': label_text,
                    'confidence': float(confidence),
                    'probability': float(prediction[0][0]),
                    'box': [int(startX), int(startY), int(endX), int(endY)]
                })
            except Exception as e:
                st.error(f"Error processing face: {e}")
                continue
    
    return image_np, results

# --- Image Upload Option ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        uploaded_bytes = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(uploaded_bytes), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        with col2:
            with st.spinner('Analyzing image...'):
                processed_image, results = detect_masks(image_np.copy(), is_live=False)
                st.image(processed_image, caption="Detection Results", use_container_width=True)
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("🎯 Detection Results:")
                
                if not results:
                    st.warning("⚠️ No faces detected or confidence too low.")
                else:
                    for result in results:
                        st.write(f"- {result['label']}: Confidence {result['confidence']:.2%}, Probability: {result['probability']:.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)

# --- Live Camera Option ---
elif option == "Live Camera":
    st.warning("Note: For best results, please use Chrome or Edge browser.")
    st.info("Tip: Make sure your face is well-lit and clearly visible for better detection.")

    class MaskDetectorTransformer(VideoTransformerBase):
        def __init__(self):
            self.results = []
            self.frame_count = 0
        
        def transform(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            # Process every frame for better responsiveness
            processed_img, results = detect_masks(img, is_live=True)
            self.results = results
            return processed_img
    
    # Create a placeholder for results
    result_placeholder = st.empty()
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="mask-detector",
        video_transformer_factory=MaskDetectorTransformer,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
    
    # Display results in real-time
    if ctx.video_transformer:
        while True:
            with result_placeholder.container():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("🎯 Live Detection Results:")
                
                if hasattr(ctx.video_transformer, 'results'):
                    if not ctx.video_transformer.results:
                        st.warning("⚠️ No faces detected or confidence too low.")
                    else:
                        for result in ctx.video_transformer.results:
                            st.write(f"- {result['label']}: Confidence {result['confidence']:.2%}, Probability: {result['probability']:.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Small delay to prevent high CPU usage
            time.sleep(0.1)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed By Hossam Ali</p>
    <p>Face Mask Detection AI Model</p>
</div>
""", unsafe_allow_html=True)
