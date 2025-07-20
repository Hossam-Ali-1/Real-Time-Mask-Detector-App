import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'object_detection'))
sys.path.append(os.path.join(os.getcwd(), 'slim'))
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

# --- Path Setup ---
WORKSPACE_PATH = 'Tensorflow'
MODEL_PATH = os.path.join(WORKSPACE_PATH, 'models', 'my_ssd_mobnet', 'exported-model', 'saved_model')
LABELMAP_PATH = os.path.join(WORKSPACE_PATH, 'annotations', 'label_map.pbtxt')

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    print("🔄 Loading model...")
    detect_fn = tf.saved_model.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    return detect_fn

@st.cache_resource
def load_label_map():
    return label_map_util.create_category_index_from_labelmap(LABELMAP_PATH)

detect_fn = load_model()
category_index = load_label_map()

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
    
    .css-1aumxhk {
        background-color: var(--secondary-background);
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

# --- Image Processing Function ---
def process_image(image_np):
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
    
    # Run detection
    detections = detect_fn.signatures['serving_default'](input_tensor)
    
    # Extract results
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(np.int32)
    scores = detections['detection_scores'].numpy()[0]
    
    # Visualize detections
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.5,
        line_thickness=3,
        agnostic_mode=False
    )
    
    return image_np_with_detections, classes, scores

# --- Image Upload Option ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        with col2:
            with st.spinner('Analyzing image...'):
                processed_image, classes, scores = process_image(image_np)
                st.image(processed_image, caption="Detection Results", use_container_width=True)
                
                # Display text results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("🎯 Detection Results:")
                
                detected_objects = {}
                for i in range(len(scores)):
                    if scores[i] > 0.5:
                        class_name = category_index[classes[i]]['name']
                        score_percent = int(scores[i] * 100)
                        
                        if class_name not in detected_objects or scores[i] > detected_objects[class_name]['score']:
                            detected_objects[class_name] = {
                                'score': scores[i],
                                'count': detected_objects.get(class_name, {}).get('count', 0) + 1
                            }
                
                if not detected_objects:
                    st.warning("⚠️ No confident detections found.")
                else:
                    for obj, data in detected_objects.items():
                        st.write(f"- {obj}: {int(data['score']*100)}% confidence ({data['count']} detected)")
                
                st.markdown('</div>', unsafe_allow_html=True)

# --- Live Camera Option ---
elif option == "Live Camera":
    st.warning("Note: For best results in Streamlit Cloud, please use Chrome or Edge browser.")
    
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            processed_frame, classes, scores = process_image(img_rgb)
            
            # Convert back to BGR for display
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            # Store results for display
            self.detected_objects = {}
            for i in range(len(scores)):
                if scores[i] > 0.5:
                    class_name = category_index[classes[i]]['name']
                    if class_name not in self.detected_objects or scores[i] > self.detected_objects[class_name]['score']:
                        self.detected_objects[class_name] = {
                            'score': scores[i],
                            'count': self.detected_objects.get(class_name, {}).get('count', 0) + 1
                        }
            
            return processed_frame_bgr
    
    # Create a placeholder for results
    result_placeholder = st.empty()
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
    
    # Display results
    if ctx.video_transformer:
        with result_placeholder.container():
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("🎯 Live Detection Results:")
            
            if hasattr(ctx.video_transformer, 'detected_objects'):
                if not ctx.video_transformer.detected_objects:
                    st.warning("⚠️ No confident detections found.")
                else:
                    for obj, data in ctx.video_transformer.detected_objects.items():
                        st.write(f"- {obj}: {int(data['score']*100)}% confidence ({data['count']} detected)")
            
            st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed By Hossam Ali</p>
    <p>Face Mask Detection AI Model</p>
</div>
""", unsafe_allow_html=True)
