# 😷 Real-Time-Mask-Detector-App | Face Mask Detection AI Model

🔗 **Try the live app:** [real-time-mask-detector.streamlit.app](https://real-time-mask-detector.streamlit.app/)

A real-time face mask detection web app built with **Streamlit**, **OpenCV**, **TensorFlow/Keras**, and **streamlit-webrtc**.  
The app accesses your camera, runs on-device inference, and overlays predictions (Mask / No Mask) directly on the video stream.

---

## 🚀 Project Overview

This project demonstrates a **browser-based, real-time** mask detection pipeline:

- Capture live video via WebRTC
- Detect faces and run a CNN model for mask/no-mask classification
- Render predictions with bounding boxes and labels on the stream
- Simple, user-friendly Streamlit UI with dark theme

---

## 🎯 Key Features

- **Real-time inference** with `streamlit-webrtc`
- **On-device processing** (no server upload for inference)
- **Two input modes**: image upload and live camera
- **Adjustable detection settings** for live vs. static images
- **Included model artifacts** for out-of-the-box runs

---

## 🛠️ Tech Stack

- **Python 3.10+**
- Streamlit `1.47.0`
- streamlit-webrtc `0.63.3`
- OpenCV (headless) `4.12.0.88`
- TensorFlow `2.18.0` + Keras `3.8.0`
- NumPy, Pillow

---

## 📂 Project Structure

```plaintext
📂 Real-Time-Mask-Detector-App/
 ├── app.py               # Main Streamlit app with WebRTC + inference pipeline
 ├── requirements.txt     # Python dependencies (pinned)
 ├── model.keras          # Trained Keras model used for classification
 ├── weights.caffemodel   # Face detector weights (OpenCV DNN)
 ├── architecture.txt     # Face detector prototxt / model arch config
 └── README.md            # Project documentation
