# 😷 Real-Time Face Mask Detection | Streamlit + WebRTC

🔗 **Try the live app:** [real-time-mask-detector.streamlit.app](https://real-time-mask-detector.streamlit.app/)

A real-time face mask detection web app built with **Streamlit**, **OpenCV**, **TensorFlow/Keras**, and **streamlit-webrtc**.  
The app accesses your camera, runs on-device inference, and overlays predictions (Mask / No Mask) directly on the video stream.

---

## 🚀 Project Overview

This project demonstrates a **browser-based, real-time** mask detection pipeline:

- Capture live video via WebRTC
- Detect faces and run a CNN model for mask/no-mask classification
- Render predictions with bounding boxes and labels on the stream
- All through a simple, user-friendly Streamlit UI

---

## 🎯 Key Features

- **Real-time inference** in the browser (WebRTC)
- **On-device processing** — no data is uploaded to a server for inference
- **Lightweight UI** with start/stop controls
- **Model files included** for easy local runs (`model.keras`, `weights.caffemodel`, `architecture.txt`)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io/) `v1.47.0`
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) `v0.63.3`
- [OpenCV (headless)](https://opencv.org/) `opencv-python-headless==4.12.0.88`
- [NumPy](https://numpy.org/) (latest compatible)
- [Pillow](https://python-pillow.org/) `v11.3.0`
- [TensorFlow](https://www.tensorflow.org/) `v2.18.0`
- [Keras](https://keras.io/) `v3.8.0`

---

## 📦 Installation

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
