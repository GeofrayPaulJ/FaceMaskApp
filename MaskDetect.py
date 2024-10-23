import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer
import ultralytics

# Load YOLO model
model = ultralytics.YOLO('best_yet.pt')

st.title("Real-Time Face Mask Detection using YOLOv8")

# Define session state for camera control
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

# WebRTC streamer callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")  # Convert the frame to ndarray

    # Perform face mask detection using YOLO model
    results = model(img)
    
    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Convert the annotated frame to RGB for Streamlit display
    img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    return img_pil

# Button to start/stop streaming
if st.button("Start Camera") and not st.session_state.streaming:
    st.session_state.streaming = True

if st.button("Stop Camera") and st.session_state.streaming:
    st.session_state.streaming = False

# If streaming, show the video stream
if st.session_state.streaming:
    webrtc_streamer(key="face-mask-detection", video_frame_callback=video_frame_callback)
