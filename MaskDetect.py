import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from PIL import Image
import ultralytics

# Load YOLO model
model = ultralytics.YOLO('best_yet.pt')

# App Title
st.title("Real-Time and Video-based Face Mask Detection using YOLOv8")

# Initialize session state to manage camera status
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# Placeholder for live video stream
video_placeholder = st.empty()

# Class for Real-Time Processing via Webcam
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO model on the frame
        results = model(img)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Return the annotated frame as a video stream
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Camera Toggle Button
if st.session_state.camera_on:
    button_label = "Turn Off Camera"
else:
    button_label = "Turn On Camera"

if st.button(button_label):
    st.session_state.camera_on = not st.session_state.camera_on

# Start or Stop the webcam stream
if st.session_state.camera_on:
    st.subheader("Live Video Stream (Webcam)")
    webrtc_streamer(key="real-time-detection", video_processor_factory=VideoProcessor)
else:
    st.warning("Camera is turned off.")
