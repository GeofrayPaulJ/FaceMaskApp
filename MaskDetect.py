import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from PIL import Image
import ultralytics

# Load YOLO model
model = ultralytics.YOLO('best_yet.pt')

st.title("Real-Time Face Mask Detection using YOLOv8")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO model on the frame
        results = model(img)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Use webrtc-streamer to capture and display webcam stream
webrtc_streamer(key="mask-detection", video_processor_factory=VideoProcessor)
