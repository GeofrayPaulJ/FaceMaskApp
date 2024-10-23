import streamlit as st
import ultralytics
import cv2
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the YOLOv8 model
model = ultralytics.YOLO('best_yet.pt')

# Streamlit app title
st.title("Real-Time Face Mask Detection using YOLOv8")

# Placeholder for displaying the video
video_placeholder = st.empty()

# Define the video transformer class for processing frames
class MaskDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to ndarray
        img = frame.to_ndarray(format="bgr24")

        # Run the YOLO model on the frame
        results = model(img)

        # Annotate the frame with detections
        annotated_frame = results[0].plot()

        # Convert the frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # Display the frame in the placeholder
        video_placeholder.image(img_pil)
        
        return frame

# Button to start the camera stream
if st.button("Turn On Camera"):
    webrtc_streamer(key="face-mask-detection", video_transformer_factory=MaskDetectionTransformer)

# Button to stop the camera stream (controlled by refreshing or app reset)
if st.button("Turn Off Camera"):
    st.warning("To turn off the camera, please refresh or restart the app.")

