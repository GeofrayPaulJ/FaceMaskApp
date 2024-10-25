import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np

# Streamlit app title
st.title("Real-Time Face Mask Detection using YOLOv8")

# Toggle button for camera
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

button_label = "Turn Off Camera" if st.session_state.camera_on else "Turn On Camera"

# Button to toggle camera
if st.button(button_label):
    st.session_state.camera_on = not st.session_state.camera_on

# Video transformer class for mask detection
class MaskDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Add mask detection processing here (using YOLOv8 or similar model)
        img = frame.to_ndarray(format="bgr24")

        # Return unprocessed frame (as an example, for actual detection, add YOLOv8 inference here)
        return img

# Start or stop video stream based on the button
if st.session_state.camera_on:
    webrtc_streamer(key="mask_detection", video_transformer_factory=MaskDetectionTransformer)
else:
    st.write("Camera is off")
