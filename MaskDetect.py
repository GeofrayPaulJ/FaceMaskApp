import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Streamlit app title
st.title("Real-Time Face Mask Detection using YOLOv8")

# Session state to track camera status
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# Toggle button for camera
button_label = "Turn Off Camera" if st.session_state.camera_on else "Turn On Camera"

# Function to handle camera stream
class MaskDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame.to_ndarray(format="bgr24")

# Button to toggle camera
if st.button(button_label):
    st.session_state.camera_on = not st.session_state.camera_on

# Start or stop video stream based on button
if st.session_state.camera_on:
    webrtc_streamer(key="mask_detection", video_transformer_factory=MaskDetectionTransformer)
