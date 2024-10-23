import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Streamlit app title
st.title("Real-Time Face Mask Detection using YOLOv8")

# Session state to track camera status
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# Toggle button for camera
if st.session_state.camera_on:
    button_label = "Turn Off Camera"
else:
    button_label = "Turn On Camera"

# Placeholder for displaying video
video_placeholder = st.empty()

# Function to handle camera stream
def video_streaming():
    class MaskDetectionTransformer(VideoTransformerBase):
        def transform(self, frame):
            return frame.to_ndarray(format="bgr24")

    webrtc_streamer(key="mask_detection", video_transformer_factory=MaskDetectionTransformer)

# Button to toggle camera
if st.button(button_label):
    st.session_state.camera_on = not st.session_state.camera_on

# Start or stop video stream based on button
if st.session_state.camera_on:
    video_streaming()
else:
    video_placeholder.empty()
