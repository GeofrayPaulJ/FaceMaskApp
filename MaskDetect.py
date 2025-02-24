import streamlit as st
import cv2
import numpy as np

# Streamlit app title
st.title("Real-Time Face Mask Detection using YOLOv8")

# Session state to track camera status
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# Toggle button for camera
button_label = "Turn Off Camera" if st.session_state.camera_on else "Turn On Camera"

# Button to toggle camera
if st.button(button_label):
    st.session_state.camera_on = not st.session_state.camera_on

# Function to display video frames using OpenCV VideoCapture
def display_video():
    cap = cv2.VideoCapture(0)  # Initialize the camera
    while st.session_state.camera_on:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture frame")
            break

        # Optional: process the frame with YOLOv8 for mask detection
        # For now, convert frame to RGB to display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB")

    cap.release()  # Release the camera when stopped

# Start or stop video display
if st.session_state.camera_on:
    display_video()
else:
    st.write("Camera is off")
