import streamlit as st
import ultralytics
import cv2
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model = ultralytics.YOLO('best_yet.pt')

# Streamlit app title
st.title("Real-Time Face Mask Detection using YOLOv8")

# Initialize session state variables for the camera and video capture
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

if 'cap' not in st.session_state:
    st.session_state.cap = None  

# Placeholder for displaying the video
video_placeholder = st.empty()

# Button to turn on the camera
if st.button("Turn On Camera") and not st.session_state.camera_on:
    st.session_state.camera_on = True
    st.session_state.cap = cv2.VideoCapture(0)

    # Check if the camera is accessible
    if not st.session_state.cap.isOpened():
        st.error("Error: Could not access the camera. Please check the device.")
        st.session_state.camera_on = False

# Button to turn off the camera
if st.button("Turn Off Camera") and st.session_state.camera_on:
    st.session_state.camera_on = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

# Stream the video if the camera is on
if st.session_state.camera_on:
    cap = st.session_state.cap
    if cap is not None:
        # Capture and process frames
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from camera. Please check your device.")
        else:
            # Run the YOLO model on the frame
            results = model(frame)

            # Annotate the frame with detections
            annotated_frame = results[0].plot()

            # Convert the frame to RGB for display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Display the frame
            video_placeholder.image(img)

    # Release the camera when the session ends
    if not st.session_state.camera_on and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        cv2.destroyAllWindows()
