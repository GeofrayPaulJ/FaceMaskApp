import streamlit as st
import ultralytics
import cv2
from PIL import Image
import numpy as np

# Load YOLO model
model = ultralytics.YOLO('best_yet.pt')

st.title("Real-Time Face Mask Detection using YOLOv8")

# Initialize camera state
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

if 'cap' not in st.session_state:
    st.session_state.cap = None  

video_placeholder = st.empty()

# Turn on the camera
if st.button("Turn On Camera") and not st.session_state.camera_on:
    st.session_state.camera_on = True
    st.session_state.cap = cv2.VideoCapture(0)

# Turn off the camera
if st.button("Turn Off Camera") and st.session_state.camera_on:
    st.session_state.camera_on = False
    if st.session_state.cap is not None:
        st.session_state.cap.release() 
        st.session_state.cap = None

# Display the camera feed and run YOLO model
if st.session_state.camera_on:
    cap = st.session_state.cap
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from camera. Please check your device.")
        else:
            # Run YOLO model on the frame
            results = model(frame)

            # Annotate the frame with detection results
            annotated_frame = results[0].plot()

            # Convert frame to RGB format for display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Display the frame in Streamlit
            video_placeholder.image(img)

# Release camera and destroy windows (optional for Streamlit, might not be necessary)
if not st.session_state.camera_on and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None
    cv2.destroyAllWindows()
