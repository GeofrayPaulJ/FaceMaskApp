import streamlit as st
import ultralytics
import cv2
from PIL import Image
import numpy as np

model = ultralytics.YOLO('model/best_yet.pt')

st.title("Real-Time Face Mask Detection using YOLOv8")

# Using session state to track the camera status and capture object
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

if 'cap' not in st.session_state:
    st.session_state.cap = None  # To hold the video capture object

video_placeholder = st.empty()

# Create a button to turn on the camera
if st.button("Turn On Camera") and not st.session_state.camera_on:
    st.session_state.camera_on = True
    st.session_state.cap = cv2.VideoCapture(0)  # Start video capture

# Create a button to turn off the camera
if st.button("Turn Off Camera") and st.session_state.camera_on:
    st.session_state.camera_on = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()  # Release the camera if it's on
        st.session_state.cap = None

# If the camera is on, display the video feed with detected objects
if st.session_state.camera_on:
    cap = st.session_state.cap
    if cap is not None:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read from camera. Please check your device.")
                break

            # Process the frame using the YOLO model
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()  # Assuming this plots the detections

            # Convert frame to RGB format for Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Display the frame in the placeholder
            video_placeholder.image(img)

            # Stop the loop if camera is turned off
            if not st.session_state.camera_on:
                break

        # Safely release the video capture object when done
        st.session_state.cap.release()
        st.session_state.cap = None
        cv2.destroyAllWindows()
