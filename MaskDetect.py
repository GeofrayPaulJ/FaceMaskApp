import streamlit as st
import ultralytics
import cv2
from PIL import Image
import numpy as np

model = ultralytics.YOLO('best_yet.pt')

st.title("Real-Time Face Mask Detection using YOLOv8")

if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

if 'cap' not in st.session_state:
    st.session_state.cap = None  

video_placeholder = st.empty()

camera_index = 0  # Try changing this index if needed

if st.button("Turn On Camera") and not st.session_state.camera_on:
    st.session_state.camera_on = True
    st.session_state.cap = cv2.VideoCapture(camera_index)
    
    if not st.session_state.cap.isOpened():
        st.warning("Failed to open camera. Please check your device and permissions.")

if st.button("Turn Off Camera") and st.session_state.camera_on:
    st.session_state.camera_on = False
    if st.session_state.cap is not None:
        st.session_state.cap.release() 
        st.session_state.cap = None

if st.session_state.camera_on:
    cap = st.session_state.cap
    if cap is not None:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read from camera. Please check your device.")
                break

            results = model(frame)
            annotated_frame = results[0].plot()  
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            video_placeholder.image(img, use_column_width=True)

            if not st.session_state.camera_on:
                break

        if st.session_state.cap is not None:
            st.session_state.cap.release() 
            st.session_state.cap = None
