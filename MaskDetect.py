import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from PIL import Image
import ultralytics

# Load YOLO model
model = ultralytics.YOLO('best_yet.pt')

# App Title
st.title("Real-Time and Video-based Face Mask Detection using YOLOv8")

# Sidebar for mode selection
st.sidebar.title("Choose Detection Mode")
mode = st.sidebar.radio(
    "Select how you want to perform face mask detection:",
    ('Real-Time Webcam Stream', 'Upload a Video')
)

# Class for Real-Time Processing via Webcam
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO model on the frame
        results = model(img)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Return the annotated frame as a video stream
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Real-Time Webcam Stream Option
if mode == 'Real-Time Webcam Stream':
    st.subheader("Real-Time Face Mask Detection (Webcam)")

    st.write("Press **Start** to begin real-time detection using your webcam.")
    
    # Start webcam stream and run the YOLO model on each frame
    webrtc_streamer(key="real-time-detection", video_processor_factory=VideoProcessor)

# Video Upload Option
elif mode == 'Upload a Video':
    st.subheader("Face Mask Detection on Uploaded Video")

    # File uploader for video upload
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if video_file is not None:
        # Convert the uploaded video to a format OpenCV can read
        video_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
        video = cv2.VideoCapture(video_bytes)

        video_placeholder = st.empty()

        # Process the uploaded video frame by frame
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                st.warning("End of video or error reading file.")
                break

            # Run YOLO model on the frame
            results = model(frame)

            # Annotate the frame with detection results
            annotated_frame = results[0].plot()

            # Convert the frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Display the annotated video frame by frame
            video_placeholder.image(img)

        video.release()
