import cv2
from ultralytics import YOLO
import supervision as sv
import time

# Choose between camera and video (0 for camera, path for video)
use_camera = True
video_path = "input/sample2.mp4"  # Replace with your video path if using video

frame_width = 1280
frame_height = 720

def main():
    # Capture object (initialized inside main for flexibility)
    cap = None

    # Use camera
    if use_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        print("Using camera...")
    # Use video
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        print(f"Using video: {video_path}")

    model = YOLO("model/best_yet.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Set execution duration in seconds
    duration = 30

    start_time = time.time()  # Record start time for duration control

    while True:
        ret, frame = cap.read()
        if ret : 
            if time.time() - start_time >= duration:  # Check if duration has elapsed
                break

            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)

            filtered_detections = []
            labels = []

            # Filter and create labels based on class and confidence
            for detection in detections:
                class_id = detection[2]  # Assuming class ID is at index 2
                confidence = detection[1]  # Assuming confidence is at index 1

                if class_id == 1 and confidence > 0.85:
                    filtered_detections.append(detection)
                    label = f"Mask {confidence:.2f}"
                    labels.append(label)

                # Class 0: Show bounding box only if confidence > 0.5
                elif class_id == 0 and confidence > 0.25:      
                    filtered_detections.append(detection)
                    label = f"No-Mask {confidence:.2f}"
                    labels.append(label)

                # All other cases
                else:
                    filtered_detections.append(detection)
                    label = "Unknown"  # Include confidence for better debugging
                    labels.append(label)
                                    

            # Annotate the frame with filtered detections and labels
            frame = box_annotator.annotate(
                scene=frame,
                detections=filtered_detections,
                labels=labels
            )

            cv2.imshow("yolov8", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cap:
        cap.release()
    cv2.destroyAllWindows()  # Close windows properly


if __name__ == "__main__":
    main()

