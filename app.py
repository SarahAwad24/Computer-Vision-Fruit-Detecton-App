import streamlit as st
import numpy as np
from pathlib import Path
from typing import List, NamedTuple
import av
import torch
from ultralytics import YOLO  # Assumed correct import based on context; typically, it's yolov5 for ultralytics
import cv2

# Define detection namedtuple
class Detection(NamedTuple):
    class_name: str
    score: float
    box: np.ndarray

# Path setup
HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_PATH = "C:\Users\sarah\OneDrive\Documents\GitHub\Computer-Vision\best.pt"

# Load model
cache_key = "object_detection_model"
if cache_key in st.session_state:
    model = st.session_state[cache_key]
else:
    model = YOLO(MODEL_PATH)  # Assumed usage; adjust based on actual ultralytics API
    st.session_state[cache_key] = model

# UI elements
score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
result_queue = queue.Queue()

# Adjust video processing to work with your PyTorch model
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")

    # Convert image for model input
    results = model(image)

    # Process results
    detections = [
        Detection(
            class_name=res['name'],
            score=res['confidence'],
            box=res['box']
        )
        for res in results.xyxy[0]  # Adjust based on how results are structured
    ]

    # Render bounding boxes and captions
    for detection in detections:
        if detection.score >= score_threshold:
            xmin, ymin, xmax, ymax = detection.box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{detection.class_name}: {round(detection.score * 100, 2)}%",
                (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    result_queue.put(detections)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

# WebRTC streamer setup
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display detection results
if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)

st.markdown("This demo uses a YOLO model for object detection.")
