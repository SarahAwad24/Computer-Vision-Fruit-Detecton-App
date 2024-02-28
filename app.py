import streamlit as st
import numpy as np
import av
import torch
from ultralytics import YOLO  # Adjust based on the actual import from ultralytics
import cv2
import logging
import queue
from pathlib import Path
from typing import NamedTuple
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Assume this script is being run in the project directory where the model is stored.
# For deployment, ensure the model path is relative to the project structure and correctly set.

# Define detection namedtuple
class Detection(NamedTuple):
    class_name: str
    score: float
    box: np.ndarray

# UI elements
score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
result_queue = queue.Queue()

# Path setup - Check if the model path needs to be environment-specific
MODEL_PATH = Path.cwd() / "best.pt"  # Adjusted to current working directory for simplicity

# Confirm model path exists, else raise error
if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()  # Stop execution if model file is not found

# Load model
cache_key = "object_detection_model"
if cache_key in st.session_state:
    model = st.session_state[cache_key]
else:
    model = YOLO(str(MODEL_PATH))  # Ensure the path is converted to string for compatibility
    st.session_state[cache_key] = model

# Adjust video processing to work with your PyTorch model
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    results = model(image)  # Model inference

    # Process results
    detections = [
        Detection(
            class_name=res['class_name'],
            score=res['confidence'],
            box=res['box']
        ) for res in results.xyxy[0]  # Adjust based on results structure
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
