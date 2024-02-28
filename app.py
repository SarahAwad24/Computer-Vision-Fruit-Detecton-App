import streamlit as st
from PIL import Image
import torch as torch
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av as av
import logging
import queue
from pathlib import Path
from typing import List, NamedTuple, Tuple
import cv2
import torch
from ultralytics import YOLO

HERE = Path(__file__).parent
ROOT = HERE

logger = logging.getLogger(__name__)

MODEL_PATH = ROOT / "best.pt"


CLASSES = [
    "Apple",
    "Banana",
    "Beetroot",
    "Bitter_Gourd",
    "Bottle_Gourd",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Cherry",
    "Chilli",
    "Coconut",
    "Cucumber",
    "EggPlant",
    "Ginger",
    "Grape",
    "Green_Orange",
    "Kiwi",
    "Maize",
    "Mango",
    "Melon",
    "Okra",
    "Onion",
    "Orange",
    "Peach",
    "Pear",
    "Peas",
    "Pineapple",
    "Pomegranate",
    "Potato",
    "Radish",
    "Strawberry",
    "Tomato",
    "Turnip",
    "Watermelon",
    "almond",
    "walnut"
]
# Load YOLO model
#net = cv2.dnn.readNetFromDarknet(str(HERE / "yolov8m.cfg"), str(MODEL_PATH))
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = YOLO(MODEL_PATH)
score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# Setup the result queue
result_queue = queue.Queue()

class Detection(NamedTuple):
    class_name: str
    label: str
    score: float
    box: np.ndrray

    def generate_label_colors():
        return np.random.uniform(0, 255, size=(len(classes), 3))

    COLORS = generate_label_colors()




# Define the callback function for video frames
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")

    # Run inference
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    output = net.forward()

    h, w = image.shape[:2]

    # Convert the output array into a structured form.
    output = output.squeeze()  # (1, 1, N, 7) -> (N, 7)
    output = output[output[:, 2] >= score_threshold]
    detections = [
        Detection(
            class_id=int(detection[1]),
            label=CLASSES[int(detection[1])],
            score=float(detection[2]),
            box=(detection[3:7] * np.array([w, h, w, h])),
        )
        for detection in output
    ]

    # Render bounding boxes and captions
    for detection in detections:
        caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
        color = COLORS[detection.class_id]
        xmin, ymin, xmax, ymax = detection.box.astype("int")

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image,
            caption,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    result_queue.put(detections)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# Setup the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=video_frame_callback,
)

# Add a button for taking a snapshot
if st.button("Take Snapshot"):
    if not webrtc_ctx.state.playing:
        st.warning("No video stream is running!")
    else:
        snapshot = result_queue.get()
        # Output the data frame or do any other processing with the snapshot
        # For example:
        st.dataframe(snapshot)