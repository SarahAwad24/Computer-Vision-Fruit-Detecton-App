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


import torch

# Assuming your model is a PyTorch model
MODEL_PATH = ROOT / "best.pt"


classes = [
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

class Detection(NamedTuple):
    class_name: str
    label: str
    score: float
    box: np.ndarray

def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(classes), 3))

COLORS = generate_label_colors()

if cache_key in st.session_state:
    model = st.session_state[cache_key]
else:
    model = torch.load(MODEL_PATH)
    model.eval()  # Set the model to evaluation mode
    st.session_state[cache_key] = model

score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

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

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)

st.markdown(
    "This demo uses a model and code from "
    "https://github.com/robmarkcole/object-detection-app. "
    "Many thanks to the project."
)