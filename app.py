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
    box: np.ndarray

def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(classes), 3))

COLORS = generate_label_colors()
