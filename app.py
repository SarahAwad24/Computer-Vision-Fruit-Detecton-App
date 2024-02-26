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

class Detection(NamedTuple):
    class_name: str
    label: str
    score: float
    box: np.ndrray

    def generate_label_colors():
        return np.random.uniform(0, 255, size=(len(classes), 3))

    COLORS = generate_label_colors()
