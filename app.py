from PIL import Image
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


# Assume this script is being run in the project directory where the model is stored.
# For deployment, ensure the model path is relative to the project structure and correctly set.
def main():
    # Load data
    st.set_page_config(page_title='Nutrivision', page_icon='🍎', layout='centered', initial_sidebar_state='auto')

    # Display logo
    image = Image.open('logo.jpg')
    with st.beta_container():
        st.image(image, use_column_width=False, width=500, output_format='auto')
        st.markdown("""
            <style>
            .stImage {
                display: flex;
                justify-content: center;
            }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("""
    <style>.title {text-align: center;}</style><h1 class="title">Welcome to Nutrivision</h1>
    """, unsafe_allow_html=True)

    CLASSES = [
    "Apple", "Banana", "Beetroot", "Bitter_Gourd", "Bottle_Gourd", "Cabbage",
    "Capsicum", "Carrot", "Cauliflower", "Cherry", "Chilli", "Coconut",
    "Cucumber", "EggPlant", "Ginger", "Grape", "Green_Orange", "Kiwi",
    "Maize", "Mango", "Melon", "Okra", "Onion", "Orange", "Peach", "Pear",
    "Peas", "Pineapple", "Pomegranate", "Potato", "Radish", "Strawberry",
    "Tomato", "Turnip", "Watermelon", "walnut", "almond"
]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


if __name__ == "__main__":
    main()
# Define detection namedtuple
class Detection(NamedTuple):
    class_name: str
    score: float
    box: np.ndarray
