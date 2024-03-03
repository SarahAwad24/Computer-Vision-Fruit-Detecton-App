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
from const import CLASSES, COLORS


# Assume this script is being run in the project directory where the model is stored.
# For deployment, ensure the model path is relative to the project structure and correctly set.
def main():
    # Load data
    st.set_page_config(page_title='Nutrivision', page_icon='🍎', layout='centered', initial_sidebar_state='auto')

    # Display logo
    image = Image.open('C:/Users/sarah/OneDrive/Documents/GitHub/Computer-Vision-Deep-Learning/logo.jpg')
    st.image(image, use_column_width=True)

    st.markdown("""
    <style>.title {text-align: center;}</style><h1 class="title">Live Fuel Leakage Monitoring</h1>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
# Define detection namedtuple
class Detection(NamedTuple):
    class_name: str
    score: float
    box: np.ndarray
