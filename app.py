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
import streamlit as st
import pandas as pd
import cv2
import torch
import tempfile
import settings
import helper




# Assume this script is being run in the project directory where the model is stored.
# For deployment, ensure the model path is relative to the project structure and correctly set.
def main():
    

    # Define your fruit classes
    # Replace these with the actual classes you have
    CLASSES = [
    "Apple", "Banana", "Beetroot", "Bitter_Gourd", "Bottle_Gourd", "Cabbage",
    "Capsicum", "Carrot", "Cauliflower", "Cherry", "Chilli", "Coconut",
    "Cucumber", "EggPlant", "Ginger", "Grape", "Green_Orange", "Kiwi",
    "Maize", "Mango", "Melon", "Okra", "Onion", "Orange", "Peach", "Pear",
    "Peas", "Pineapple", "Pomegranate", "Potato", "Radish", "Strawberry",
    "Tomato", "Turnip", "Watermelon", "walnut", "almond"
    ]

    st.title('Fruit Detection in Video')
    st.write('Upload a video, and the model will detect fruits in it.')

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

    model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
    

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        detections_df = pd.DataFrame(columns=['Frame', 'Fruit', 'Confidence', 'x1', 'y1', 'x2', 'y2'])
        
        vid = cv2.VideoCapture(tfile.name)

        frame_number = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            results = model(frame)

        if len(results.xyxy[0]) > 0:
    # Iterate over detections
        for detection in results.xyxy[0]:
        # Unpack detection tensor, converting tensor elements to Python scalars with .item()
            x_min, y_min, x_max, y_max, conf, cls_id = detection[:6].cpu().numpy()
            label = CLASSES[int(cls_id)]  # Get the class label using detected class ID
        # Append detection info to your DataFrame as before
            detections_df = detections_df.append({
            'Frame': frame_number,
            'Fruit': label,
            'Confidence': conf,
            'x1': x_min,
            'y1': y_min,
            'x2': x_max,
            'y2': y_max
            }, ignore_index=True)
        else:
    # Handle cases with no detections
            print("No detections")
            

        st.dataframe(detections_df)


if __name__ == "__main__":
    main()
# Define detection namedtuple
