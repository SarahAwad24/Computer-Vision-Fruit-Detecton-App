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

    # Assuming your model is trained and saved as 'best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model.eval()

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

            # Use your predefined classes to interpret the model's output
            for *xyxy, conf, cls in results.xyxy[0]:
                # Use FRUIT_CLASSES to get the actual fruit name
                label = CLASSES[int(cls)]
                detections_df = detections_df.append({
                    'Frame': frame_number,
                    'Fruit': label,
                    'Confidence': conf,
                    'x1': xyxy[0],
                    'y1': xyxy[1],
                    'x2': xyxy[2],
                    'y2': xyxy[3]
                }, ignore_index=True)

            frame_number += 1

        st.dataframe(detections_df)


if __name__ == "__main__":
    main()
# Define detection namedtuple
