import streamlit as st
import pandas as pd
import cv2
import torch
import tempfile
import settings
import helper
from pathlib import Path


def main():
    CLASSES = [
        "Apple", "Banana", "Beetroot", "Bitter Gourd", "Bottle Gourd", "Cabbage",
        "Capsicum", "Carrot", "Cauliflower", "Cherry", "Chilli", "Coconut",
        "Cucumber", "EggPlant", "Ginger", "Grape", "Green Orange", "Kiwi",
        "Maize", "Mango", "Melon", "Okra", "Onion", "Orange", "Peach", "Pear",
        "Peas", "Pineapple", "Pomegranate", "Potato", "Radish", "Strawberry",
        "Tomato", "Turnip", "Watermelon", "Walnut", "Almond"
    ]

    st.title('Fruit Detection in Video')
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    model_path = 'best.pt'
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        return  # Exit if the model cannot be loaded

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
                for detection in results.xyxy[0]:
                    x_min, y_min, x_max, y_max, conf, cls_id = detection[:6].cpu().numpy()
                    label = CLASSES[int(cls_id)]
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
                print(f"No detections in frame {frame_number}")

            frame_number += 1

        st.dataframe(detections_df)

if __name__ == "__main__":
    main()
