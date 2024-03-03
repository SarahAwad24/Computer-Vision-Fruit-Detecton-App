import streamlit as st
import pandas as pd
import cv2
import torch
import tempfile
import settings
import helper
from pathlib import Path



def draw_labels_and_boxes(image, detections, class_names):
    for detection in detections:
        x_min, y_min, x_max, y_max, conf, cls_id = detection[:6]
        label = class_names[int(cls_id)]
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(image, f'{label} {conf:.2f}', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

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

    model_path = Path(settings.DETECTION_MODEL)

    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())

        vid = cv2.VideoCapture(tfile.name)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        while True:
            ret, frame = vid.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            if len(results.xyxy[0]) > 0:
                annotated_frame = draw_labels_and_boxes(frame, results.xyxy[0].cpu().numpy(), CLASSES)
                out.write(annotated_frame)
            else:
                out.write(frame)  # Write original frame if no detections

        vid.release()
        out.release()

        # Display the processed video
        st.video('output.mp4')

if __name__ == "__main__":
    main()