import streamlit as st
import pandas as pd
import cv2
import torch
import tempfile
import settings
import helper
from pathlib import Path
from utils.plots import plot_one_box
from models.experimental import attempt_load
from utils.general import non_max_suppression



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

    # Assuming the model path is correctly set to where you uploaded your model
    model_path = 'path/to/best (2).pt'  # Adjust this to your model's upload path

    # Load the YOLO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(model_path, map_location=device)  # Load your model
    model.eval()

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())

        vid = cv2.VideoCapture(tfile.name)

        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            # Preprocess the frame for YOLO model
            # This part might need adjustments based on how you trained your model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(frame_rgb).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

            # Parse detections (adjust based on your needs)
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to frame size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                    # Print results
                    for *xyxy, conf, cls in det:
                        label = f'{CLASSES[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=3)

            # Display frame
            st.image(frame)

        vid.release()

if __name__ == "__main__":
    main()