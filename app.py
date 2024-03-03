import streamlit as st
import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Union
from ultralytics import YOLO
import io
import pandas as pd

from ultralytics.utils.plotting import Annotator

# Define the device to be used for computation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO model
model = YOLO('best_no_transfer.pt')


# Function for removing temporary files
def remove_temp(temp_file: str = 'temp') -> None:
    """
    Remove all files in the specified temporary directory. Creates the directory if it does not exist.

    Args:
        temp_file (str, optional): Path to the temporary directory. Defaults to 'temp'.
    """
    # Check if the directory exists
    if not os.path.exists(temp_file):
        # Create the directory if it does not exist
        os.makedirs(temp_file)
        print(f"Directory '{temp_file}' was created")
        return

    # If the directory exists, proceed to remove files
    for file in os.listdir(temp_file):
        os.remove(os.path.join(temp_file, file))
    print(f"All files in '{temp_file}' have been removed.")


# Function for downloading an image with detected objects
def download_image(image: np.ndarray) -> None:
    """
    Downloads the image with detected objects.

    Args:
        image (np.ndarray): Image array with detected objects.
    """
    # Convert NumPy array to PIL.Image object
    image = Image.fromarray(image)

    # Convert image to bytes
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()

    # Display download button
    if st.download_button(label="Download Image", data=img_byte_array, file_name='detected_image.png',
                          mime='image/png'):
        st.success("Downloaded successfully!")


# Function for detecting objects in an image
def image_detect(image: str, confidence_threshold: float, max_detections: int) -> None:
    """
    Detects objects in an image using YOLO model.

    Args:
        image (str): Path to the input image.
        confidence_threshold (float): Confidence threshold for object detection.
        max_detections (int): Maximum number of detections.
    """
    # Open the image
    image = Image.open(image)

    # Perform object detection
    results = model.predict(image, conf=confidence_threshold,
                            max_det=max_detections, device=DEVICE)

    # Plot the detected objects on the image
    plot = results[0].plot()

    # Convert color space from BGR to RGB
    processed_image = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        # Show the original image
        st.image(image, caption='Original Image.', use_column_width='auto', output_format='auto', width=None)

    with col2:
    # Show the detected image
        st.image(processed_image, caption='Detected Image.', use_column_width='auto', output_format='auto', width=None)

    # Offer download option for the detected image
    download_image(processed_image)

    # Button to get labels and fruits
    if st.button("Get Labels and Fruits"):
        labels, fruits = get_labels_and_fruits(results)
        st.write("Detected Fruits:", fruits)
        goal_dicc = {
                '1': 'lose weight',
                '2': 'gain weight',
                '3': 'maintain weight'
            }
            
        st.radio("Select your goal", list(goal_dicc.keys()), format_func=lambda x: goal_dicc[x], horizontal=True)

# Function for real-time object detection in a video stream
def video_detect(uploaded_video: Union[None, io.BytesIO], confidence_threshold: float,
                 max_detections: int) -> None:
    """
    Performs real-time object detection in a video stream.

    Args:
        uploaded_video (Union[None, io.BytesIO]): Uploaded video file.
        confidence_threshold (float): Confidence threshold for object detection.
        max_detections (int): Maximum number of detections.
    """
    fruits = []
    class_names = ['Apple', 'Almond', 'Banana', 'Beetroot', 'Bitter_Gourd', 'Bottle_Gourd', 'Cabbage', 'Capsicum',
                   'Carrot', 'Cauliflower', 'Cherry', 'Chilli', 'Coconut', 'Cucumber', 'EggPlant',
                   'Ginger', 'Grape', 'Green_Orange', 'Kiwi', 'Maize', 'Mango', 'Melon',
                   'Okra', 'Onion', 'Orange', 'Peach', 'Pear', 'Peas', 'Pineapple', 'Pomegranate',
                   'Potato', 'Radish', 'Strawberry', 'Tomato', 'Turnip', 'Walnut', 'Watermelon']
    # Check if a video is uploaded
    if uploaded_video is not None:
        # Create a temporary file to save the uploaded video
        temp_video_path = 'temp_video.mp4'

        # Write uploaded video content to the temporary file
        with open(temp_video_path, "wb") as temp_video_file:
            temp_video_file.write(uploaded_video.read())

        # Open the uploaded video file
        cap = cv2.VideoCapture(temp_video_path)

        # Display for video feed
        stframe = st.empty()

        # Process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection - assuming your model has a similar API to Ultralytics YOLO
            results = model.predict(frame, conf=confidence_threshold, max_det=max_detections, device=DEVICE)

            for r in results:
                annotator = Annotator(frame)

                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])
                detection_count = r.boxes.shape[0]

                for i in range(detection_count):
                    cls = int(r.boxes.cls[i].item())
                    name = r.names[cls]
                    if name in class_names and name not in fruits:
                        fruits.append(name)
                        print(fruits)
            stframe.image(frame, channels="BGR", use_column_width=True)

        # Release the video capture object and remove the temp file
        cap.release()
        os.remove(temp_video_path)

    

        # Button to get labels and fruits
    
    if st.button("Get Labels and Fruits"):
        labels, fruits = get_labels_and_fruits(results)
        st.write("Detected Labels:", labels)
        st.write("Detected Fruits:", fruits)


def get_labels_and_fruits(results):
    labels = []
    fruits = []
    class_names = ['Apple', 'Almond', 'Banana', 'Beetroot', 'Bitter_Gourd', 'Bottle_Gourd', 'Cabbage', 'Capsicum',
                   'Carrot', 'Cauliflower', 'Cherry', 'Chilli', 'Coconut', 'Cucumber', 'EggPlant',
                   'Ginger', 'Grape', 'Green_Orange', 'Kiwi', 'Maize', 'Mango', 'Melon',
                   'Okra', 'Onion', 'Orange', 'Peach', 'Pear', 'Peas', 'Pineapple', 'Pomegranate',
                   'Potato', 'Radish', 'Strawberry', 'Tomato', 'Turnip', 'Walnut', 'Watermelon']
    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            labels.append(model.names[int(c)])
        detection_count = r.boxes.shape[0]
        for i in range(detection_count):
            cls = int(r.boxes.cls[i].item())
            name = r.names[cls]
            if name in class_names and name not in fruits:
                fruits.append(name)
    return labels, fruits


# Set Streamlit page configuration
st.set_page_config(
    page_title=" Welcome to Nutrivision",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title for the web app
st.title("Welcome to Nutrivision!")

# Sidebar for selecting image source
# insert image "logo.jpg" into sidebar
image = Image.open('logo.jpg')
st.sidebar.image(image, use_column_width=True)
st.sidebar.title("Model Settings")
source = st.sidebar.radio("Select source:", ("Image", "Video"))

uploaded_image = None
uploaded_video = None
youtube_url = None

# Widget for uploading files
if source == "Image":
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

elif source == "Video":
    uploaded_video = st.sidebar.file_uploader("Choose a video...", type=["mp4"])

# Confidence threshold and max detections sliders
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
max_detections = st.sidebar.slider("Max Detections", min_value=1, max_value=500, value=300, step=1)

# Perform object detection based on the selected source
if uploaded_image is not None:
    # Object detection for uploaded image
    image_detect(image=uploaded_image, confidence_threshold=confidence_threshold,
                 max_detections=max_detections)

elif uploaded_video is not None:
    # Object detection for uploaded video
    video_detect(uploaded_video=uploaded_video, confidence_threshold=confidence_threshold,
                 max_detections=max_detections)

    # Remove temporary files
    remove_temp()

