import streamlit as st
import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Union
from ultralytics import YOLO
import io

# Define the device to be used for computation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO model
model = YOLO('best.pt')


# Function for removing temporary files
def remove_temp(temp_file: str = 'temp') -> None:
    
    for file in os.listdir(temp_file):
        os.remove(os.path.join(temp_file, file))


# Function for downloading an image with detected objects
def download_image(image: np.ndarray) -> None:
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
    
    # Open the image
    image = Image.open(image)

    # Perform object detection
    results = model.predict(image, conf=confidence_threshold,
                            max_det=max_detections, device=DEVICE)

    # Plot the detected objects on the image
    plot = results[0].plot()

    # Convert color space from BGR to RGB
    processed_image = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

    # Show the detected image
    st.image(processed_image, caption='Detected Image.', use_column_width='auto', output_format='auto', width=None)

    # Offer download option for the detected image
    download_image(processed_image)


# Function for real-time object detection in a video stream
def video_detect(uploaded_video: Union[None, io.BytesIO], confidence_threshold: float,
                 max_detections: int) -> None:
    
    # Check if a video is uploaded
    if uploaded_video is not None:
        # Create a temporary file to save the uploaded video
        temp_video_path = 'temp_video.mp4'

        # Write uploaded video content to the temporary file
        with open(temp_video_path, "wb") as temp_video_file:
            temp_video_file.write(uploaded_video.getbuffer())

        # Open the uploaded video file
        cap = cv2.VideoCapture(temp_video_path)

        # Display for video feed
        stframe = st.empty()

        # Process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL Image
            img = Image.fromarray(frame)

            # Perform object detection - assuming your model has a similar API to Ultralytics YOLO
            results = model.predict(img, conf=confidence_threshold, max_det=max_detections, device=DEVICE)

            # Iterate over each result and process it
            for result in results.xyxy[0]:
                # Extract bounding box coordinates and class label
                x1, y1, x2, y2, class_id, confidence = result
                class_name = model.names[int(class_id)]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name}: {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Update the frame in the Streamlit app
            stframe.image(frame, channels="BGR", use_column_width=True)

        # Release the video capture object and remove the temp file
        cap.release()
        os.remove(temp_video_path)



st.set_page_config(
    page_title=" Welcome to Nutrivision",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Welcome to Nutrivision!")


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


# Perform object detection based on the selected source
if uploaded_image is not None:
    # Object detection for uploaded image
    image_detect(image=uploaded_image, confidence_threshold=confidence_threshold
                 )

elif uploaded_video is not None:
    # Object detection for uploaded video
    video_detect(uploaded_video=uploaded_video, confidence_threshold=confidence_threshold
                 )

    # Remove temporary files
    remove_temp()
