import os
import io
import cv2
import yaml
import torch
import numpy as np
from PIL import Image
import streamlit as st
from typing import Union
from ultralytics import YOLO
import yaml
import streamlit as st

# Define the device to be used for computation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO model
model = YOLO('best.pt')


# Function for loading data from yaml file
def load_yaml(file_path: str) -> dict:
    """
    Load YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Loaded YAML data.
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(exc)


# Function for removing temporary files
def remove_temp(temp_file: str = 'temp') -> None:
    """
    Remove all files in the specified temporary directory.

    Args:
        temp_file (str, optional): Path to the temporary directory. Defaults to 'temp'.
    """
    for file in os.listdir(temp_file):
        os.remove(os.path.join(temp_file, file))


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
def image_detect(image: str, confidence_threshold: float, max_detections: int, class_ids: list) -> None:
    """
    Detects objects in an image using YOLO model.

    Args:
        image (str): Path to the input image.
        confidence_threshold (float): Confidence threshold for object detection.
        max_detections (int): Maximum number of detections.
        class_ids (list): List of class IDs to consider for detection.
    """
    # Open the image
    image = Image.open(image)

    # Perform object detection
    results = model.predict(image, conf=confidence_threshold,
                            max_det=max_detections,classes=class_ids, device=DEVICE)

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
                 max_detections: int, class_ids: list) -> None:
    """
    Performs real-time object detection in a video stream.

    Args:
        uploaded_video (Union[None, io.BytesIO, str]): Uploaded video file.
        confidence_threshold (float): Confidence threshold for object detection.
        max_detections (int): Maximum number of detections.
        class_ids (list): List of class IDs to consider for detection.
    """

    # Ensure there is a video file uploaded
    if uploaded_video is not None:
        # Create a temporary file to save the uploaded video
        temp_video_path = f"temp_video.mp4"
        
        # Write uploaded video content to the temporary file
        with open(temp_video_path, "wb") as temp_video_file:
            temp_video_file.write(uploaded_video.getbuffer())  # Using getbuffer for BytesIO object

        # Open the uploaded video file
        cap = cv2.VideoCapture(temp_video_path)

        # Width and height for cv2.VideoWriter
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        
        # Define the codec and create VideoWriter object to save the output video
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
        
        # Process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to image format understood by the model
            img = Image.fromarray(frame)

            # Perform object detection
            # Define the video_feed object
            video_feed = st.empty()

            results = model.predict(img, conf=confidence_threshold,
                                    max_det=max_detections, classes=class_ids, device=DEVICE)

            # Process the results and draw bounding boxes on the frame
            # Note: You'll need to adapt this if your model.predict doesn't return the expected format
            for result in results:
                # Example: Extract bounding box coordinates, labels, and confidence scores
                # and draw them on `frame`. This part highly depends on your `model.predict` output format.
                pass  # Adapt based on your model's specific output format.

            # Write the frame into the file 'output.avi'
            out.write(frame)

            # Display the frame
            video_feed.image(frame, channels="BGR", use_column_width=True)

        # Release everything when job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Display the output video
        st.video('output.avi')
