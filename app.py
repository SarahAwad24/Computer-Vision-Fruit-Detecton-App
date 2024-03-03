import streamlit as st
from utils import *


# Set Streamlit page configuration
st.set_page_config(
    page_title=" Welcome to Nutrivision",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title for the web app
st.title("Streamlit Object Tracker with YOLOv8")

# Sidebar for selecting image source
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



class_names = [
    "Apple", "Banana", "Beetroot", "Bitter Gourd", "Bottle Gourd", "Cabbage",
    "Capsicum", "Carrot", "Cauliflower", "Cherry", "Chilli", "Coconut",
    "Cucumber", "EggPlant", "Ginger", "Grape", "Green Orange", "Kiwi",
    "Maize", "Mango", "Melon", "Okra", "Onion", "Orange", "Peach", "Pear",
    "Peas", "Pineapple", "Pomegranate", "Potato", "Radish", "Strawberry",
    "Tomato", "Turnip", "Watermelon", "Walnut", "Almond"
]

# Implementing multiselect in Streamlit using the defined list
selected_class_names = st.sidebar.multiselect('Select classes:', class_names, default=class_names[0])

# Perform object detection based on the selected source
if uploaded_image is not None:
    # Object detection for uploaded image
    image_detect(image=uploaded_image, confidence_threshold=confidence_threshold,
                 max_detections=max_detections, class_ids=selected_class_ids)

elif uploaded_video:
    # Object detection for uploaded video
    video_detect(source='video', uploaded_video=uploaded_video, confidence_threshold=confidence_threshold,
                 max_detections=max_detections, class_ids=selected_class_ids)

    # Remove temporary files
    remove_temp()