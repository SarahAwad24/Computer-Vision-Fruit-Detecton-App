import torch
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms as T

# Load your model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Adjust path if necessary
model.eval()

# Define your classes
classes = [
    "Apple", "Banana", "Beetroot", "Bitter_Gourd", "Bottle_Gourd", "Cabbage",
    "Capsicum", "Carrot", "Cauliflower", "Cherry", "Chilli", "Coconut",
    "Cucumber", "EggPlant", "Ginger", "Grape", "Green_Orange", "Kiwi",
    "Maize", "Mango", "Melon", "Okra", "Onion", "Orange", "Peach", "Pear",
    "Peas", "Pineapple", "Pomegranate", "Potato", "Radish", "Strawberry",
    "Tomato", "Turnip", "Watermelon", "walnut", "almond"
]

# Define the inference function
def detect(image):
    # Transform the image to tensor
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        detections = model(input_tensor)[0]

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(image)
    for detection in detections:
        # Each detection includes [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, conf, cls = detection
        if conf >= 0.5:  # Consider detections with confidence >= 0.5
            label = classes[int(cls)]
            draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
            draw.text((x1, y1), f"{label} ({conf:.2f})", fill="red")

    return np.array(image)

# Create a Gradio interface
iface = gr.Interface(
    fn=detect,
    inputs=gr.inputs.Image(source="webcam", tool="editor"),
    outputs="image",
    live=True,
)

# Launch the app
iface.launch()
