# Nutrivision

Nutrivision is a cutting-edge application leveraging the power of Computer Vision and YOLO v8 to help users make healthy dietary choices. Simply upload an image or video of fruits, and Nutrivision will detect the types of fruits present and suggest custom smoothie recipes tailored to your health goals.

## Features

- **Fruit Detection:** Utilizes YOLO v8 for accurate and fast detection of various fruits from images and videos.
- **Recipe Generation:** Integrates with OpenAI to generate personalized smoothie recipes based on detected fruits.
- **Health Goals:** Offers recipe suggestions based on user-selected health goals (weight loss, maintenance, or gain).

## How It Works

1. **Upload Your Image/Video:** Start by uploading an image or video of the fruits you have.
2. **Detect Fruits:** The app processes the uploaded content to identify and list the fruits.
3. **Get Recipes:** Based on the detected fruits and your selected health goal, receive smoothie recipes tailored just for you.

## Technology Stack
- **Frontend:** Streamlit
- **ML Model:** YOLO v8 for object detection
- **Backend:** Python, OpenAI for recipe generation
