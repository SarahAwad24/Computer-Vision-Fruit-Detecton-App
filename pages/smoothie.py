import streamlit as st
import joblib
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI'))

st.title('Create your Smoothie!')

goal_dicc = {
    '1': 'lose weight',
    '2': 'gain weight',
    '3': 'maintain weight'
}

try:
    fruits = joblib.load('fruits.joblib')
    print(fruits)
except Exception:
    print('Error')
goal = st.radio("Select your goal", list(goal_dicc.keys()), format_func=lambda x: goal_dicc[x], horizontal=True)
print(goal)

prompt = f"""
        Objective: Generate smoothie recipes based on the user's dietary goal and a list of ingredients recognized by the Nutivision application.

    Input Format:
    The input will be a list of ingredients identified by the Nutivision application, followed by a number representing the user's dietary goal:
    - The user's goal is to {goal_dicc[goal]}
    - The ingredients identified are: {fruits}

    The fruit list will be in the format: ['ingredient 1', 'ingredient n']

    Step 1: Interpretation of Application Output
    - Receive and interpret the ingredient list and user goal from the Nutivision application output.

    Step 2: Recipe Generation
    - Using the provided ingredient list, create two unique smoothie recipes that align with the user's dietary goal. For each recipe:
      - Include a variety of base ingredients from the list.
      - Promote the user's goal of either losing, gaining, or maintaining weight.
      - Clearly mark the start and end of each recipe using the markers:
        "---Smoothie Recipe 1 Start---" and "---Smoothie Recipe 1 End---" for the first recipe.
        "---Smoothie Recipe 2 Start---" and "---Smoothie Recipe 2 End---" for the second recipe.

    Step 3: Additional Ingredient Suggestion
    - Suggest one additional ingredient per smoothie that complements the user's goal, which is not present in the initial list. Provide options, such as various types of protein if the goal is weight gain. Clearly mark the suggestions using the markers:
        "---Additional Ingredient Recommendation 1 Start---" and "---Additional Ingredient Recommendation 1 End---" for the first smoothie.
        "---Additional Ingredient Recommendation 2 Start---" and "---Additional Ingredient Recommendation 2 End---" for the second smoothie.

    Step 4: Recipe Detailing and Nutritional Content
    - Provide precise measurements for each ingredient (e.g., 1/2 cup, 1 full banana).
    - Estimate the nutritional content (carbs, proteins, and fats) for each smoothie.
    - Clearly delineate the nutritional content using the markers:
        "---Nutritional Content 1 Start---" and "---Nutritional Content 1 End---" for the first smoothie.
        "---Nutritional Content 2 Start---" and "---Nutritional Content 2 End---" for the second smoothie.


    Available liquids include:
    water, 2% milk, skim milk, Greek yogurt, whole milk, vanilla ice cream

    Using the existing ingredients, generate two smoothie recipes tailored to weight loss. For each recipe, also suggest an additional ingredient that supports the dietary goal, offering options such as different types of protein, and specify the suggested amount in grams. Include markers for each section of the output for easy extraction by a Python script.
    """
try:
    if st.button('Create Smoothie'):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        recommendation = response.choices[0].message.content
        print(recommendation)
        st.text_area(
            'Smootie Recipe',
            recommendation,
            1200
        )
except Exception as e:
    recommendation = f"Failed to generate recommendation. Please try again. {e}"


