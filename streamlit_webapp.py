# streamlit_app.py

import requests
import streamlit as st

st.title("Image Processing App")

# Define input fields for image URL, format, and name
image_url = st.text_input("Enter Image URL:")
image_format = st.text_input("Enter Image Format:")
name = st.text_input("Enter Name:")

# Create a radio button to choose between processing and recognition
action = st.radio("Select Action", ["Image Processing", "Image Recognition"])

if st.button("Submit"):
    if action == "Image Processing":
        # Create a dictionary with the input data
        data = {"image_url": image_url, "image_format": image_format, "name": name}

        # Send a POST request to the FastAPI endpoint for image processing
        response = requests.post("http://localhost:8080/process_image/", json=data)

        if response.status_code == 200:
            st.success("Image processed successfully.")
            st.json(response.json())
        else:
            st.error(
                "Error processing image. Please check your input data and try again."
            )

    elif action == "Image Recognition":
        # Send a POST request to the image recognition endpoint
        image_recognition_data = {"image_url": image_url, "image_format": image_format}

        recognition_response = requests.post(
            "http://localhost:8080/recognize_image/", json=image_recognition_data
        )

        if recognition_response.status_code == 200:
            st.success("Image recognized successfully.")
            st.json(recognition_response.json())
        else:
            st.error("Error recognizing image.")
