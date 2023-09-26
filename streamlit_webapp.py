# streamlit_app.py

import requests
import streamlit as st

st.title("Deep Facial Recognition App :sunglasses:")
st.text("https://github.com/gabrieltardochi/full-stack-deep-facial-recognition")
st.subheader("", divider="rainbow")

#  input fields for image URL, format, and name
image_url = st.text_input("Enter Image URL:")
image_format = st.text_input("Enter Image Format:")
name = st.text_input("Enter Name (only needed if indexing):")

# radio button to choose between processing and recognition
action = st.radio("Select Action", ["Recognize Face", "Index Person's Face"])

if st.button("Submit"):
    if action == "Recognize Face":
        image_recognition_data = {"image_url": image_url, "image_format": image_format}

        recognition_response = requests.post(
            "http://localhost:8080/api/v1/recognize", json=image_recognition_data
        )

        if recognition_response.status_code == 200:
            st.success("Image recognition ran successfully.")
            st.json(recognition_response.json())
        else:
            st.error("Error trying to recognizing image.")
    elif action == "Index Person's Face":
        data = {"image_url": image_url, "image_format": image_format, "name": name}

        response = requests.post("http://localhost:8080/api/v1/index", json=data)

        if response.status_code == 200:
            st.success("Image indexed successfully.")
            st.json(response.json())
        else:
            st.error(
                "Error indexing image. Please check your input data and try again."
            )
