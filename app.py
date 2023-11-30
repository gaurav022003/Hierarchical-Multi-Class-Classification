import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Load your disease prediction model
model = tf.keras.models.load_model('D:/disease-prediction/models/new-model.h5', compile=False)  # Update with your actual model file
class_names=['Tomato_Lateblight', 'Potato_Lateblight', 'Pepperbell_healthy', 'Tomato_Bacterialspot', 'Tomato_SpiderMitesTwoSpottedSpiderMite', 'Tomato_TargetSpot','Tomato_healthy', 'Potato_Earlyblight','Tomato_TomatoYellowLeafCurlVirus','Tomato_Septorialeafspot','Tomato_TomatoMosaicVirus','Tomato_Earlyblight','Pepperbell_Bacterial_spot','Potato_healthy','Tomato_LeafMold']
# Define a function to preprocess the image for prediction
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image)
    image = np.array(image)
    img = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img = np.expand_dims(img, 0)
    return img

# Define a function to make predictions
def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction[1])]

# Streamlit App
st.title("Disease Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = predict_disease(uploaded_file)

    # Display the prediction results
    st.write(f"Prediction: {prediction}")
