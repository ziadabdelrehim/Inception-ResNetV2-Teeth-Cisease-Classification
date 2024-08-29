import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('teeth_model.h5')

# Streamlit App
st.title('Teeth Disease Classification')

# Upload an image
uploaded_file = st.file_uploader("Choose a teeth image...", type="jpg")

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0  # normalize
    image = np.expand_dims(image, axis=0)  # add batch dimension

    # Make a prediction
    predictions = model.predict(image)
    class_names = ['Disease 1', 'Disease 2', 'Disease 3', 'Disease 4', 'Disease 5', 'Disease 6', 'Disease 7']
    st.write("Prediction: ", class_names[np.argmax(predictions)])
