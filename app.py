import streamlit as st
from keras.models import load_model
from keras.utils import custom_object_scope
import numpy as np
from PIL import Image

# Define or import your custom layer
class CustomScaleLayer(Layer):
    def __init__(self, scale_factor=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return inputs * self.scale_factor

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({"scale_factor": self.scale_factor})
        return config

# Load model with custom objects
with custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
    model = load_model('teeth_model.h5')

# Streamlit app
st.title('Teeth Disease Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file)
        image = image.resize((256, 256))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(image_array)
        class_index = np.argmax(prediction[0])

        # Display result
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write(f'Predicted class index: {class_index}')
    except Exception as e:
        st.error(f"Error processing image: {e}")
