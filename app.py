import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import random
import time

# Hide deprecation warnings
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Potato Leaf Disease Detection",
    page_icon=":potato:",
    initial_sidebar_state='auto'
)

# Hide CSS elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load the model using the new caching method
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    start_time = time.time()
    model = tf.keras.models.load_model('model1.h5')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()


# Preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Predict disease
def predict_disease(image):
    processed_image = preprocess_image(image)
    start_time = time.time()
    predictions = model.predict(processed_image)
    st.write(f'Prediction time: {time.time() - start_time:.2f} seconds')
    class_names = ['Early Blight', 'Late Blight', 'Healthy']
    predicted_class = class_names[np.argmax(predictions)]
    predicted_prob = np.max(predictions)
    return predicted_class, predicted_prob

# Remedies for diseases
def get_remedy(disease):
    remedies = {
        'Early Blight': "Apply fungicides such as chlorothalonil or copper-based products. Practice crop rotation and remove infected plant debris.",
        'Late Blight': "Use fungicides like metalaxyl or mefenoxam. Ensure proper drainage and avoid overhead watering.",
        'Healthy': "No treatment needed. Continue regular crop management practices."
    }
    return remedies.get(disease, "No remedy information available.")

# Define the Streamlit app
def main():
    st.title('Potato Leaf Disease Detection')
    st.write('Upload an image of a potato leaf to classify its disease.')

    uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('')
        st.write('Classifying...')

        with st.spinner('Processing...'):
            prediction, prob = predict_disease(image)
        
        st.write(f'Predicted Disease: {prediction} with confidence: {prob*100:.2f}%')
        
        st.sidebar.success(f"Predicted Disease: {prediction}")
        st.sidebar.write(f"Confidence: {prob*100:.2f}%")
        
        st.markdown("## Remedy")
        st.info(get_remedy(prediction))

if __name__ == "__main__":
    main()
