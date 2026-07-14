import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_cnn_v1.keras")

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title(" Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Preprocess to match training pipeline: 224x224, RGB, rescaled 0-1
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show probability breakdown for all classes
    st.write("### Class probabilities")
    for cls, prob in zip(class_names, predictions):
        st.write(f"{cls}: {prob*100:.2f}%")
        st.progress(float(prob))