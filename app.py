import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Page settings
st.set_page_config(
    page_title="Multiple Sclerosis Detection",
    page_icon="🧠",
    layout="centered"
)

# Load model with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/ms_model.h5")
    return model

model = load_model()

# Class names
class_names = ["MS", "No_MS"]

# Title
st.title("🧠 Multiple Sclerosis Detection using Deep Learning")

st.write(
"""
Upload a **Brain MRI image** and the AI model will detect whether
**Multiple Sclerosis (MS)** is present or not.
"""
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # Preprocess image
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Analyzing MRI image..."):
        prediction = model.predict(img_array)

    index = np.argmax(prediction)
    confidence = prediction[0][index]

    st.subheader("Prediction Result")

    if class_names[index] == "MS":
        st.error("⚠ Multiple Sclerosis Detected")
    else:
        st.success("✅ No Multiple Sclerosis Detected")

    # Confidence
    st.subheader("Confidence Score")

    confidence_percent = round(confidence * 100, 2)

    st.progress(int(confidence_percent))
    st.write(f"{confidence_percent}%")

    # Probability chart
    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame(
        prediction,
        columns=class_names
    )

    st.bar_chart(prob_df.T)

else:
    st.info("Please upload an MRI image to start detection.")