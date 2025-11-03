# app.py
import streamlit as st
from PIL import Image
import numpy as np
from model import get_detector

st.title("Emotion Detection App (Pretrained FER Model)")
st.write("Upload an image and the app will detect the emotion")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    detector = get_detector()  # load FER detector

    result = detector.detect_emotions(img_array)

    if result:
        st.write("Detected Emotions:")
        for face in result:
            st.write(face["emotions"])
        dominant_emotion, score = detector.top_emotion(img_array)
        st.write(f"**Dominant Emotion:** {dominant_emotion} ({score:.2f})")
    else:
        st.write("No faces detected.")
