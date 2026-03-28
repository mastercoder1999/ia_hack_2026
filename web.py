import streamlit as st
import numpy as np
import tempfile
import os

from part_1 import predict_file

st.set_page_config(page_title="Whale Classifier", layout="centered")

st.title("Whale Species Classifier")

st.write("Upload a .wav file to classify the species.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save to temp file (librosa needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing..."):
        pred, proba, le = predict_file(tmp_path)

    label = le.inverse_transform([pred])[0]

    st.success(f"Prediction: {label}")

    # Probability display
    st.subheader("Confidence")
    probs = {
        cls: float(p)
        for cls, p in zip(le.classes_, proba)
    }

    st.bar_chart(probs)

    # Clean up temp file
    os.remove(tmp_path)