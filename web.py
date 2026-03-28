import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

from part_1 import (
    predict_file,
    load,
    construire_dataframe,
    evaluer
)

st.set_page_config(page_title="Whale Classifier", layout="wide")

st.title("Whale Species Classification")

@st.cache_resource
def load_model():
    return load()

pipeline, le = load_model()

tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "Evaluation",
    "About"
])

with tab1:
    st.header("Test the model")

    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(uploaded_file)

        with st.spinner("Analyzing..."):
            pred, proba, le = predict_file(tmp_path)

        label = le.inverse_transform([pred])[0]

        st.success(f"Prediction: {label}")

        # Probabilities
        probs = pd.DataFrame({
            "Species": le.classes_,
            "Probability": proba
        }).sort_values("Probability", ascending=False)

        st.subheader("Confidence")
        st.bar_chart(probs.set_index("Species"))

        os.remove(tmp_path)

with tab2:
    st.header("Model Evaluation")

    if st.button("Run evaluation on test set"):
        with st.spinner("Running evaluation..."):
            from part_1 import TEST_DIR

            df_test = construire_dataframe(TEST_DIR)

            X_test = df_test.drop(columns=["label"]).values
            y_test = le.transform(df_test["label"])

            y_pred = pipeline.predict(X_test)

            # Metrics
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                confusion_matrix
            )

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")

            st.subheader("Metrics")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("F1-score", f"{f1:.3f}")
            col3.metric("Precision", f"{precision:.3f}")
            col4.metric("Recall", f"{recall:.3f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d",
                        xticklabels=le.classes_,
                        yticklabels=le.classes_,
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            st.subheader("Confusion Matrix")
            st.pyplot(fig)

with tab3:
    st.header("Project Overview")

    st.write("""
    Maksim Déry, Julien Otis
             
    This project classifies whale species from audio recordings using:

    - Feature extraction with Librosa
    - MFCC, spectral features, chroma, RMS
    - Machine learning models (SVM, Random Forest, etc.)
    - Best model selected via cross-validation

    Pipeline:
    1. Audio → Features
    2. Features → Model
    3. Model → Prediction
    """)

    st.write("Classes:")
    st.write(list(le.classes_))