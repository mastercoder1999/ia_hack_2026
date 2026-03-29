import os
import tempfile

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    jaccard_score,
)

from part_1 import (
    predict_file as predict_file_part1,
    load as load_part1,
    construire_dataframe as construire_dataframe_part1,
)

from part_2 import (
    load as load_part2,
    construire_dataframe_windows as construire_dataframe_part2,
    detect_long_audio,
    plot_timeline,
)

st.set_page_config(page_title="Whale Audio Presentation", layout="wide")
st.title("Whale Audio Classification and Detection")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Configuration")

part1_test_dir = st.sidebar.text_input(
    "Part 1 - Test directory",
    value="data/part_1/test"
)

part2_test_dir = st.sidebar.text_input(
    "Part 2 - Test directory",
    value="data/part_2/test"
)

part2_long_audio_dir = st.sidebar.text_input(
    "Part 2 - Long audio directory",
    value="data/long_audio/audio"
)

st.sidebar.markdown("---")
st.sidebar.write("Make sure the models are already trained and saved:")
st.sidebar.code(
    "meilleur_modele_part1.pkl\n"
    "label_encoder_part1.pkl\n"
    "meilleur_modele_part2.pkl\n"
    "label_encoder_part2.pkl"
)

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource
def get_part1_model():
    return load_part1()

@st.cache_resource
def get_part2_model():
    return load_part2()

def safe_load_part1():
    try:
        return get_part1_model()
    except Exception as e:
        st.error(f"Impossible de charger le modèle Partie 1 : {e}")
        return None, None

def safe_load_part2():
    try:
        return get_part2_model()
    except Exception as e:
        st.error(f"Impossible de charger le modèle Partie 2 : {e}")
        return None, None

pipeline1, le1 = safe_load_part1()
pipeline2, le2 = safe_load_part2()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Partie 1", "Partie 2", "À propos"])

# =========================================================
# PARTIE 1
# =========================================================
with tab1:
    st.header("Partie 1 - Classification d'un extrait audio")

    st.subheader("Prédiction sur un fichier WAV")
    uploaded_file_part1 = st.file_uploader(
        "Uploader un fichier WAV (partie 1)",
        type=["wav"],
        key="part1_uploader"
    )

    if uploaded_file_part1 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file_part1.read())
            tmp_path = tmp.name

        st.audio(uploaded_file_part1)

        if pipeline1 is not None and le1 is not None:
            with st.spinner("Analyse en cours..."):
                pred, proba, label_encoder = predict_file_part1(tmp_path)

            predicted_label = pred
            if not isinstance(pred, str):
                predicted_label = label_encoder.inverse_transform([pred])[0]

            st.success(f"Prédiction : {predicted_label}")

            probs = pd.DataFrame({
                "Espèce": label_encoder.classes_,
                "Probabilité": proba
            }).sort_values("Probabilité", ascending=False)

            st.subheader("Confiance par classe")
            st.bar_chart(probs.set_index("Espèce"))

        os.remove(tmp_path)

    st.markdown("---")
    st.subheader("Évaluation sur le dossier de test")

    if st.button("Lancer l'évaluation Partie 1"):
        if pipeline1 is None or le1 is None:
            st.error("Le modèle Partie 1 n'est pas disponible.")
        elif not os.path.isdir(part1_test_dir):
            st.error(f"Dossier invalide : {part1_test_dir}")
        else:
            with st.spinner("Évaluation Partie 1 en cours..."):
                df_test = construire_dataframe_part1(part1_test_dir)

                X_test = df_test.drop(columns=["label"]).values
                y_test = le1.transform(df_test["label"])
                y_pred = pipeline1.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")
                precision = precision_score(y_test, y_pred, average="macro")
                recall = recall_score(y_test, y_pred, average="macro")

                st.subheader("Métriques")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("F1 macro", f"{f1:.3f}")
                c3.metric("Precision macro", f"{precision:.3f}")
                c4.metric("Recall macro", f"{recall:.3f}")

                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    xticklabels=le1.classes_,
                    yticklabels=le1.classes_,
                    ax=ax
                )
                ax.set_xlabel("Prédit")
                ax.set_ylabel("Vrai")
                ax.set_title("Matrice de confusion - Partie 1")
                st.pyplot(fig)

# =========================================================
# PARTIE 2
# =========================================================
with tab2:
    st.header("Partie 2 - Détection dans un audio long")

    st.subheader("Détection sur un fichier WAV long")
    uploaded_file_part2 = st.file_uploader(
        "Uploader un fichier WAV long (partie 2)",
        type=["wav"],
        key="part2_uploader"
    )

    if uploaded_file_part2 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file_part2.read())
            tmp_path = tmp.name

        st.audio(uploaded_file_part2)

        if pipeline2 is not None and le2 is not None:
            with st.spinner("Détection en cours..."):
                df_detect = detect_long_audio(tmp_path, verbose=False)

            if df_detect.empty:
                st.warning("Aucun appel détecté.")
            else:
                st.subheader("Détections")
                st.dataframe(df_detect, use_container_width=True)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
                    timeline_path = img_tmp.name

                plot_timeline(df_detect, tmp_path, save_path=timeline_path)
                st.subheader("Timeline")
                st.image(timeline_path, use_container_width=True)

                if os.path.exists(timeline_path):
                    os.remove(timeline_path)

        os.remove(tmp_path)

    st.markdown("---")
    st.subheader("Évaluation sur le dossier de test de la partie 2")

    if st.button("Lancer l'évaluation Partie 2"):
        if pipeline2 is None or le2 is None:
            st.error("Le modèle Partie 2 n'est pas disponible.")
        elif not os.path.isdir(part2_test_dir):
            st.error(f"Dossier invalide : {part2_test_dir}")
        else:
            with st.spinner("Évaluation Partie 2 en cours..."):
                df_test = construire_dataframe_part2(part2_test_dir)

                X_test = df_test.drop(columns=["label"]).values
                y_test = le2.transform(df_test["label"])
                y_pred = pipeline2.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                jacc = jaccard_score(y_test, y_pred, average="macro")
                precision = precision_score(y_test, y_pred, average="macro")
                recall = recall_score(y_test, y_pred, average="macro")

                st.subheader("Métriques")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Jaccard macro", f"{jacc:.3f}")
                c3.metric("Precision macro", f"{precision:.3f}")
                c4.metric("Recall macro", f"{recall:.3f}")

                cm = confusion_matrix(y_test, y_pred, labels=range(len(le2.classes_)))

                fig, ax = plt.subplots(figsize=(9, 7))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    xticklabels=le2.classes_,
                    yticklabels=le2.classes_,
                    ax=ax,
                    cmap="Blues"
                )
                ax.set_xlabel("Prédit")
                ax.set_ylabel("Vrai")
                ax.set_title("Matrice de confusion - Partie 2")
                st.pyplot(fig)

    st.markdown("---")
    st.subheader("Démo batch sur le dossier d'audios longs")

    if st.button("Analyser quelques fichiers du dossier d'audios longs"):
        if pipeline2 is None or le2 is None:
            st.error("Le modèle Partie 2 n'est pas disponible.")
        elif not os.path.isdir(part2_long_audio_dir):
            st.error(f"Dossier invalide : {part2_long_audio_dir}")
        else:
            wav_files = sorted([
                os.path.join(part2_long_audio_dir, f)
                for f in os.listdir(part2_long_audio_dir)
                if f.lower().endswith(".wav")
            ])

            if not wav_files:
                st.warning("Aucun fichier WAV trouvé.")
            else:
                max_files = min(3, len(wav_files))
                st.info(f"Présentation sur {max_files} fichier(s).")

                for audio_path in wav_files[:max_files]:
                    st.markdown(f"### {os.path.basename(audio_path)}")

                    df_detect = detect_long_audio(audio_path, verbose=False)

                    if df_detect.empty:
                        st.write("Aucun appel détecté.")
                        continue

                    st.dataframe(df_detect, use_container_width=True)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
                        timeline_path = img_tmp.name

                    plot_timeline(df_detect, audio_path, save_path=timeline_path)
                    st.image(timeline_path, use_container_width=True)

                    if os.path.exists(timeline_path):
                        os.remove(timeline_path)

# =========================================================
# ABOUT
# =========================================================
with tab3:
    st.header("À propos du projet")

    st.write("""
    **Auteurs :** Maksim Déry, Julien Otis

    Cette application présente deux volets du projet :

    **Partie 1**
    - Classification d'un extrait audio court
    - Extraction de features avec Librosa
    - Comparaison de plusieurs modèles
    - Sélection du meilleur modèle par validation croisée

    **Partie 2**
    - Détection d'appels dans un audio long
    - Analyse par fenêtres glissantes
    - Fusion des détections
    - Visualisation sur une timeline
    """)

    if le1 is not None:
        st.subheader("Classes - Partie 1")
        st.write(list(le1.classes_))

    if le2 is not None:
        st.subheader("Classes - Partie 2")
        st.write(list(le2.classes_))