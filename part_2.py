import os
import sys
import warnings
import pickle

import matplotlib
matplotlib.use("Agg")  # backend non-GUI (safe pour scripts)

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    jaccard_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# Paths & constants
# TRAIN_DIR = os.path.join("data", "part_2", "train")
# TEST_DIR = os.path.join("data", "part_2", "test")
# LONG_AUDIO_DIR = os.path.join("data", "long_audio", "audio")

SAMPLE_RATE = 22050
N_MFCC = 13
RANDOM_STATE = 69

OUTPUT_MODEL = "meilleur_modele_part2.pkl"
OUTPUT_ENCODER = "label_encoder_part2.pkl"

ANNOTATIONS_DIR = "annotations_after_training"

NOISE_LABEL = "Bruit"

LABEL_MAP = {
    "Beluga_WhiteWhale": "Béluga",
    "Fin_FinbackWhale": "Rorqual commun",
    "HumpbackWhale": "Baleine à bosse",
    "SpermWhale": "Cachalot",
    "White_sidedDolphin": "Dauphin à flancs blancs",
}

# Sliding-window settings
WINDOW_SEC = 1.5
HOP_SEC = 0.25
MIN_CONF = 0.5
MIN_CALL_SEC = 1.0


# Feature extraction
def extraire_features(path: str = None, y: np.ndarray = None, sr: int = SAMPLE_RATE) -> np.ndarray:
    if y is None:
        y, sr = librosa.load(path, sr=sr, mono=True)

    def stats(x):
        mean = np.mean(x, axis=1) if x.ndim > 1 else np.array([np.mean(x)])
        std = np.std(x, axis=1) if x.ndim > 1 else np.array([np.std(x)])
        return np.concatenate([mean, std])

    feats = []

    # MFCC + temporal dynamics
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feats.extend(stats(mfcc))
    feats.extend(stats(delta))
    feats.extend(stats(delta2))

    # spectral features
    feats.extend(stats(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats.extend(stats(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats.extend(stats(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feats.extend(stats(librosa.feature.spectral_contrast(y=y, sr=sr)))

    # mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    feats.extend(stats(mel))

    # temporal / energy
    feats.extend(stats(librosa.feature.zero_crossing_rate(y)))
    feats.extend(stats(librosa.feature.rms(y=y)))

    return np.array(feats)


# Dataset construction
def lister_audio_files(root):
    for specie in sorted(os.listdir(root)):
        d = os.path.join(root, specie)
        if not os.path.isdir(d):
            continue

        files = [
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(".wav")
        ]
        yield specie, files


def construire_dataframe_windows(root, window_sec=WINDOW_SEC, hop_sec=HOP_SEC):
    X, y = [], []

    win_samples = int(window_sec * SAMPLE_RATE)
    hop_samples = int(hop_sec * SAMPLE_RATE)

    for specie, files in lister_audio_files(root):
        label = LABEL_MAP.get(specie, NOISE_LABEL)

        for f in files:
            try:
                audio, sr = librosa.load(f, sr=SAMPLE_RATE)

                for start in range(0, len(audio) - win_samples + 1, hop_samples):
                    chunk = audio[start:start + win_samples]
                    energy = np.mean(np.abs(chunk))

                    # Explicit silence modeling + sous-échantillonnage du bruit
                    if energy < 0.01:
                        if np.random.rand() < 0.3:
                            X.append(extraire_features(y=chunk, sr=sr))
                            y.append(NOISE_LABEL)
                    else:
                        X.append(extraire_features(y=chunk, sr=sr))
                        y.append(label)

            except Exception as e:
                print(f"[ERREUR] {f}: {e}")

    df = pd.DataFrame(X)
    df["label"] = y
    return df


def get_models():
    return {
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=50,
            gamma=0.01,
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def evaluate_models(models, X, y):
    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=5, scoring="jaccard_macro")
        results[name] = {
            "pipeline": pipe,
            "f1": scores.mean(),
            "std": scores.std(),
        }
        print(f"  {name:25s}  Jaccard={scores.mean():.4f} ± {scores.std():.4f}")
    return results


def select_best(results):
    name = max(results, key=lambda k: results[k]["f1"])
    return name, results[name]["pipeline"]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "jaccard_macro": jaccard_score(y_true, y_pred, average="macro"),
    }


def plot_confusion(cm, classes, model_name):
    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )

    plt.title(f"Matrice de confusion – {model_name}")
    plt.ylabel("Vrai label")
    plt.xlabel("Prédit")

    plt.tight_layout()

    plt.show()


def plot_importance(clf):
    if not hasattr(clf, "feature_importances_"):
        return

    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:20]

    plt.figure(figsize=(10, 4))
    plt.bar(range(20), imp[idx])
    plt.xticks(range(20), idx, rotation=45)
    plt.title("Top-20 feature importances")
    plt.tight_layout()
    plt.savefig("feature_importance_part2.png", dpi=150)
    plt.close()


def evaluer(pipeline, X, y, le, name):
    y_pred = pipeline.predict(X)
    metrics = compute_metrics(y, y_pred)

    print(f"\n{'=' * 55}")
    print(f"  Meilleur modèle : {name}")
    print(f"  Accuracy={metrics['accuracy']:.4f}  |  Jaccard-macro={metrics['jaccard_macro']:.4f}")
    print(classification_report(y, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y, y_pred, labels=range(len(le.classes_)))
    plot_confusion(cm, le.classes_, name)
    plot_importance(pipeline.named_steps["clf"])


def save(pipeline, le):
    pickle.dump(pipeline, open(OUTPUT_MODEL, "wb"))
    pickle.dump(le, open(OUTPUT_ENCODER, "wb"))


def load():
    return (
        pickle.load(open(OUTPUT_MODEL, "rb")),
        pickle.load(open(OUTPUT_ENCODER, "rb")),
    )


# Sliding window detection on long audio
def _windows(y: np.ndarray, sr: int, window_sec: float = WINDOW_SEC, hop_sec: float = HOP_SEC):
    win_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    n_samples = len(y)

    start = 0
    while start + win_samples <= n_samples:
        chunk = y[start:start + win_samples]
        t_start = start / sr
        t_end = (start + win_samples) / sr
        yield t_start, t_end, chunk
        start += hop_samples

    if start < n_samples:
        chunk = y[start:]
        t_start = start / sr
        t_end = n_samples / sr
        chunk = np.pad(chunk, (0, win_samples - len(chunk)))
        yield t_start, t_end, chunk


def smooth_predictions(detections, window_size=3):
    if not detections:
        return detections

    smoothed = []
    n = len(detections)

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)

        window = detections[start:end]
        labels = [d["label"] for d in window]
        confidences = [d["confidence"] for d in window]

        label = max(
            set(labels),
            key=lambda l: sum(c for l2, c in zip(labels, confidences) if l2 == l)
        )
        conf = np.mean(confidences)

        smoothed.append({
            **detections[i],
            "label": label,
            "confidence": conf,
            "confidences": [conf],
        })

    return smoothed


def _merge_detections(detections, min_gap_sec=MIN_CALL_SEC):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x["t_start"])

    merged = []
    cur = dict(detections[0])

    for d in detections[1:]:
        same_label = d["label"] == cur["label"]
        gap = d["t_start"] - cur["t_end"]
        overlap = not (d["t_start"] >= cur["t_end"] or d["t_end"] <= cur["t_start"])

        if overlap and d["label"] != cur["label"]:
            continue

        if same_label and gap <= min_gap_sec:
            cur["t_end"] = max(cur["t_end"], d["t_end"])
            cur["confidences"].append(d["confidence"])
        else:
            merged.append(cur)
            cur = dict(d)

    merged.append(cur)

    results = []
    for seg in merged:
        confs = seg.get("confidences", [seg["confidence"]])
        results.append({
            "species": seg["label"],
            "t_start": round(seg["t_start"], 2),
            "t_end": round(seg["t_end"], 2),
            "duration": round(seg["t_end"] - seg["t_start"], 2),
            "mean_confidence": round(float(np.mean(confs)), 3),
        })

    return results


def detect_long_audio(
    path: str,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC,
    min_conf: float = MIN_CONF,
    min_call_sec: float = MIN_CALL_SEC,
    verbose: bool = True,
):
    pipeline, le = load()

    if verbose:
        print(f"\n[Détection] {path}")
        print(f"  fenêtre={window_sec}s | hop={hop_sec}s | seuil={min_conf}")

    y_full, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    total_sec = len(y_full) / sr

    if verbose:
        print(f"  Durée totale : {total_sec:.1f}s ({total_sec / 60:.2f} min)")

    raw_detections = []

    for t_start, t_end, chunk in _windows(y_full, sr, window_sec, hop_sec):
        try:
            feats = extraire_features(y=chunk, sr=sr).reshape(1, -1)
        except Exception as e:
            if verbose:
                print(f"  [ERREUR feature] t={t_start:.1f}s : {e}")
            continue

        proba = pipeline.predict_proba(feats)[0]

        if verbose and np.random.rand() < 0.01:
            print({le.classes_[i]: round(p, 3) for i, p in enumerate(proba)})

        sorted_proba = np.sort(proba)[::-1]
        margin = sorted_proba[0] - sorted_proba[1]

        label_idx = np.argmax(proba)
        confidence = float(proba[label_idx])
        label = le.inverse_transform([label_idx])[0]

        energy = np.mean(np.abs(chunk))
        if energy < 0.01:
            continue

        if label == NOISE_LABEL:
            continue

        if confidence < min_conf or margin < 0.25:
            continue

        raw_detections.append({
            "t_start": t_start,
            "t_end": t_end,
            "label": label,
            "confidence": confidence,
            "confidences": [confidence],
        })

    if verbose:
        print(f"  Fenêtres retenues : {len(raw_detections)}")

    raw_detections = smooth_predictions(raw_detections)
    filtered = [d for d in raw_detections if d["label"] != "Inconnu"]
    merged = _merge_detections(filtered, min_gap_sec=min_call_sec)

    df = pd.DataFrame(
        merged,
        columns=["species", "t_start", "t_end", "duration", "mean_confidence"]
    ) if merged else pd.DataFrame(
        columns=["species", "t_start", "t_end", "duration", "mean_confidence"]
    )

    df = df[df["duration"] >= 1.0]
    return df


# Timeline plot
_SPECIES_COLOURS = {
    "Béluga": "#4e9af1",
    "Rorqual commun": "#f4a261",
    "Baleine à bosse": "#2a9d8f",
    "Cachalot": "#e76f51",
    "Dauphin à flancs blancs": "#8ecae6",
}
_DEFAULT_COLOUR = "#aaa"


def plot_timeline(df: pd.DataFrame, audio_path: str, total_sec: float = None, save_path: str = "timeline.png"):
    if df.empty:
        return

    if total_sec is None:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        total_sec = len(y) / sr

    species_list = df["species"].unique().tolist()
    y_pos = {sp: i for i, sp in enumerate(species_list)}

    fig, ax = plt.subplots(figsize=(14, max(3, len(species_list) * 1.4)))

    for _, row in df.iterrows():
        colour = _SPECIES_COLOURS.get(row["species"], _DEFAULT_COLOUR)
        ax.barh(
            y=y_pos[row["species"]],
            width=row["duration"],
            left=row["t_start"],
            height=0.6,
            color=colour,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        if row["duration"] > total_sec * 0.015:
            ax.text(
                row["t_start"] + row["duration"] / 2,
                y_pos[row["species"]],
                f"{row['duration']:.1f}s\n({row['mean_confidence']:.0%})",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    ax.set_xlim(0, total_sec)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))

    max_min = int(total_sec // 60) + 1
    tick_secs = [m * 60 for m in range(max_min + 1) if m * 60 <= total_sec]
    ax.set_xticks(tick_secs)
    ax.set_xticklabels([f"{t // 60:.0f}:{t % 60:02.0f}" for t in tick_secs])

    ax.set_xlabel("Temps (mm:ss)")
    ax.set_title(f"Détection d'appels – {os.path.basename(audio_path)}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    patches = [
        mpatches.Patch(color=_SPECIES_COLOURS.get(sp, _DEFAULT_COLOUR), label=sp)
        for sp in species_list
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def afficher_resultats(df: pd.DataFrame):
    if df.empty:
        print("\n  Aucun appel détecté dans cet enregistrement.")
        return

    print(f"\n{'─' * 65}")
    print(f"  {'Espèce':<30} {'Début':>7} {'Fin':>7} {'Durée':>7} {'Confiance':>9}")
    print(f"{'─' * 65}")

    for _, r in df.iterrows():
        def fmt(s):
            m, sec = divmod(s, 60)
            return f"{int(m)}:{sec:05.2f}"

        print(
            f"  {r['species']:<30} {fmt(r['t_start']):>7} {fmt(r['t_end']):>7}"
            f" {r['duration']:>6.1f}s {r['mean_confidence']:>8.1%}"
        )

    print(f"{'─' * 65}")
    print(f"  Total : {len(df)} animaux détectés\n")


def main():

    if len(sys.argv) != 4:
        print("Usage: python3 part_2.py <train_dir> <test_dir> <long_audio_dir>")
        sys.exit(1)

    TRAIN_DIR = sys.argv[1]
    TEST_DIR = sys.argv[2]
    LONG_AUDIO_DIR = sys.argv[3]

    print("\n[1] Chargement des données")
    df_train = construire_dataframe_windows(TRAIN_DIR)
    df_test = construire_dataframe_windows(TEST_DIR)

    le = LabelEncoder()
    le.fit(df_train["label"])

    X_train = df_train.drop(columns=["label"]).values
    y_train = le.transform(df_train["label"])
    X_test = df_test.drop(columns=["label"]).values
    y_test = le.transform(df_test["label"])

    print(f"\n  Classes : {list(le.classes_)}")
    print(f"  Train : {X_train.shape[0]} échantillons | Test : {X_test.shape[0]}")

    print("\n[2] Entraînement (cross-validation 5-fold)")
    models = get_models()
    results = evaluate_models(models, X_train, y_train)

    best_name, best_pipe = select_best(results)
    print(f"\n  Meilleur modèle : {best_name}")
    best_pipe.fit(X_train, y_train)

    print("\n[3] Évaluation sur le jeu de test")
    evaluer(best_pipe, X_test, y_test, le, best_name)

    save(best_pipe, le)

    if not os.path.isdir(LONG_AUDIO_DIR):
        print(f"\n[5] (Passer – dossier non trouvé : {LONG_AUDIO_DIR})")
        return

    wav_files = sorted([
        os.path.join(LONG_AUDIO_DIR, f)
        for f in os.listdir(LONG_AUDIO_DIR)
        if f.lower().endswith(".wav")
    ])

    if not wav_files:
        print(f"\n[5] (Passer – aucun fichier .wav dans {LONG_AUDIO_DIR})")
        return

    print(f"\n[5] Détection sur {len(wav_files)} séquence(s) audio longue(s)")

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    all_detections = []

    for audio_path in wav_files:
        df_results = detect_long_audio(audio_path)
        afficher_resultats(df_results)

        if not df_results.empty:
            df_results.insert(0, "fichier", os.path.basename(audio_path))
            all_detections.append(df_results)

            stem = os.path.splitext(os.path.basename(audio_path))[0]
            csv_path = os.path.join(ANNOTATIONS_DIR, f"{stem}_detections.csv")
            df_results.to_csv(csv_path, index=False)

            timeline_path = os.path.join(ANNOTATIONS_DIR, f"{stem}_timeline.png")
            plot_timeline(df_results, audio_path, save_path=timeline_path)

    if all_detections:
        merged_csv = os.path.join(ANNOTATIONS_DIR, "all_detections.csv")
        pd.concat(all_detections, ignore_index=True).to_csv(merged_csv, index=False)
        print(f"\n  → {ANNOTATIONS_DIR}/ contient {len(all_detections)} CSV(s) + all_detections.csv")


if __name__ == "__main__":
    main()