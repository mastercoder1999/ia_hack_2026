import os
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

TRAIN_DIR = os.path.join("data", "part_2", "train")
TEST_DIR = os.path.join("data", "part_2", "test")

SAMPLE_RATE = 22050
N_MFCC = 13
RANDOM_STATE = 69

OUTPUT_MODEL = "meilleur_modele_part2.pkl"
OUTPUT_ENCODER = "label_encoder_part2.pkl"

# folder where all per-sequence CSVs and the merged CSV will be saved
ANNOTATIONS_DIR = "annotations_after_training"

NOISE_LABEL = "Bruit"

LABEL_MAP = {
    "Beluga_WhiteWhale": "Béluga",
    "Fin_FinbackWhale": "Rorqual commun",
    "HumpbackWhale": "Baleine à bosse",
    "SpermWhale": "Cachalot",
    "White_sidedDolphin": "Dauphin à flancs blancs",
}
# Any folder not listed above is treated as noise.

# Sliding-window settings for long audio detection

WINDOW_SEC = 1       # length of each analysis window (seconds)
HOP_SEC = 0.25       # step between consecutive windows (seconds) 0.5
MIN_CONF = 0.5       # minimum confidence to accept a prediction (not noise) 0.35
MIN_CALL_SEC = 0.5   # merge gaps shorter than this (seconds) within the same species

# folder that contains the long audio sequences to analyse
LONG_AUDIO_DIR = os.path.join("data", "long_audio", "audio")


# Feature extraction
def extraire_features(path: str = None, y: np.ndarray = None, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract the 58-dimensional feature vector from either:
      - a file path  (pass `path=`)
      - a raw numpy array already loaded (pass `y=` and `sr=`)
    """
    if y is None:
        y, sr = librosa.load(path, sr=sr, mono=True)

    # np.mean = the average value of each feature over the entire clip, np.std = the standard deviation, capturing how much each feature varies over time
    def stats(x):
        mean = np.mean(x, axis=1) if x.ndim > 1 else np.array([np.mean(x)])
        std = np.std(x, axis=1) if x.ndim > 1 else np.array([np.std(x)])
        return np.concatenate([mean, std])

    feats = []

    # computing the MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # computing the features of the MFCCs (mean and std) and adding them to the list of features
    feats.extend(stats(mfcc))

    for func in [
        # "brightness" of the sound, whale low/dolphin high
        librosa.feature.spectral_centroid,
        # how spread out the frequencies are around that centroid, Narrow = tonal/pure tone, wide = noisy/complex
        librosa.feature.spectral_bandwidth,
    ]:
        feats.extend(stats(func(y=y, sr=sr)))

    # how often the signal flips from 0 to 1 over time
    feats.extend(stats(librosa.feature.zero_crossing_rate(y)))
    # how much loudness varies over time
    feats.extend(stats(librosa.feature.rms(y=y)))
    # 12 pitch classes tracked over time
    feats.extend(stats(librosa.feature.chroma_stft(y=y, sr=sr)))

    feats = np.array(feats)
    assert feats.shape[0] == 58, f"Feature size mismatch: {feats.shape}"
    # its done and all in one place
    return feats


# Dataset construction

def lister_audio_files(root):
    # Looping over species folders
    for specie in sorted(os.listdir(root)):
        # building the full path to the respective species folder
        d = os.path.join(root, specie)
        # skips over non important folders
        if not os.path.isdir(d):
            continue
        # collecting .wav files
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.lower().endswith(".wav")]
        yield specie, files


def construire_dataframe(root):
    X, y = [], []
    for specie, files in lister_audio_files(root):
        label = LABEL_MAP.get(specie, NOISE_LABEL)
        for f in files:
            try:
                X.append(extraire_features(path=f))
                y.append(label)
            except Exception as e:
                print(f"  [ERREUR] {f}: {e}")

    df = pd.DataFrame(X)
    df["label"] = y
    return df
def construire_dataframe_avec_contexte(root, window_sec=1.5, n_augmented=5):
    X, y = [], []
    for specie, files in lister_audio_files(root):
        label = LABEL_MAP.get(specie, NOISE_LABEL)
        for f in files:
            try:
                clip, sr = librosa.load(f, sr=SAMPLE_RATE, mono=True)
                win_samples = int(window_sec * sr)

                # Original clip sans padding
                X.append(extraire_features(y=clip, sr=sr))
                y.append(label)

                # n_augmented versions avec position aléatoire dans la fenêtre
                for _ in range(n_augmented):
                    max_offset = max(0, win_samples - len(clip))
                    offset = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
                    window = np.zeros(win_samples)
                    end = min(offset + len(clip), win_samples)
                    window[offset:end] = clip[:end - offset]
                    X.append(extraire_features(y=window, sr=sr))
                    y.append(label)
            except Exception as e:
                print(f"  [ERREUR] {f}: {e}")

    df = pd.DataFrame(X)
    df["label"] = y

    return df

def get_models():
    return {
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
    }


def evaluate_models(models, X, y):
    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=5, scoring="jaccard_macro")
        results[name] = {"pipeline": pipe, "f1": scores.mean(), "std": scores.std()}
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
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues")
    plt.title(f"Matrice de confusion – {model_name}")
    plt.ylabel("Vrai label")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig("confusion_matrix_part2.png", dpi=150)
    plt.close()


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
    print(f"\n{'='*55}")
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
        chunk = y[start : start + win_samples]
        t_start = start / sr
        t_end = (start + win_samples) / sr
        yield t_start, t_end, chunk
        start += hop_samples

    # last partial window (if anything is left)
    if start < n_samples:
        chunk = y[start:]
        t_start = start / sr
        t_end = n_samples / sr
        # zero-pad to guarantee feature extraction works
        chunk = np.pad(chunk, (0, win_samples - len(chunk)))
        yield t_start, t_end, chunk


def _merge_detections(detections, min_gap_sec=MIN_CALL_SEC):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    selected = []

    for d in detections:
        overlap = False

        for s in selected:
            if not (d["t_end"] <= s["t_start"] or d["t_start"] >= s["t_end"]):
                overlap = True
                break

        if not overlap:
            selected.append(d)

    selected = sorted(selected, key=lambda x: x["t_start"])

    merged = []
    cur = dict(selected[0])

    for d in selected[1:]:
        same_species = d["label"] == cur["label"]
        small_gap = (d["t_start"] - cur["t_end"]) <= min_gap_sec

        if same_species and small_gap:
            cur["t_end"] = max(cur["t_end"], d["t_end"])
            cur["confidences"].append(d["confidence"])
        else:
            merged.append(cur)
            cur = dict(d)

    merged.append(cur)

    # résumé final
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
        print(f"  Durée totale : {total_sec:.1f}s ({total_sec/60:.2f} min)")

    raw_detections = []

    for t_start, t_end, chunk in _windows(y_full, sr, window_sec, hop_sec):
        try:
            feats = extraire_features(y=chunk, sr=sr).reshape(1, -1)
        except Exception as e:
            if verbose:
                print(f"  [ERREUR feature] t={t_start:.1f}s : {e}")
            continue

        proba = pipeline.predict_proba(feats)[0]

        # 1. Compute top-2 margin FIRST
        sorted_proba = np.sort(proba)[::-1]
        margin = sorted_proba[0] - sorted_proba[1]

        # 2. Get predicted class
        label_idx = np.argmax(proba)
        confidence = float(proba[label_idx])
        label = le.inverse_transform([label_idx])[0]

        # 3. HARD FILTERS (order matters)
        if label == NOISE_LABEL:
            continue

        if confidence < min_conf:
            continue

        if margin < 0.15:
            continue

        # 4. Class-specific suppression
        if label == "Baleine à bosse" and confidence < 0.85:
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

    filtered = [d for d in raw_detections if d["label"] != "Inconnu"]

    merged = _merge_detections(filtered, min_gap_sec=min_call_sec)

    df = pd.DataFrame(merged, columns=[
        "species", "t_start", "t_end", "duration", "mean_confidence"
    ]) if merged else pd.DataFrame(columns=[
        "species", "t_start", "t_end", "duration", "mean_confidence"
    ])

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
    # Draw a horizontal timeline showing where each species call was detected.
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
        # label the bar
        if row["duration"] > total_sec * 0.015:
            ax.text(
                row["t_start"] + row["duration"] / 2,
                y_pos[row["species"]],
                f'{row["duration"]:.1f}s\n({row["mean_confidence"]:.0%})',
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
            )

    ax.set_xlim(0, total_sec)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))

    # minutes on x-axis
    max_min = int(total_sec // 60) + 1
    tick_secs = [m * 60 for m in range(max_min + 1) if m * 60 <= total_sec]
    ax.set_xticks(tick_secs)
    ax.set_xticklabels([f"{t//60:.0f}:{t%60:02.0f}" for t in tick_secs])

    ax.set_xlabel("Temps (mm:ss)")
    ax.set_title(f"Détection d'appels – {os.path.basename(audio_path)}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    patches = [mpatches.Patch(color=_SPECIES_COLOURS.get(sp, _DEFAULT_COLOUR), label=sp) for sp in species_list]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# print results

def afficher_resultats(df: pd.DataFrame):
    if df.empty:
        print("\n  Aucun appel détecté dans cet enregistrement.")
        return

    print(f"\n{'─'*65}")
    print(f"  {'Espèce':<30} {'Début':>7} {'Fin':>7} {'Durée':>7} {'Confiance':>9}")
    print(f"{'─'*65}")
    for _, r in df.iterrows():
        def fmt(s):
            m, sec = divmod(s, 60)
            return f"{int(m)}:{sec:05.2f}"
        print(f"  {r['species']:<30} {fmt(r['t_start']):>7} {fmt(r['t_end']):>7}"
              f" {r['duration']:>6.1f}s {r['mean_confidence']:>8.1%}")
    print(f"{'─'*65}")
    print(f"  Total : {len(df)} appel(s) détecté(s)\n")


def main():
    # 1. Build datasets
    print("\n[1] Chargement des données")
    df_train = construire_dataframe_avec_contexte(TRAIN_DIR, window_sec=1.5, n_augmented=5)
    df_test = construire_dataframe(TEST_DIR)

    le = LabelEncoder()
    le.fit(df_train["label"])

    X_train = df_train.drop(columns=["label"]).values
    y_train = le.transform(df_train["label"])
    X_test = df_test.drop(columns=["label"]).values
    y_test = le.transform(df_test["label"])

    print(f"\n  Classes : {list(le.classes_)}")
    print(f"  Train : {X_train.shape[0]} échantillons | Test : {X_test.shape[0]}")

    # 2. Train & cross-validate
    print("\n[2] Entraînement (cross-validation 5-fold)")
    models = get_models()
    results = evaluate_models(models, X_train, y_train)

    best_name, best_pipe = select_best(results)
    print(f"\n  Meilleur modèle : {best_name}")
    best_pipe.fit(X_train, y_train)

    # 3. Evaluate on held-out test set
    print("\n[3] Évaluation sur le jeu de test")
    evaluer(best_pipe, X_test, y_test, le, best_name)

    # 4. Save
    save(best_pipe, le)

    # 5. Detect calls in every .wav found in LONG_AUDIO_DIR
    if not os.path.isdir(LONG_AUDIO_DIR):
        print(f"\n[5] (Passer – dossier non trouvé : {LONG_AUDIO_DIR})")
        return

    # collect all wav files in the folder, sorted by name so sequence_06 comes before sequence_26
    wav_files = sorted([
        os.path.join(LONG_AUDIO_DIR, f)
        for f in os.listdir(LONG_AUDIO_DIR)
        if f.lower().endswith(".wav")
    ])

    if not wav_files:
        print(f"\n[5] (Passer – aucun fichier .wav dans {LONG_AUDIO_DIR})")
        return

    print(f"\n[5] Détection sur {len(wav_files)} séquence(s) audio longue(s)")

    # create the annotations output folder if it doesn't exist yet
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    all_detections = []   # will hold every detection across all sequences

    for audio_path in wav_files:
        df_results = detect_long_audio(audio_path)
        afficher_resultats(df_results)

        if not df_results.empty:
            # add a column so we know which file each detection came from
            df_results.insert(0, "fichier", os.path.basename(audio_path))
            all_detections.append(df_results)

            # save a per-sequence CSV in the annotations folder
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            csv_path = os.path.join(ANNOTATIONS_DIR, f"{stem}_detections.csv")
            df_results.to_csv(csv_path, index=False)

            # save a per-sequence timeline image in the same folder
            timeline_path = os.path.join(ANNOTATIONS_DIR, f"{stem}_timeline.png")
            plot_timeline(df_results, audio_path, save_path=timeline_path)

    # save one merged CSV that contains every detection from every sequence
    if all_detections:
        merged_csv = os.path.join(ANNOTATIONS_DIR, "all_detections.csv")
        pd.concat(all_detections, ignore_index=True).to_csv(merged_csv, index=False)
        print(f"\n  → {ANNOTATIONS_DIR}/ contient {len(all_detections)} CSV(s) + all_detections.csv")


if __name__ == "__main__":
    main()
