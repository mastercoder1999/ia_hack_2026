import os
import warnings
import pickle

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

TRAIN_DIR = os.path.join("data", "part_1", "train")
TEST_DIR = os.path.join("data", "part_1", "test")

SAMPLE_RATE = 22050
N_MFCC = 13
RANDOM_STATE = 42

OUTPUT_MODEL = "meilleur_modele_part1.pkl"
OUTPUT_ENCODER = "label_encoder_part1.pkl"

LABEL_MAP = {
    "Beluga_WhiteWhale": "Béluga",
    "Fin_FinbackWhale": "Rorqual commun",
    "HumpbackWhale": "Baleine à bosse",
    "SpermWhale": "Cachalot",
    "White_sidedDolphin": "Dauphin à flancs blancs",
}

def extraire_features(path: str, sr=SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr, mono=True)
    # np.mean =  the average value of each feature over the entire clip, np.std =  the standard deviation, capturing how much each feature varies over time
        
    def stats(x):
        mean = np.mean(x, axis=1) if x.ndim > 1 else np.array([np.mean(x)])
        std  = np.std(x, axis=1)  if x.ndim > 1 else np.array([np.std(x)])
        return np.concatenate([mean, std])
    feats = []

    # computing the MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # computing the features of the MFCCs (mean and std) and adding them to the list of features
    feats.extend(stats(mfcc))

    # Features avec sr
    for func in [
        # "brightness" of the sound,whale low/dolphin high
        librosa.feature.spectral_centroid,
        # how spread out the frequencies are around that centroid, Narrow = tonal/pure tone, wide = noisy/complex
        librosa.feature.spectral_bandwidth,
    ]:
        f = func(y=y, sr=sr)
        feats.extend(stats(f))

    # how often the signal flips from 0 to 1 over time
    zcr = librosa.feature.zero_crossing_rate(y)
    feats.extend(stats(zcr))

    # how much loudness varies over time
    rms = librosa.feature.rms(y=y)
    feats.extend(stats(rms))

    # 12 pitch classes tracked over time
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(stats(chroma))

    feats = np.array(feats)

    assert feats.shape[0] == 58, f"Feature size mismatch: {feats.shape}"

    # its done and all in one place
    return feats

def lister_audio_files(root):
    #Looping over species folders
    for specie in sorted(os.listdir(root)):
        #building the full path to the respective species folder
        d = os.path.join(root, specie)
        #skips over non important folders
        if not os.path.isdir(d):
            continue

        #collecting .wav files
        files = [
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(".wav")
        ]

        yield specie, files

def construire_dataframe(root):
    X, y = [], []
    # Loop through all les fichiers audios et mettre la réponse dans un label lié à l'audio.
    for specie, files in lister_audio_files(root):
        label = LABEL_MAP.get(specie, specie)
        print(f"{label:30s} : {len(files)} fichiers")

        for f in files:
            try:
                X.append(extraire_features(f))
                y.append(label)
            except Exception as e:
                print(f"[ERREUR] {f} : {e}")

    # Mettre features dans un dataframe et le retourner
    df = pd.DataFrame(X)
    df["label"] = y
    return df

def get_models():
    # Will return all MLs used for this project.
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=RANDOM_STATE
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

def evaluate_models(models, X, y):
    results = {}
    # For EVERY model : entourer dans une pipeline, evaluation avec f1, stocker mean et std des résultats
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro")

        results[name] = {
            "pipeline": pipe,
            "f1": scores.mean(),
            "std": scores.std(),
        }

        print(f"{name:25s} F1={scores.mean():.4f} ± {scores.std():.4f}")

    return results

def select_best(results):
    # Compare les modèles selon le f1 et retourne le meilleur
    name = max(results, key=lambda k: results[k]["f1"])
    return name, results[name]["pipeline"]

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }

def plot_confusion(cm, classes, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes)
    plt.title(model_name)
    plt.tight_layout()
    plt.savefig("confusion_matrix_part_1.png")
    plt.close()

def plot_importance(clf):
    if not hasattr(clf, "feature_importances_"):
        return

    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:20]

    plt.figure(figsize=(10, 4))
    plt.bar(range(20), imp[idx])
    plt.xticks(range(20), idx, rotation=45)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

def evaluer(pipeline, X, y, label, name):
    # Générer des prédictions à partir de la pipeline entraîné
    y_pred = pipeline.predict(X)

    metrics = compute_metrics(y, y_pred)

    print(f"\n{name}")
    print(metrics)
    print(classification_report(y, y_pred, target_names=label.classes_))
    # rows = true labels, columns = predicted labels
    cm = confusion_matrix(y, y_pred)
    plot_confusion(cm, label.classes_, name)
    plot_importance(pipeline.named_steps["clf"])

def save(pipeline, label):
    # Sauvegarde du model.
    pickle.dump(pipeline, open(OUTPUT_MODEL, "wb"))
    pickle.dump(label, open(OUTPUT_ENCODER, "wb"))

def load():
    # Load le model.
    return (
        pickle.load(open(OUTPUT_MODEL, "rb")),
        pickle.load(open(OUTPUT_ENCODER, "rb")),
    )

def predict_file(path):
    # Seulement utiliser pour web.
    # Load le modèle préfait pour faire de l'analyse de son manuelle.
    pipeline, label = load()
    x = extraire_features(path).reshape(1, -1)

    pred = pipeline.predict(x)[0]
    proba = pipeline.predict_proba(x)[0]

    return pred, proba, label

def print_prediction(pred, proba, label):
    label = label.inverse_transform([pred])[0]
    print(f"Prediction: {label}")

    for cls, p in zip(label.classes_, proba):
        print(f"{cls:30s} {'█'*int(p*30)} {p:.3f}")

def main():
    print("[1] Loading data")
    # contruire dataframe de train et test avec les images présentes
    df_train = construire_dataframe(TRAIN_DIR)
    df_test = construire_dataframe(TEST_DIR)

    # Merge le data de train dans un labelEncoder
    label = LabelEncoder()
    label.fit(df_train["label"])

    # Convertir Dataset dans un format que scikit peut lire
    X_train = df_train.drop(columns=["label"]).values
    y_train = label.transform(df_train["label"])

    X_test = df_test.drop(columns=["label"]).values
    y_test = label.transform(df_test["label"])

    print("[2] Training")
    # Get models and evaluates them
    models = get_models()
    results = evaluate_models(models, X_train, y_train)
    # Choisir le meilleur modèle selon les résultats
    best_name, best_pipe = select_best(results)
    best_pipe.fit(X_train, y_train)

    print("[3] Evaluation")
    # Evaluation avec le meilleur pipeline
    evaluer(best_pipe, X_test, y_test, label, best_name)

    print("[4] Saving")
    # Sauvegarde du Pipeline pour utilisation web
    save(best_pipe, label)

if __name__ == "__main__":
    main()
