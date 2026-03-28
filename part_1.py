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

TRAIN_DIR = os.path.join("data", "train")
TEST_DIR = os.path.join("data", "test")

SAMPLE_RATE = 22050
N_MFCC = 13
RANDOM_STATE = 42

OUTPUT_MODEL = "meilleur_modele.pkl"
OUTPUT_ENCODER = "label_encoder.pkl"

LABEL_MAP = {
    "Beluga_WhiteWhale": "Béluga",
    "Fin_FinbackWhale": "Rorqual commun",
    "HumpbackWhale": "Baleine à bosse",
    "SpermWhale": "Cachalot",
    "White_sidedDolphin": "Dauphin à flancs blancs",
}

def extraire_features(path: str, sr=SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr, mono=True)

    def stats(x):
        return np.mean(x, axis=-1), np.std(x, axis=-1)

    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats += list(stats(mfcc)[0]) + list(stats(mfcc)[1])

    # Features avec sr
    for func in [
        librosa.feature.spectral_centroid,
        librosa.feature.spectral_bandwidth,
    ]:
        f = func(y=y, sr=sr)
        feats += list(stats(f))

    # Features sans sr
    zcr = librosa.feature.zero_crossing_rate(y)
    feats += list(stats(zcr))

    rms = librosa.feature.rms(y=y)
    feats += list(stats(rms))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats += list(stats(chroma)[0]) + list(stats(chroma)[1])

    return np.array(feats)

def iter_audio_files(root):
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

def construire_dataframe(root):
    X, y = [], []

    for specie, files in iter_audio_files(root):
        label = LABEL_MAP.get(specie, specie)
        print(f"{label:30s} : {len(files)} fichiers")

        for f in files:
            try:
                X.append(extraire_features(f))
                y.append(label)
            except Exception as e:
                print(f"[ERREUR] {f} : {e}")

    df = pd.DataFrame(X)
    df["label"] = y
    return df

def get_models():
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
    plt.savefig("confusion_matrix.png")
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

def evaluer(pipeline, X, y, le, name):
    y_pred = pipeline.predict(X)

    metrics = compute_metrics(y, y_pred)

    print(f"\n{name}")
    print(metrics)
    print(classification_report(y, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y, y_pred)
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

def predict_file(path):
    pipeline, le = load()
    x = extraire_features(path).reshape(1, -1)

    pred = pipeline.predict(x)[0]
    proba = pipeline.predict_proba(x)[0]

    return pred, proba, le

def print_prediction(pred, proba, le):
    label = le.inverse_transform([pred])[0]
    print(f"Prediction: {label}")

    for cls, p in zip(le.classes_, proba):
        print(f"{cls:30s} {'█'*int(p*30)} {p:.3f}")

def main():
    print("[1] Loading data")
    df_train = construire_dataframe(TRAIN_DIR)
    df_test = construire_dataframe(TEST_DIR)

    le = LabelEncoder()
    le.fit(df_train["label"])

    X_train = df_train.drop(columns=["label"]).values
    y_train = le.transform(df_train["label"])

    X_test = df_test.drop(columns=["label"]).values
    y_test = le.transform(df_test["label"])

    print("[2] Training")
    models = get_models()
    results = evaluate_models(models, X_train, y_train)

    best_name, best_pipe = select_best(results)
    best_pipe.fit(X_train, y_train)

    print("[3] Evaluation")
    evaluer(best_pipe, X_test, y_test, le, best_name)

    print("[4] Saving")
    save(best_pipe, le)

if __name__ == "__main__":
    main()