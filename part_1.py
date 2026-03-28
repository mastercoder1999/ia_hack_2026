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

TRAIN_DIR    = os.path.join("data", "train")
TEST_DIR     = os.path.join("data", "test")
SAMPLE_RATE  = 22050
N_MFCC       = 13
RANDOM_STATE = 42
OUTPUT_MODEL   = "meilleur_modele.pkl"
OUTPUT_ENCODER = "label_encoder.pkl"

# Mapping nom de dossier → label lisible
LABEL_MAP = {
    "Beluga_WhiteWhale"  : "Béluga",
    "Fin_FinbackWhale"   : "Rorqual commun",
    "HumpbackWhale"      : "Baleine à bosse",
    "SpermWhale"         : "Cachalot",
    "White_sidedDolphin" : "Dauphin à flancs blancs",
}

def extraire_features(chemin: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Retourne un vecteur de 58 features (mean + std) pour un fichier .wav.

    Détail :
      MFCC (13)           → 26 dims
      Centroïde spectral  →  2 dims
      Bandwidth spectrale →  2 dims
      ZCR                 →  2 dims
      Chroma STFT (12)    → 24 dims
      RMS énergie         →  2 dims
    """
    y, sr = librosa.load(chemin, sr=sr, mono=True)
    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))

    centroide = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats += [np.mean(centroide), np.std(centroide)]

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats += [np.mean(bandwidth), np.std(bandwidth)]

    zcr = librosa.feature.zero_crossing_rate(y)
    feats += [np.mean(zcr), np.std(zcr)]

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1))
    feats.extend(np.std(chroma, axis=1))

    rms = librosa.feature.rms(y=y)
    feats += [np.mean(rms), np.std(rms)]

    return np.array(feats)


def construire_dataframe(dossier_racine: str) -> pd.DataFrame:
    """
    Parcourt dossier_racine/<espece>/*.wav et retourne un DataFrame
    avec colonnes feat_0 … feat_N + 'label'.
    """
    enregistrements, labels = [], []

    especes = sorted([
        d for d in os.listdir(dossier_racine)
        if os.path.isdir(os.path.join(dossier_racine, d))
    ])

    if not especes:
        raise FileNotFoundError(
            f"Aucun sous-dossier dans '{dossier_racine}'. "
            "Vérifie le chemin."
        )

    for espece in especes:
        dossier = os.path.join(dossier_racine, espece)
        fichiers = sorted([
            f for f in os.listdir(dossier)
            if f.lower().endswith(".wav")
        ])
        label = LABEL_MAP.get(espece, espece)
        print(f"  → {label:30s} : {len(fichiers)} fichiers")

        for fichier in fichiers:
            chemin = os.path.join(dossier, fichier)
            try:
                enregistrements.append(extraire_features(chemin))
                labels.append(label)
            except Exception as e:
                print(f"     [ERREUR] {fichier} : {e}")

    df = pd.DataFrame(enregistrements)
    df.columns = [f"feat_{i}" for i in range(df.shape[1])]
    df["label"] = labels
    return df


# ─────────────────────────────────────────────
# 2. ENTRAÎNEMENT ET SÉLECTION DU MODÈLE
# ─────────────────────────────────────────────

def entrainer_modeles(X_train: np.ndarray, y_train: np.ndarray):
    """
    Compare 4 classifieurs par cross-validation 5-fold (F1-macro).
    Entraîne le meilleur sur tout X_train et le retourne.
    """
    modeles = {
        "Random Forest"    : RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (RBF)"        : SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=RANDOM_STATE),
        "KNN (k=5)"        : KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
    }

    resultats = {}
    print("\n  Cross-validation 5-fold (F1-macro) :")

    for nom, clf in modeles.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=5, scoring="f1_macro", n_jobs=-1,
        )
        resultats[nom] = {
            "pipeline": pipeline,
            "f1": scores.mean(),
            "std": scores.std(),
        }
        print(f"    {nom:25s}  F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    meilleur_nom = max(resultats, key=lambda k: resultats[k]["f1"])
    print(f"\n  → Meilleur modèle : {meilleur_nom}")

    meilleur = resultats[meilleur_nom]["pipeline"]
    meilleur.fit(X_train, y_train)
    return meilleur, meilleur_nom


# ─────────────────────────────────────────────
# 3. ÉVALUATION
# ─────────────────────────────────────────────

def evaluer(pipeline, X_test: np.ndarray, y_test: np.ndarray,
            le: LabelEncoder, nom_modele: str):
    """Affiche les métriques et sauvegarde les graphiques."""
    y_pred  = pipeline.predict(X_test)
    classes = le.classes_

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")

    print(f"\n=== Résultats test set — {nom_modele} ===")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-macro  : {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=classes))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title(f"Matrice de confusion — {nom_modele}")
    plt.ylabel("Vrai label")
    plt.xlabel("Label prédit")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("  → confusion_matrix.png sauvegardé")

    # Importance des features (Random Forest uniquement)
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        idx = np.argsort(imp)[::-1][:20]
        plt.figure(figsize=(10, 4))
        plt.bar(range(20), imp[idx])
        plt.xticks(range(20), [f"feat_{i}" for i in idx],
                   rotation=45, ha="right")
        plt.title("Top 20 features les plus importantes")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()
        print("  → feature_importance.png sauvegardé")


# ─────────────────────────────────────────────
# 4. SAUVEGARDE / CHARGEMENT
# ─────────────────────────────────────────────

def sauvegarder(pipeline, le: LabelEncoder):
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(pipeline, f)
    with open(OUTPUT_ENCODER, "wb") as f:
        pickle.dump(le, f)
    print(f"\n  → {OUTPUT_MODEL} sauvegardé  (pipeline complet scaler + clf)")
    print(f"  → {OUTPUT_ENCODER} sauvegardé")


def charger():
    with open(OUTPUT_MODEL, "rb") as f:
        pipeline = pickle.load(f)
    with open(OUTPUT_ENCODER, "rb") as f:
        le = pickle.load(f)
    return pipeline, le


# ─────────────────────────────────────────────
# 5. PRÉDICTION SUR UN FICHIER ISOLÉ
# ─────────────────────────────────────────────

def predire(chemin_wav: str) -> str:
    """Prédit l'espèce d'un fichier .wav avec le modèle sauvegardé."""
    pipeline, le = charger()
    feats = extraire_features(chemin_wav).reshape(1, -1)
    pred  = pipeline.predict(feats)[0]
    proba = pipeline.predict_proba(feats)[0]
    label = le.inverse_transform([pred])[0]

    print(f"\nFichier : {chemin_wav}")
    print(f"Espèce prédite : {label}")
    for cls, p in zip(le.classes_, proba):
        barre = "█" * int(p * 30)
        print(f"  {cls:30s} {barre} {p:.3f}")
    return label


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  IA'Hack 2026 — Partie 1 : Classification d'espèces")
    print("=" * 55)

    # 1. Extraction train
    print(f"\n[1/4] Extraction des features — train ({TRAIN_DIR}) ...")
    df_train = construire_dataframe(TRAIN_DIR)
    print(f"  → {df_train.shape[0]} échantillons · {df_train.shape[1]-1} features")

    # 2. Extraction test
    print(f"\n[2/4] Extraction des features — test ({TEST_DIR}) ...")
    df_test = construire_dataframe(TEST_DIR)
    print(f"  → {df_test.shape[0]} échantillons · {df_test.shape[1]-1} features")

    # Encodage des labels (fit uniquement sur train)
    le = LabelEncoder()
    le.fit(df_train["label"])

    X_train = df_train.drop(columns=["label"]).values
    y_train = le.transform(df_train["label"].values)
    X_test  = df_test.drop(columns=["label"]).values
    y_test  = le.transform(df_test["label"].values)

    print(f"\n  Classes : {list(le.classes_)}")

    # Sauvegarde CSV optionnelle
    df_train.to_csv("features_train.csv", index=False)
    df_test.to_csv("features_test.csv",   index=False)
    print("  → features_train.csv / features_test.csv sauvegardés")

    # 3. Entraînement + sélection
    print("\n[3/4] Entraînement et sélection du modèle ...")
    meilleur_pipeline, meilleur_nom = entrainer_modeles(X_train, y_train)

    # 4. Évaluation
    print("\n[4/4] Évaluation sur le test set ...")
    evaluer(meilleur_pipeline, X_test, y_test, le, meilleur_nom)

    # Sauvegarde
    sauvegarder(meilleur_pipeline, le)

    print("\nPartie 1 terminée. Modèle prêt pour la Partie 2 (détection).")