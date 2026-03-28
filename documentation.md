# Documentation
Maksim Déry et Julien otis

## Fonctions
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

def construire_dataframe(dossier_racine: str) -> pd.DataFrame:
    """
    Parcourt dossier_racine/<espece>/*.wav et retourne un DataFrame
    avec colonnes feat_0 … feat_N + 'label'.
    """

def entrainer_modeles(X_train: np.ndarray, y_train: np.ndarray):
    """
    Compare 4 classifieurs par cross-validation 5-fold (F1-macro).
    Entraîne le meilleur sur tout X_train et le retourne.
    """
def evaluer(pipeline, X_test: np.ndarray, y_test: np.ndarray,
            le: LabelEncoder, nom_modele: str):
    """Affiche les métriques et sauvegarde les graphiques."""