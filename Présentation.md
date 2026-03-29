librairie utiliser:
    numpy
    pandas
    librosa
    matplotlib
    seaborn
    sklearn(model essayer: Random forest, SVM, KNN, Gradient boosting)
        julien: focus sur random forest(résultat légerement inférieur au SVM de Maksim mais tôt de confiance plus bas, maksim: focus sur SVM(meilleur resultat et meilleur tot de confiance)

maniere d'extract les features:
    MFCC
    Spectral centroid(Brightness)
    spectal bandwidth(how spread out the frequencys)
    Zero crossing rate(how often the signal flips from 0 to 1 over time)
    RMS(loudness variation)
    Chroma(répartition des pitches)



Probleme rencontrer

================================
M 

Timestamps (overlap, longeur)
Overlap: ajouter une condition qui s'assure que si des valeurs overlap, on en supprime une.

Longeur: Jouer avec nos paramètres pour trouver la config parfaite. 
    WINDOW_SEC = 1.5
    HOP_SEC = 0.25
    MIN_CONF = 0.5
    MIN_CALL_SEC = 1.0

=============================
J

BALEINE À BOSSE ET BÉLUGA SPAM

pourquoi?