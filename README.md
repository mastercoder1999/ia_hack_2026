# ia_hack_2026
Maksim Déry, Julien Otis
Enzo Chassé dans nos coeurs
Our project repo for IA Hack 2026 hosted by UQAR

## Installation

### Venv
```
python3 -m venv /home/toko/Documents/GitHub/ia_hack_2026/venv
source venv/bin/activate
```
### Libs

(dans venv)
```
pip -r requirements.txt
```
### Dataset
```
git clone https://huggingface.co/datasets/confit/wmms-parquet
```

## Utilisation
Être dans le venv
### Différence importante
Nous utilisons un dossier part_1 et part_2 pour les données. le dossier part_2 contient les samples de noise. (Le code va marcher pareil si le noise est afficher, mais il va être entraîner avec la partie 1)
### Cas Partie 1
```
python3 part_1.py "/home/toko/Documents/GitHub/ia_hack_2026/data/part_1/train/" "/home/toko/Documents/GitHub/ia_hack_2026/data/part_1/test/"
```
### Cas Partie 2
Order : Train, Test, Long Sounds
```
python3 part_2_Maksim.py "/home/toko/Documents/GitHub/ia_hack_2026/data/part_2/train/" "/home/toko/Documents/GitHub/ia_hack_2026/data/part_2/test/" "/home/toko/Documents/GitHub/ia_hack_2026/data/long_audio/audio/"
```
### Cas Web
L'interface web utilise les deux parties pour faire un affichage complet des deux parties dans une belle page web
```
python3 web.py
```
