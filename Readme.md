# BrainScan AI

Classification d'images IRM cérébrales avec un réseau de neurones convolutif (CNN).
Le modèle reconnaît quatre classes : `glioma`, `meningioma`, `notumor` et `pituitary`.

> Ce projet est pédagogique. Il aide à explorer une image, mais ne remplace jamais
> l'avis d'un médecin et ne constitue pas un outil de diagnostic certifié.

## Structure

```text
BrainScan/
├── Data/                         # Images classées par dossier
├── notebooks/01_traitement.ipynb # Explication étape par étape
├── src/step1/
│   ├── inspect_folders.py        # Vérification des fichiers
│   ├── prepare_data.py           # Chargement, nettoyage et séparation
│   ├── preprocess_images.py      # Prétraitement d'une image
│   └── visualize_dataset.py      # Graphiques du dataset
├── src/step2/
│   ├── build_model.py            # Architecture CNN
│   └── train_model.py            # Entraînement et évaluation
├── src/predict.py                # Prédiction sur une image
├── app.py                        # Interface Streamlit
├── requirements.txt
└── .gitignore
```

## Installation

Depuis le dossier `BrainScan` :

```bash
python -m venv .venv
```

Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Vérifier le dataset

```bash
python src/step1/inspect_folders.py
python src/step1/visualize_dataset.py
```

Les extensions acceptées sont `jpg`, `jpeg`, `png` et `bmp`. Les images illisibles
sont ignorées pendant le chargement.

## Entraîner le modèle

```bash
python -m src.step2.train_model
```

Le programme crée le dossier `models/` avec :

- `brain_scan.keras` : meilleur modèle sauvegardé ;
- `training_curves.png` : évolution de l'accuracy et de la loss ;
- `confusion_matrix.png` : erreurs par classe ;
- `history.json` : valeurs de l'entraînement.

Le dataset est séparé en 70 % entraînement, 15 % validation et 15 % test.
Les pixels sont redimensionnés en `224x224` puis normalisés entre 0 et 1.
Les images sont chargées par lots avec `tf.data` pour éviter de remplir toute la
mémoire RAM avant le début de l'entraînement. La taille utilisée est de 16 images
par lot pour rester compatible avec une carte graphique de 4 Go.

## Utiliser le GPU sous Windows

TensorFlow récent n'utilise pas CUDA directement sous Windows natif. Pour utiliser
une carte NVIDIA, il faut passer par WSL2 avec Ubuntu et un pilote NVIDIA compatible.
Le GPU doit être visible avec :

```bash
wsl -d Ubuntu -- nvidia-smi
```

Dans Ubuntu WSL2, créer un environnement séparé et installer TensorFlow avec CUDA :

```bash
python3.13 -m venv ~/brainscan-venv
~/brainscan-venv/bin/pip install "tensorflow[and-cuda]"
~/brainscan-venv/bin/pip install opencv-python scikit-learn matplotlib pillow
```

L'entraînement doit ensuite être lancé depuis la racine du projet avec le Python
de cet environnement. Si WSL2 devient instable pendant l'entraînement, utiliser
le mode CPU avec l'environnement Windows ou mettre à jour le pilote NVIDIA et WSL2.

## Tester une image

```bash
python src/predict.py
```

Ou directement avec l'interface :

```bash
streamlit run app.py
```

La commande `python -v streamlit run app.py` est incorrecte : `-v` active le mode
verbeux de Python et ne lance pas Streamlit.

## Planning Jira proposé

| Epic | Tickets |
|---|---|
| Préparer les données | Inspecter les dossiers, vérifier les extensions, prétraiter et visualiser |
| Construire le CNN | Définir l'architecture, compiler le modèle, vérifier `summary()` |
| Entraîner et évaluer | Séparer les données, entraîner, sauvegarder, mesurer les performances |
| Déployer une démonstration | Ajouter `predict_image`, créer l'interface Streamlit |
| Documenter le projet | Compléter le notebook, le README et préparer la démonstration |

## Limites

La performance dépend de la qualité et de la représentativité du dataset. Une
bonne accuracy ne suffit pas pour valider un usage clinique : il faudrait des
tests externes, une validation médicale, une analyse des biais et une certification.
Le modèle ne fournit pas un diagnostic médical et les prédictions doivent être
interprétées uniquement comme une démonstration technique.