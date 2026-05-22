# 🖐️ Détection et Reconnaissance de Chiffres avec les Mains

Application de **vision par ordinateur en temps réel** qui détecte jusqu'à **2 mains** devant la webcam, affiche le chiffre correspondant et permet d'effectuer des **opérations mathématiques** avec les doigts.

> **Paume ET dos de la main** · **Score lissé (médiane 7 frames)** · **Compteur FPS** · **4 modes opératoires** · **Mode binaire (0–31)**

---

## 📸 Aperçu

| Geste | Mode | Résultat affiché |
|-------|------|-----------------|
| Poing fermé | Normal | **0 — ZERO** |
| 1 doigt levé | Normal | **1 — UN** |
| Main G × 3 + Main D × 4 | Addition | **3 + 4 = 7 — SEPT** |
| Main G × 5 − Main D × 2 | Soustraction | **5 - 2 = 3 — TROIS** |
| Main G × 3 × Main D × 4 | Multiplication | **3 x 4 = 12** |
| Main G × 6 ÷ Main D × 2 | Division | **6 / 2 = 3 — TROIS** |
| Pouce + Index + Auriculaire | **Binaire** | **10011 = 19** |
| 2 mains en binaire | Binaire + Addition | **10011 + 00110 = 25** |

---

## 🛠️ Technologies utilisées

| Librairie | Version | Rôle |
|-----------|---------|------|
| [OpenCV](https://opencv.org/) | 4.13.0 | Capture vidéo et affichage graphique |
| [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | 0.10.35 | Détection des 21 points clés de la main |
| [NumPy](https://numpy.org/) | 2.4.6 | Calculs matriciels sur les frames |
| Python | 3.10+ | Langage principal |

---

## 📁 Structure du projet

```
hand-digit-detection/
│
├── 📄 main.py               ← Point d'entrée — fichier à lancer
├── 📄 config.py             ← Toutes les constantes (couleurs, chemins, modes)
├── 📄 requirements.txt      ← Dépendances Python à installer
├── 📄 README.md             ← Documentation du projet
├── 📄 .gitignore            ← Fichiers exclus de git (.venv, *.task, ...)
│
├── 📂 core/                 ← Logique métier (traitement des données)
│   ├── __init__.py          ← Marqueur de package Python
│   ├── camera.py            ← Recherche et validation de la webcam
│   ├── hand_analyzer.py     ← Comptage normal + binaire des doigts (21 landmarks)
│   └── model.py             ← Téléchargement automatique du modèle IA
│
└── 📂 ui/                   ← Interface graphique (rendu OpenCV)
    ├── __init__.py          ← Marqueur de package Python
    └── overlay.py           ← Squelette, score, badges de mode, alertes
```

### Rôle de chaque fichier

| Fichier | Responsabilité |
|---------|---------------|
| `main.py` | Orchestre les étapes : modèle → caméra → détection → modes → affichage |
| `config.py` | **Source unique** de vérité : couleurs, seuils, liste des modes opératoires |
| `core/camera.py` | Trouve la première webcam réelle (critères mean + std) |
| `core/hand_analyzer.py` | Comptage normal (0–5) et binaire (0–31) des doigts |
| `core/model.py` | Télécharge le modèle `hand_landmarker.task` une seule fois |
| `ui/overlay.py` | Squelette, résultat, badges [M]/[B], FPS, alertes caméra |

---

## ⚙️ Installation

### 1. Cloner / télécharger le projet

```bash
git clone https://github.com/Tmousnier/hand-digit-detection.git
cd hand-digit-detection
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

### 3. Activer l'environnement virtuel

**Windows :**
```bash
.venv\Scripts\activate
```

**macOS / Linux :**
```bash
source .venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 5. Lancer l'application

```bash
python main.py
```

> **Note :** Le modèle MediaPipe (~40 Mo) est téléchargé automatiquement lors du premier lancement dans `C:\Users\<nom>\hand_landmarker.task`.

---

## 🎮 Utilisation

### Raccourcis clavier

| Touche | Action |
|--------|--------|
| **Q** ou **Échap** | Quitter l'application |
| **M** | Changer de mode opératoire (Addition → Soustraction → Multiplication → Division) |
| **B** | Activer / désactiver le mode binaire (0–31 par main) |
| Croix ✕ | Fermer la fenêtre |

### Interface

```
┌─────────────────────────────────────────────────┐
│ Vision IA : Chiffres    28.4 FPS    [Q] Quitter │  ← haut
│                                   [M] Addition  │
│                                  [B] Binaire:ON │
│                                                 │
│               [squelette main]                  │  ← centre
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │        3  +  4  =  7  —  SEPT           │   │  ← bas
│  │  Main G.: 3        Main D.: 4           │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## 🧠 Fonctionnement technique

### Modes opératoires (touche M)

Quand **2 mains** sont détectées, l'opération choisie est appliquée entre la main gauche et la main droite :

| Mode | Symbole | Exemple (G=3, D=4) | Résultat |
|------|---------|-------------------|---------|
| Addition | `+` | `3 + 4` | `7 — SEPT` |
| Soustraction | `-` | `3 - 4` | `-1` |
| Multiplication | `x` | `3 x 4` | `12` |
| Division | `/` | `3 / 4` | `0.75` |

> Division par zéro → affiche `Err /0`
> Le nom français n'est affiché que si le résultat est un entier entre 0 et 10.

### Mode binaire (touche B)

Chaque doigt représente un **bit** (puissance de 2) :

| Doigt | Bit | Valeur |
|-------|-----|--------|
| Pouce | bit 0 | 1 |
| Index | bit 1 | 2 |
| Majeur | bit 2 | 4 |
| Annulaire | bit 3 | 8 |
| Auriculaire | bit 4 | 16 |

**Maximum par main : 31** (tous les doigts levés = `11111` en binaire)

Exemple : Pouce + Index + Auriculaire levés → `10011 = 19`

Le mode binaire est **compatible avec les 4 opérateurs** : `M` + `B` pour combiner deux valeurs de 0 à 31.

### Détection de l'orientation de la main

Le module `core/hand_analyzer.py` détecte automatiquement si la **paume** ou le **dos** de la main fait face à la caméra via le **produit vectoriel** entre les landmarks 0, 5 et 17 :

| Orientation | Pouce | Index → Auriculaire |
|---|---|---|
| Dos face caméra | `TIP.x < BASE.x` (main droite) | `TIP.y < BASE.y` |
| **Paume face caméra** | `TIP.x > BASE.x` (main droite) ← **corrigé** | `TIP.y < BASE.y` (inchangé) |

### Lissage du score (médiane glissante)

Pour éviter les **clignotements**, le score est lissé via une **médiane glissante sur 7 frames** :

```python
from collections import deque
import statistics

score_buffer = deque(maxlen=7)
score_buffer.append(raw_score)
smooth_score = int(statistics.median(score_buffer))
```

### Compteur FPS

| Couleur | Seuil | Interprétation |
|---------|-------|---------------|
| 🟢 Vert | ≥ 24 FPS | Fluide |
| 🟠 Orange | ≥ 15 FPS | Acceptable |
| 🔴 Rouge | < 15 FPS | Lent |

### Détection des doigts (21 landmarks)

```
        8   12  16  20       ← Bouts de doigts (TIPs)
        |   |   |   |
        7   11  15  19
        |   |   |   |
        6   10  14  18
        |   |   |   |
   4    5   9   13  17
   |    |
   3    0 ← Poignet
   |
   2
   |
   1
```

### Détection de la webcam

Le module `core/camera.py` valide chaque flux avec **deux critères** :
- `mean > 10` → image non noire
- `std > 15` → image non uniforme (rejette les fonds gris virtuels)

---

## 🔧 Configuration

Tous les paramètres sont dans **`config.py`** :

```python
# Résolution de la fenêtre
WIDTH, HEIGHT = 1280, 720

# Seuils de confiance MediaPipe (entre 0.0 et 1.0)
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE       = 0.5

# Lissage du score
SMOOTH_WINDOW = 7   # Nombre de frames pour la médiane glissante (dans main.py)

# Modes opératoires disponibles (touche M pour cycler)
MODES = [
    {"symbol": "+", "label": "Addition"},
    {"symbol": "-", "label": "Soustraction"},
    {"symbol": "x", "label": "Multiplication"},
    {"symbol": "/", "label": "Division"},
]

# Couleurs (format BGR d'OpenCV)
C_YELLOW = (0, 220, 255)   # Score principal
C_GREEN  = (0, 200, 80)    # Détail par main / FPS fluide
C_ORANGE = (0, 165, 255)   # Badge mode opératoire
```

---

## 🐛 Problèmes courants

| Problème | Solution |
|----------|----------|
| **Aucune caméra détectée** | Vérifier les paramètres de confidentialité Windows → Caméra → autoriser Python |
| **Image grise / fond neutre** | Fermer Teams, Zoom, OBS ou tout autre logiciel utilisant la caméra |
| **L'application se ferme immédiatement** | Les touches ne sont acceptées qu'après **4 secondes** (protection buffer clavier) |
| **Modèle non trouvé** | Le fichier `hand_landmarker.task` est téléchargé automatiquement au premier lancement |
| **Mode binaire instable** | Tenir la main bien perpendiculaire à la caméra pour une meilleure détection des bits |

---

## 📄 Licence

Projet éducatif — libre d'utilisation et de modification.
