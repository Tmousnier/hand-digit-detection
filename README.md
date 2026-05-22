# 🖐️ Détection et Reconnaissance de Chiffres avec les Mains

Application de **vision par ordinateur en temps réel** qui détecte jusqu'à **2 mains** devant la webcam et affiche le **chiffre correspondant au nombre de doigts levés** (de 0 à 10).

---

## 📸 Aperçu

| Geste | Résultat affiché |
|-------|-----------------|
| Poing fermé | **0 — ZERO** |
| 1 doigt levé | **1 — UN** |
| 2 mains × 5 doigts | **10 — DIX** |

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
├── 📄 config.py             ← Toutes les constantes (couleurs, chemins, seuils)
├── 📄 requirements.txt      ← Dépendances Python à installer
├── 📄 README.md             ← Documentation du projet
├── 📄 .gitignore            ← Fichiers exclus de git (.venv, *.task, ...)
│
├── 📂 core/                 ← Logique métier (traitement des données)
│   ├── __init__.py          ← Marqueur de package Python
│   ├── camera.py            ← Recherche et validation de la webcam
│   ├── hand_analyzer.py     ← Comptage des doigts levés (21 landmarks)
│   └── model.py             ← Téléchargement automatique du modèle IA
│
└── 📂 ui/                   ← Interface graphique (rendu OpenCV)
    ├── __init__.py          ← Marqueur de package Python
    └── overlay.py           ← Squelette de la main, score, alertes
```

### Rôle de chaque fichier

| Fichier | Responsabilité |
|---------|---------------|
| `main.py` | Orchestre les 5 étapes : modèle → caméra → fenêtre → détection → nettoyage |
| `config.py` | **Source unique** de vérité pour toutes les constantes du projet |
| `core/camera.py` | Trouve la première webcam réelle parmi les indices/backends disponibles |
| `core/hand_analyzer.py` | Détermine si chaque doigt est levé à partir des 21 landmarks MediaPipe |
| `core/model.py` | Télécharge le modèle `hand_landmarker.task` une seule fois |
| `ui/overlay.py` | Dessine le squelette, le panneau de score et les messages d'erreur |

---

## ⚙️ Installation

### 1. Cloner / télécharger le projet

```bash
git clone <url-du-repo>
cd detection_main_chiffre
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

| Action | Commande |
|--------|----------|
| Quitter l'application | Touche **Q** ou **Échap** |
| Fermer la fenêtre | Clic sur la **croix** ✕ |

- Placez **1 ou 2 mains** face à la webcam.
- Levez les doigts souhaités.
- Le chiffre s'affiche en temps réel en bas de l'écran.

---

## 🧠 Fonctionnement technique

### Détection des doigts

MediaPipe renvoie **21 points (landmarks)** par main, numérotés de 0 à 20 :

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

**Règle de détection :**

| Doigt | Critère "levé" |
|-------|----------------|
| Index, Majeur, Annulaire, Auriculaire | `TIP.y < BASE.y` (TIP plus haut que la base — axe Y inversé en image) |
| Pouce | `TIP.x < BASE.x` (main droite) ou `TIP.x > BASE.x` (main gauche) — comparaison horizontale |

### Détection de la webcam

Sur Windows, certains indices de caméra retournent un fond gris uniforme (dispositif virtuel).
Le module `core/camera.py` valide chaque flux avec **deux critères** :
- `mean > 10` → image non noire
- `std > 15` → image non uniforme (rejette les placeholders)

---

## 🔧 Configuration

Tous les paramètres sont regroupés dans **`config.py`** :

```python
# Résolution de la fenêtre
WIDTH, HEIGHT = 1280, 720

# Seuils de confiance MediaPipe (entre 0.0 et 1.0)
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE       = 0.5

# Couleurs (format BGR d'OpenCV)
C_YELLOW = (0, 220, 255)  # Score principal
C_GREEN  = (0, 200, 80)   # Détail par main
```

---

## 🐛 Problèmes courants

| Problème | Solution |
|----------|----------|
| **Aucune caméra détectée** | Vérifier les paramètres de confidentialité Windows → Caméra → autoriser Python |
| **Image grise / fond neutre** | Fermer Teams, Zoom, OBS ou tout autre logiciel utilisant la caméra |
| **L'application se ferme immédiatement** | Les touches ne sont acceptées qu'après **4 secondes** au démarrage (protection buffer clavier) |
| **Modèle non trouvé** | Le fichier `hand_landmarker.task` est téléchargé automatiquement au premier lancement |

---

## 📄 Licence

Projet éducatif — libre d'utilisation et de modification.

