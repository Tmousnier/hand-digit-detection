"""
config.py
─────────
Centralise toutes les constantes du projet :
  - Chemins des fichiers
  - Dimensions de la fenêtre
  - Palette de couleurs (format BGR d'OpenCV)
  - Paramètres du détecteur MediaPipe
  - Dictionnaire chiffres → noms français

Modifier ce fichier suffit pour changer l'apparence ou le comportement
de l'application sans toucher au code métier.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
#  MODÈLE MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────

# Chemin de sauvegarde du modèle — placé dans le dossier personnel pour éviter
# les problèmes de MediaPipe avec les chemins contenant des accents / espaces.
MODEL_PATH = os.path.join(os.path.expanduser("~"), "hand_landmarker.task")

# URL de téléchargement automatique (exécuté une seule fois si absent)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ─────────────────────────────────────────────────────────────────────────────
#  FENÊTRE D'AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_NAME   = "Reconnaissance de Chiffres"
WIDTH         = 1280   # Largeur de la fenêtre en pixels
HEIGHT        = 720    # Hauteur de la fenêtre en pixels


# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE DE COULEURS  (ordre BGR — Blue, Green, Red — utilisé par OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

C_DARK   = (30,  30,  30)    # Fond sombre du panneau bas
C_WHITE  = (255, 255, 255)   # Contours des points, texte secondaire
C_YELLOW = (0,   220, 255)   # Score principal
C_GREEN  = (0,   200, 80)    # Détail par main
C_BLUE   = (220, 100, 20)    # Titre de l'application
C_JOINT  = (0,   150, 255)   # Articulations normales (orange)
C_BONE   = (180, 180, 180)   # Os / connexions entre points (gris clair)
C_TIP    = (0,   255, 100)   # Bouts de doigts (vert clair)
C_RED    = (0,   0,   255)   # Messages d'erreur caméra


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMÈTRES DU DÉTECTEUR MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────

MAX_HANDS                    = 2     # Nombre maximum de mains détectées simultanément
MIN_HAND_DETECTION_CONFIDENCE = 0.6  # Confiance minimale pour détecter une nouvelle main
MIN_HAND_PRESENCE_CONFIDENCE  = 0.5  # Confiance minimale pour confirmer qu'une main est présente
MIN_TRACKING_CONFIDENCE       = 0.5  # Confiance minimale pour le suivi entre frames


# ─────────────────────────────────────────────────────────────────────────────
#  DICTIONNAIRE CHIFFRE → NOM EN FRANÇAIS
# ─────────────────────────────────────────────────────────────────────────────

NOMBRES = {
    0: "ZERO",  1: "UN",    2: "DEUX",  3: "TROIS",
    4: "QUATRE", 5: "CINQ", 6: "SIX",  7: "SEPT",
    8: "HUIT",  9: "NEUF", 10: "DIX",
}

