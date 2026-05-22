"""
core/model.py
─────────────
Gère le téléchargement automatique du modèle MediaPipe Hand Landmarker.

Le modèle est un fichier .task (~40 Mo) fourni par Google.
Il est téléchargé une seule fois dans le dossier personnel de l'utilisateur
afin d'éviter les erreurs de MediaPipe sur les chemins avec accents/espaces.
"""

import os
import urllib.request

from config import MODEL_PATH, MODEL_URL


def download_model_if_needed() -> None:
    """
    Vérifie si le modèle est déjà présent sur le disque.
    Si non, le télécharge depuis les serveurs Google MediaPipe.
    """
    if not os.path.exists(MODEL_PATH):
        print("Telechargement du modele MediaPipe (une seule fois)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OK] Modele telecharge.")

