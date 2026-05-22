"""
main.py — Point d'entrée de l'application
──────────────────────────────────────────
Détection et Reconnaissance de Chiffres (0 à 10) avec les Mains

Structure du projet :
    main.py                ← ce fichier : orchestre toutes les étapes
    config.py              ← constantes : couleurs, chemins, seuils MediaPipe
    requirements.txt       ← dépendances pip
    │
    core/
    ├── camera.py          ← recherche et validation de la webcam
    ├── hand_analyzer.py   ← comptage des doigts levés (21 landmarks)
    └── model.py           ← téléchargement automatique du modèle IA
    │
    ui/
    └── overlay.py         ← dessin squelette, panneau de score, alertes

Utilisation :
    Lancez ce fichier directement. Pressez [Q] ou [Échap] pour quitter.
"""

import time
import statistics
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Nombre de frames conservées pour le lissage du score (médiane glissante)
# → évite les clignotements quand MediaPipe hésite entre deux valeurs
SMOOTH_WINDOW = 7

from config import (
    MODEL_PATH, WINDOW_NAME, WIDTH, HEIGHT,
    MAX_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODES,
)
from core.camera        import initialize_camera
from core.hand_analyzer import HandAnalyzer
from core.model         import download_model_if_needed
from ui.overlay         import UIOverlay

# Nombre de frames conservées pour le lissage (médiane glissante)
SMOOTH_WINDOW = 7


def main() -> None:
    """
    Boucle principale de l'application.

    Étapes :
        1. Téléchargement du modèle IA (une seule fois)
        2. Initialisation de la webcam
        3. Configuration du détecteur MediaPipe
        4. Boucle vidéo : capture → détection → lissage → affichage → clavier
        5. Nettoyage (libération caméra + fermeture fenêtres)
    """

    # ── Étape 1 : Modèle IA ───────────────────────────────────────────────────
    download_model_if_needed()

    # ── Étape 2 : Caméra ──────────────────────────────────────────────────────
    cap = initialize_camera()

    if cap is None:
        print("[ERREUR] Aucune camera utilisable detectee.")
        UIOverlay.show_no_camera()
        return

    # ── Étape 3 : Fenêtre d'affichage ─────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # ── Étape 4 : Détecteur MediaPipe ─────────────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    print("\n>>> Application demarree. Pressez [Q] pour quitter.")

    with vision.HandLandmarker.create_from_options(options) as detector:
        frame_count = 0

        # File glissante pour le lissage du score total (médiane sur SMOOTH_WINDOW frames)
        # Empêche les sauts de valeur quand MediaPipe hésite entre deux résultats
        score_buffer = deque(maxlen=SMOOTH_WINDOW)

        # ── Mode opératoire ───────────────────────────────────────────────────
        # Touche M pour cycler entre Addition / Soustraction / Multiplication / Division
        mode_index = 0   # Index dans la liste MODES de config.py

        # ── Mode binaire ──────────────────────────────────────────────────────
        # Touche B pour activer / désactiver le comptage binaire (0 à 31 par main)
        # En mode binaire : chaque doigt = un bit (pouce=1, index=2, majeur=4, ...)
        binary_mode = False

        # ── Calcul FPS ────────────────────────────────────────────────────────
        fps          = 0.0
        fps_timer    = time.time()   # Temps du dernier calcul FPS
        fps_counter  = 0             # Nombre de frames depuis le dernier calcul

        # ── Vidage du buffer clavier ───────────────────────────────────────────
        for _ in range(30):
            cv2.waitKey(1)

        # ── Garde de démarrage : immunité clavier de 4 secondes ───────────────
        start_time = time.time()
        print("[INFO] Touches ignorees pendant 4 secondes au demarrage...")

        # ── Boucle de traitement vidéo ─────────────────────────────────────────
        while True:

            ret, frame = cap.read()

            if not ret or frame is None:
                cv2.waitKey(10)
                continue

            frame_count  += 1
            fps_counter  += 1

            # ── Calcul du FPS (mis à jour toutes les secondes) ─────────────────
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps        = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.time()

            # Effet miroir
            frame = cv2.flip(frame, 1)

            # Vérification luminosité
            if frame.mean() < 5:
                UIOverlay.draw_camera_warning(frame, frame.mean())
                UIOverlay.draw_panel(frame, 0, [], fps)
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)
                continue

            # ── Préparation pour MediaPipe ─────────────────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            # ── Détection des mains ────────────────────────────────────────────
            results = detector.detect_for_video(mp_image, timestamp_ms)

            raw_total    = 0
            hand_details = []   # [(side, score, bits), ...]  bits=None si mode normal

            if results.hand_landmarks and results.handedness:
                for landmarks, handedness_list in zip(
                    results.hand_landmarks, results.handedness
                ):
                    side = handedness_list[0].category_name
                    UIOverlay.draw_skeleton(frame, landmarks)

                    if binary_mode:
                        # Mode binaire : chaque doigt = un bit, valeur 0-31
                        bits, score = HandAnalyzer.fingers_bits(landmarks, side)
                    else:
                        # Mode normal : compter le nombre de doigts levés (0-5)
                        bits  = None
                        score = HandAnalyzer.count_fingers(landmarks, side)

                    raw_total += score
                    hand_details.append((side, score, bits))

            # ── Lissage du score total (médiane glissante) ─────────────────────
            # On empile les scores bruts des dernières SMOOTH_WINDOW frames,
            # puis on prend la médiane pour éliminer les valeurs aberrantes.
            score_buffer.append(raw_total)
            smooth_total = int(statistics.median(score_buffer))

            # ── Affichage ──────────────────────────────────────────────────────
            UIOverlay.draw_panel(frame, smooth_total, hand_details, fps,
                                 MODES[mode_index], binary_mode)
            cv2.imshow(WINDOW_NAME, frame)

            # ── Gestion du clavier ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if time.time() - start_time > 4.0:
                if key in [ord("q"), ord("Q"), 27]:
                    print("Touche quitter detectee.")
                    break
                # Touche M → cycle entre les 4 modes opératoires
                if key in [ord("m"), ord("M")]:
                    mode_index = (mode_index + 1) % len(MODES)
                    print(f"[MODE] {MODES[mode_index]['label']}")
                # Touche B → active / désactive le mode binaire
                if key in [ord("b"), ord("B")]:
                    binary_mode = not binary_mode
                    state = "ON" if binary_mode else "OFF"
                    print(f"[BINAIRE] Mode binaire {state}")

            # ── Détection de fermeture via la croix ────────────────────────────
            if frame_count > 30:
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        print("Fenetre fermee par l'utilisateur.")
                        break
                except cv2.error:
                    break

    # ── Étape 5 : Nettoyage ───────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print(">>> Application fermee proprement.")


if __name__ == "__main__":
    main()