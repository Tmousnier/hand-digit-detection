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

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import (
    MODEL_PATH, WINDOW_NAME, WIDTH, HEIGHT,
    MAX_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from core.camera        import initialize_camera
from core.hand_analyzer import HandAnalyzer
from core.model         import download_model_if_needed
from ui.overlay         import UIOverlay


def main() -> None:
    """
    Boucle principale de l'application.

    Étapes :
        1. Téléchargement du modèle IA (une seule fois)
        2. Initialisation de la webcam
        3. Configuration du détecteur MediaPipe
        4. Boucle vidéo : capture → détection → affichage → gestion du clavier
        5. Nettoyage (libération caméra + fermeture fenêtres)
    """

    # ── Étape 1 : Modèle IA ───────────────────────────────────────────────────
    download_model_if_needed()

    # ── Étape 2 : Caméra ──────────────────────────────────────────────────────
    cap = initialize_camera()

    if cap is None:
        # Aucune webcam valide → fenêtre d'erreur, puis on s'arrête
        print("[ERREUR] Aucune camera utilisable detectee.")
        UIOverlay.show_no_camera()
        return

    # ── Étape 3 : Fenêtre d'affichage ─────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)
    cv2.moveWindow(WINDOW_NAME, 100, 50)   # Position initiale à l'écran

    # ── Étape 4 : Détecteur MediaPipe ─────────────────────────────────────────
    # Mode VIDEO : traitement frame par frame avec timestamps réels.
    # Plus performant que le mode IMAGE pour un flux continu.
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

        # ── Vidage du buffer clavier ───────────────────────────────────────
        # Sur Windows, des touches résiduelles (ex : Entrée, Q) utilisées
        # pour lancer le script depuis l'IDE restent dans le buffer clavier.
        # On les élimine avant d'entrer dans la boucle.
        for _ in range(30):
            cv2.waitKey(1)

        # ── Garde de démarrage : immunité clavier de 4 secondes ───────────
        # Même après le vidage, certaines touches arrivent avec du retard.
        # On les ignore pendant les 4 premières secondes.
        start_time = time.time()
        print("[INFO] Touches ignorees pendant 4 secondes au demarrage...")

        # ── Boucle de traitement vidéo ─────────────────────────────────────
        while True:

            # Lecture d'une frame depuis la webcam
            ret, frame = cap.read()

            if not ret or frame is None:
                # Lecture échouée (ponctuelle) → on réessaie à la prochaine itération
                cv2.waitKey(10)
                continue

            frame_count += 1

            # Effet miroir : l'utilisateur voit sa main comme dans un vrai miroir
            frame = cv2.flip(frame, 1)

            # Vérification de la luminosité : si l'image est presque noire,
            # la caméra ne transmet rien → avertissement et passage à la frame suivante
            if frame.mean() < 5:
                UIOverlay.draw_camera_warning(frame, frame.mean())
                UIOverlay.draw_panel(frame, 0, [])
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)
                continue

            # ── Préparation pour MediaPipe ─────────────────────────────────
            # MediaPipe attend du RGB ; OpenCV fournit du BGR → conversion nécessaire
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Timestamp réel en millisecondes.
            # MediaPipe VIDEO exige des timestamps strictement croissants et
            # correspondant au temps réel. Un incrément fixe de +33 ms serait
            # incorrect si une frame met plus ou moins longtemps à arriver.
            timestamp_ms = int(time.time() * 1000)

            # ── Détection des mains ────────────────────────────────────────
            results = detector.detect_for_video(mp_image, timestamp_ms)

            total_fingers = 0    # Nombre total de doigts levés (toutes mains)
            hand_details  = []   # [(côté, score), ...] pour l'affichage détaillé

            if results.hand_landmarks and results.handedness:
                for landmarks, handedness_list in zip(
                    results.hand_landmarks, results.handedness
                ):
                    # Récupération du côté ("Right" ou "Left")
                    side = handedness_list[0].category_name

                    # Dessin du squelette de la main sur la frame
                    UIOverlay.draw_skeleton(frame, landmarks)

                    # Comptage des doigts levés pour cette main
                    score = HandAnalyzer.count_fingers(landmarks, side)

                    total_fingers += score
                    hand_details.append((side, score))

            # ── Affichage ──────────────────────────────────────────────────
            UIOverlay.draw_panel(frame, total_fingers, hand_details)
            cv2.imshow(WINDOW_NAME, frame)

            # ── Gestion du clavier ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            # Quitter avec Q ou Échap (uniquement après la garde de 4 secondes)
            if time.time() - start_time > 4.0:
                if key in [ord("q"), ord("Q"), 27]:
                    print("Touche quitter detectee.")
                    break

            # ── Détection de fermeture via la croix de la fenêtre ─────────
            # On attend 30 frames avant de surveiller pour éviter les faux positifs
            if frame_count > 30:
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        print("Fenetre fermee par l'utilisateur.")
                        break
                except cv2.error:
                    break

    # ── Étape 5 : Nettoyage ───────────────────────────────────────────────────
    cap.release()              # Libère la webcam pour les autres applications
    cv2.destroyAllWindows()    # Ferme toutes les fenêtres OpenCV
    print(">>> Application fermee proprement.")


if __name__ == "__main__":
    main()
