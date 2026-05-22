"""
ui/overlay.py
─────────────
Regroupe toutes les fonctions de dessin OpenCV affichées par-dessus
l'image de la caméra :
  - draw_skeleton      : squelette de la main (os + articulations)
  - draw_panel         : panneau d'information en bas (score, titre, aide)
  - draw_camera_warning: message rouge si l'image est trop sombre
  - show_no_camera     : fenêtre de remplacement si aucune caméra n'est trouvée
"""

import cv2
import numpy as np

from config import (
    WIDTH, HEIGHT, WINDOW_NAME, NOMBRES,
    C_DARK, C_WHITE, C_YELLOW, C_GREEN, C_BLUE,
    C_JOINT, C_BONE, C_TIP, C_RED,
)
from core.hand_analyzer import HandAnalyzer


class UIOverlay:
    """
    Classe utilitaire regroupant les rendus graphiques OpenCV.
    Toutes les méthodes sont statiques : pas besoin d'instancier la classe.
    """

    @staticmethod
    def draw_skeleton(frame, landmarks) -> None:
        """
        Dessine le squelette complet d'une main détectée sur la frame courante.

        Étapes :
            1. Conversion des coordonnées normalisées (0.0 → 1.0) en pixels
               réels en multipliant par la largeur / hauteur de l'image.
            2. Tracé des os (lignes grises entre chaque paire de landmarks).
            3. Tracé des articulations (cercles sur chaque landmark).
               Les bouts de doigts (TIPs) sont mis en évidence en vert vif.

        Paramètres :
            frame     — image BGR OpenCV à modifier (modifiée en place)
            landmarks — liste des 21 NormalizedLandmark de MediaPipe
        """
        h, w = frame.shape[:2]

        # Conversion coordonnées normalisées → pixels
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # ── Dessin des os (lignes entre chaque paire de points) ───────────────
        for start, end in HandAnalyzer.CONNECTIONS:
            if start < len(pts) and end < len(pts):
                cv2.line(frame, pts[start], pts[end], C_BONE, 2, cv2.LINE_AA)

        # ── Dessin des articulations (cercles sur chaque point) ───────────────
        for idx, (x, y) in enumerate(pts):
            is_tip = idx in HandAnalyzer.FINGER_TIPS
            color  = C_TIP   if is_tip else C_JOINT   # Vert si TIP, orange sinon
            size   = 6       if is_tip else 4          # Plus grand pour les TIPs
            cv2.circle(frame, (x, y), size, color,   -1, cv2.LINE_AA)  # Cercle plein
            cv2.circle(frame, (x, y), size, C_WHITE,  1, cv2.LINE_AA)  # Contour blanc

    @staticmethod
    def draw_panel(frame, total_score: int, hand_details: list, fps: float = 0.0) -> None:
        """
        Affiche le panneau d'information superposé en bas et en haut de l'image.

        Contenu :
            • En bas   — fond semi-transparent + score total centré + détail par main
            • En haut  — titre à gauche + FPS au centre + raccourci "[Q] Quitter" à droite

        Paramètres :
            frame        — image BGR OpenCV à modifier (modifiée en place)
            total_score  — nombre total de doigts levés LISSÉ (médiane glissante)
            hand_details — liste de tuples (côté: str, score: int) pour chaque main
            fps          — fréquence d'images calculée dans main.py (0 si inconnue)
        """
        h, w = frame.shape[:2]

        # ── Fond semi-transparent en bas de l'image ───────────────────────────
        # On copie l'image, on peint un rectangle sombre par-dessus,
        # puis on mélange (75 % opaque) avec l'original = effet translucide.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 120), (w, h), C_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # ── Score principal (centré horizontalement, avec ombre) ──────────────
        # 2 mains détectées → afficher l'opération complète : "G + D = total - NOM"
        # 1 main ou 0      → afficher simplement            : "score - NOM"
        if len(hand_details) == 2:
            scores      = {side: score for side, score in hand_details}
            left_score  = scores.get("Left",  0)
            right_score = scores.get("Right", 0)
            text_score  = (
                f"{left_score}  +  {right_score}  =  "
                f"{total_score}  -  {NOMBRES.get(total_score, '')}"
            )
            # Police plus petite pour que la ligne rentre dans la fenêtre
            font_scale = 1.3
        else:
            text_score = f"{total_score}  -  {NOMBRES.get(total_score, '')}"
            font_scale = 1.8

        # Mesure la taille du texte pour le centrer
        text_size, _ = cv2.getTextSize(text_score, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
        pos_score  = ((w - text_size[0]) // 2, h - 45)
        shadow_pos = (pos_score[0] + 2, pos_score[1] + 2)   # Décalage 2 px = ombre portée

        # Ombre épaisse noire, puis texte jaune par-dessus
        cv2.putText(frame, text_score, shadow_pos,
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, text_score, pos_score,
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, C_YELLOW, 2, cv2.LINE_AA)

        # ── Détail par main (bas gauche, une entrée par main détectée) ────────
        x_offset = 30
        for side, score in hand_details:
            label = "Main D." if side == "Right" else "Main G."
            cv2.putText(frame, f"{label}: {score}", (x_offset, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GREEN, 2, cv2.LINE_AA)
            x_offset += 200   # Décalage horizontal pour la 2e main

        # ── Titre en haut à gauche ────────────────────────────────────────────
        cv2.putText(frame, "Vision IA : Chiffres avec les Mains", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_BLUE, 2, cv2.LINE_AA)

        # ── FPS en haut au centre ─────────────────────────────────────────────
        # Couleur adaptative selon la fluidité :
        #   vert  (≥ 24 fps) → fluide
        #   orange (≥ 15 fps) → acceptable
        #   rouge  (< 15 fps) → lent
        fps_text  = f"{fps:.1f} FPS"
        fps_color = C_GREEN if fps >= 24 else (0, 165, 255) if fps >= 15 else C_RED
        fps_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        fps_x = (w - fps_size[0]) // 2
        cv2.putText(frame, fps_text, (fps_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)

        # ── Raccourci clavier en haut à droite ────────────────────────────────
        cv2.putText(frame, "[Q] Quitter", (w - 140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def draw_camera_warning(frame, mean_value: float) -> None:
        """
        Affiche un avertissement en rouge sur la frame quand la webcam est
        ouverte, mais renvoie une image trop sombre (luminosité moyenne < 5).

        Causes possibles : caméra obturée, utilisée par une autre application
        (Teams, Zoom, OBS, etc.), ou pilote défaillant.

        Paramètres :
            frame      — image BGR OpenCV à modifier (modifiée en place)
            mean_value — luminosité moyenne calculée sur la frame (0 à 255)
        """
        cv2.putText(frame, "Camera detectee mais image noire",
                    (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, C_RED, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Luminosite moyenne : {mean_value:.2f}",
                    (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)
        cv2.putText(frame, "Ferme Teams / Discord / OBS / navigateur puis relance.",
                    (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)

    @staticmethod
    def show_no_camera() -> None:
        """
        Affiche une fenêtre de remplacement grise avec un message d'erreur
        quand aucune caméra utilisable n'a été trouvée au démarrage.

        L'utilisateur peut fermer la fenêtre en appuyant sur Q, Échap
        ou en cliquant sur la croix.
        """
        # Création d'une image grise unie comme fond neutre
        background = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        background[:] = (150, 160, 165)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

        while True:
            display = background.copy()

            # Message d'erreur principal
            cv2.putText(display, "Aucune camera utilisable detectee",
                        (WIDTH // 2 - 360, HEIGHT // 2 - 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, C_WHITE, 2, cv2.LINE_AA)

            # Conseil de dépannage
            cv2.putText(display, "Teste l'application Camera Windows puis relance Python.",
                        (WIDTH // 2 - 420, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WHITE, 2, cv2.LINE_AA)

            # Raccourci pour quitter
            cv2.putText(display, "[Q] Quitter",
                        (WIDTH // 2 - 80, HEIGHT // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WHITE, 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:   # Q ou Échap
                break

            # Fermeture via la croix de la fenêtre
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        cv2.destroyAllWindows()

