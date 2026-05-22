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
    C_JOINT, C_BONE, C_TIP, C_RED, C_ORANGE,
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
    def draw_panel(frame, total_score: int, hand_details: list,
                   fps: float = 0.0, mode: dict = None,
                   binary_mode: bool = False) -> None:
        """
        Affiche le panneau d'information superposé en bas et en haut de l'image.

        Paramètres :
            frame        — image BGR OpenCV à modifier (modifiée en place)
            total_score  — score total lissé
            hand_details — liste de tuples (côté, score, bits) par main
                           bits = liste [0/1] × 5 en mode binaire, None sinon
            fps          — fréquence d'images calculée dans main.py
            mode         — dict {"symbol": "+", "label": "Addition"}
            binary_mode  — True = comptage binaire actif (0 à 31 par main)
        """
        if mode is None:
            mode = {"symbol": "+", "label": "Addition"}

        h, w = frame.shape[:2]

        # ── Fond semi-transparent en bas de l'image ───────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 120), (w, h), C_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # ── Calcul et texte du résultat ───────────────────────────────────────
        if len(hand_details) == 2:
            scores      = {side: score for side, score in
                           [(s, sc) for s, sc, _ in hand_details]}
            left_score  = scores.get("Left",  0)
            right_score = scores.get("Right", 0)
            sym         = mode["symbol"]

            if sym == "+":
                result = left_score + right_score
                result_str = str(result)
            elif sym == "-":
                result = left_score - right_score
                result_str = str(result)
            elif sym == "x":
                result = left_score * right_score
                result_str = str(result)
            else:
                if right_score == 0:
                    result     = None
                    result_str = "Err /0"
                else:
                    raw        = left_score / right_score
                    result     = raw
                    result_str = str(int(raw)) if raw == int(raw) else f"{raw:.2f}"

            nom = ""
            if result is not None and isinstance(result, (int, float)):
                if result == int(result) and 0 <= int(result) <= 10:
                    nom = NOMBRES.get(int(result), "")
            suffix     = f"  -  {nom}" if nom else ""
            text_score = f"{left_score}  {sym}  {right_score}  =  {result_str}{suffix}"
            font_scale = 1.3
        elif len(hand_details) == 1 and binary_mode:
            # Mode binaire 1 main : afficher "01011 = 19"
            _, score, bits = hand_details[0]
            bits_str   = "".join(str(b) for b in reversed(bits))  # MSB à gauche
            text_score = f"{bits_str}  =  {score}"
            font_scale = 1.8
        else:
            score_val  = hand_details[0][1] if hand_details else total_score
            text_score = f"{score_val}  -  {NOMBRES.get(score_val, '')}"
            font_scale = 1.8

        # ── Score centré avec ombre ────────────────────────────────────────────
        text_size, _ = cv2.getTextSize(text_score, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
        pos_score  = ((w - text_size[0]) // 2, h - 45)
        shadow_pos = (pos_score[0] + 2, pos_score[1] + 2)

        cv2.putText(frame, text_score, shadow_pos,
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, text_score, pos_score,
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, C_YELLOW, 2, cv2.LINE_AA)

        # ── Détail par main (bas gauche) ──────────────────────────────────────
        x_offset = 30
        for side, score, bits in hand_details:
            label = "Main D." if side == "Right" else "Main G."
            if binary_mode and bits is not None:
                bits_str   = "".join(str(b) for b in reversed(bits))
                detail_txt = f"{label}: {bits_str} = {score}"
            else:
                detail_txt = f"{label}: {score}"
            cv2.putText(frame, detail_txt, (x_offset, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GREEN, 2, cv2.LINE_AA)
            x_offset += 280

        # ── Titre en haut à gauche ────────────────────────────────────────────
        cv2.putText(frame, "Vision IA : Chiffres avec les Mains", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_BLUE, 2, cv2.LINE_AA)

        # ── FPS en haut au centre ─────────────────────────────────────────────
        fps_text  = f"{fps:.1f} FPS"
        fps_color = C_GREEN if fps >= 24 else (0, 165, 255) if fps >= 15 else C_RED
        fps_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        fps_x = (w - fps_size[0]) // 2
        cv2.putText(frame, fps_text, (fps_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)

        # ── Raccourci clavier en haut à droite ────────────────────────────────
        cv2.putText(frame, "[Q] Quitter", (w - 140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1, cv2.LINE_AA)

        # ── Badge mode opératoire (à droite, sous [Q] Quitter) ────────────────
        mode_text  = f"[M] {mode['label']}"
        badge_size, _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        bx = w - badge_size[0] - 8
        by = 68
        cv2.rectangle(frame, (bx - 4, by - 16), (bx + badge_size[0] + 4, by + 4),
                      C_ORANGE, -1)
        cv2.putText(frame, mode_text, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_DARK, 1, cv2.LINE_AA)

        # ── Badge mode binaire (à droite, sous le badge mode) ────────────────
        bin_label  = "[B] Binaire : ON" if binary_mode else "[B] Binaire : OFF"
        bin_color  = C_GREEN if binary_mode else (100, 100, 100)   # Vert=ON, Gris=OFF
        bin_size, _ = cv2.getTextSize(bin_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        bx2 = w - bin_size[0] - 8
        by2 = 96   # Juste sous le badge mode (68 + ~28px)
        cv2.rectangle(frame, (bx2 - 4, by2 - 16), (bx2 + bin_size[0] + 4, by2 + 4),
                      bin_color, -1)
        cv2.putText(frame, bin_label, (bx2, by2),
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

