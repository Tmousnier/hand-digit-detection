"""
core/camera.py
──────────────
Gère la recherche et l'initialisation de la webcam.

Problèmes résolus :
  - Sur Windows, certains indices de caméra s'ouvrent mais retournent
    des frames vides (ret=False) ou un gris uniforme (placeholder virtuel).
  - On teste plusieurs backends (DSHOW, MSMF, ANY) pour maximiser la compatibilité.
  - On valide chaque caméra avec un double critère (luminosité + variance)
    qui distingue une vraie image d'un placeholder.
"""

import cv2

from config import WIDTH, HEIGHT


def initialize_camera() -> cv2.VideoCapture | None:
    """
    Parcourt les indices de caméras 0 à 3 et les backends Windows disponibles
    pour trouver le premier flux vidéo réel.

    Critères de validation d'un flux valide :
        - mean > 10  : l'image n'est pas entièrement noire
        - std  > 15  : l'image a de la variance (textures, couleurs variées)
                       Rejette les caméras virtuelles/placeholders qui renvoient
                       un gris parfaitement uniforme (std ≈ 0).

    Retourne :
        cv2.VideoCapture  — objet prêt à l'emploi si une caméra est trouvée
        None              — si aucune caméra valide n'est disponible
    """
    # Backends Windows testés dans cet ordre de préférence :
    #   DSHOW — DirectShow       : le plus compatible avec les webcams intégrées
    #   MSMF  — Media Foundation : natif Windows 8+, meilleur pour certaines caméras
    #   ANY   — Automatique      : OpenCV choisit lui-même le meilleur backend
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF",  cv2.CAP_MSMF),
        ("ANY",   cv2.CAP_ANY),
    ]

    for idx in range(4):
        for bname, bval in backends:
            cap = cv2.VideoCapture(idx, bval)

            # Si OpenCV ne peut pas ouvrir cet index/backend → on passe au suivant
            if not cap.isOpened():
                cap.release()
                continue

            # Paramètres de la caméra
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)   # Buffer de 3 frames pour éviter les saccades

            # ── Vérification rapide ────────────────────────────────────────────
            # Certaines caméras s'ouvrent (isOpened=True) mais ne renvoient
            # jamais de frame (ret=False). On lit 3 frames pour le détecter.
            quick_ok = sum(1 for _ in range(3) if cap.read()[0])
            if quick_ok == 0:
                cap.release()
                continue

            # ── Chauffe (warmup) ───────────────────────────────────────────────
            # On lit et jette 20 frames sans les traiter.
            # Certaines webcams sortent des frames noires ou corrompues
            # pendant les premières secondes après l'ouverture.
            for _ in range(20):
                cap.read()

            # ── Validation de la qualité d'image ──────────────────────────────
            # On cherche 3 frames consécutives qui passent le double critère :
            #
            #   mean > 10  → image non noire (caméra éteinte ou obturée : mean ≈ 0)
            #   std  > 15  → image non uniforme :
            #                   - placeholder gris Windows : std ≈ 0 (gris parfait)
            #                   - vraie caméra             : std > 15 (bruit, textures)
            valid = 0
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    mean_val = frame.mean()   # Luminosité moyenne  (0 = noir, 255 = blanc)
                    std_val  = frame.std()    # Écart-type des pixels (0 = uniforme)

                    if mean_val > 10.0 and std_val > 15.0:
                        valid += 1           # Frame valide : on incrémente
                    else:
                        valid = 0            # Frame invalide : on repart à zéro

                    if valid >= 3:
                        # 3 frames consécutives valides → vraie caméra trouvée
                        print(f"[OK] Camera detectee (index={idx}, {bname})")
                        return cap
                else:
                    valid = 0

            # Pas assez de frames valides → on essaie la combinaison suivante
            cap.release()

    # Aucune combinaison (index, backend) n'a donné un flux valide
    return None

