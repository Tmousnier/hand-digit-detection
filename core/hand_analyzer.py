"""
core/hand_analyzer.py
─────────────────────
Contient la logique métier de l'analyse des mains :
  - Définition du squelette (connexions entre les 21 landmarks)
  - Comptage des doigts levés

MediaPipe renvoie 21 points (landmarks) numérotés 0 à 20 :
  ┌─────────┬──────────────────────────────────────────┐
  │ Point 0 │ Poignet                                  │
  │ 1 – 4   │ Pouce       : CMC, MCP, IP, TIP          │
  │ 5 – 8   │ Index       : MCP, PIP, DIP, TIP         │
  │ 9 – 12  │ Majeur      : MCP, PIP, DIP, TIP         │
  │ 13 – 16 │ Annulaire   : MCP, PIP, DIP, TIP         │
  │ 17 – 20 │ Auriculaire : MCP, PIP, DIP, TIP         │
  └─────────┴──────────────────────────────────────────┘
"""


class HandAnalyzer:
    """
    Analyse statique des landmarks d'une main détectée par MediaPipe.
    Toutes les méthodes sont statiques : pas besoin d'instancier la classe.
    """

    # ── Indices des landmarks clés ────────────────────────────────────────────

    # Bouts de doigts (TIP) — point le plus distal de chaque doigt
    FINGER_TIPS  = [4, 8, 12, 16, 20]

    # Articulations de référence (MCP/IP) — point intermédiaire de comparaison
    # On compare TIP vs BASE pour savoir si le doigt est levé ou plié
    FINGER_BASES = [2, 6, 10, 14, 18]

    # ── Connexions pour le dessin du squelette ────────────────────────────────
    # Chaque tuple (A, B) = tracer une ligne du point A au point B
    CONNECTIONS = [
        # Pouce
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Majeur
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Annulaire
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Auriculaire
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Paume — connexions transversales entre la base des doigts
        (5, 9), (9, 13), (13, 17), (0, 17),
    ]

    @staticmethod
    def count_fingers(landmarks, handedness_label: str) -> int:
        """
        Compte et retourne le nombre de doigts levés (0 à 5).

        Paramètres :
            landmarks        — liste des 21 NormalizedLandmark (x, y, z normalisés)
            handedness_label — "Right" ou "Left" tel que retourné par MediaPipe

        Algorithme :
            ┌───────────────┬────────────────────────────────────────────────────┐
            │ Doigt         │ Critère "levé"                                     │
            ├───────────────┼────────────────────────────────────────────────────┤
            │ Pouce         │ Comparaison sur l'axe X (le pouce se déplace       │
            │               │ latéralement). La logique est inversée selon la    │
            │               │ main car l'image est en miroir.                    │
            ├───────────────┼────────────────────────────────────────────────────┤
            │ Index → Aur.  │ TIP.y < BASE.y  (axe Y inversé en image :         │
            │               │ vers le haut = valeur Y plus petite)               │
            └───────────────┴────────────────────────────────────────────────────┘
        """
        count = 0

        # ── Pouce ──────────────────────────────────────────────────────────────
        thumb_tip_x  = landmarks[HandAnalyzer.FINGER_TIPS[0]].x
        thumb_base_x = landmarks[HandAnalyzer.FINGER_BASES[0]].x

        # Main droite (image miroir) : pouce levé → pointe vers la gauche (X plus petit)
        if handedness_label == "Right":
            if thumb_tip_x < thumb_base_x:
                count += 1
        # Main gauche : pouce levé → pointe vers la droite (X plus grand)
        else:
            if thumb_tip_x > thumb_base_x:
                count += 1

        # ── Index, Majeur, Annulaire, Auriculaire ──────────────────────────────
        # Doigt levé = son TIP est au-dessus de sa base (Y TIP < Y base)
        for i in range(1, 5):
            tip_y  = landmarks[HandAnalyzer.FINGER_TIPS[i]].y
            base_y = landmarks[HandAnalyzer.FINGER_BASES[i]].y
            if tip_y < base_y:
                count += 1

        return count

