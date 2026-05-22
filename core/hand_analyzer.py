"""
core/hand_analyzer.py
─────────────────────
Contient la logique métier de l'analyse des mains :
  - Définition du squelette (connexions entre les 21 landmarks)
  - Détection de l'orientation de la main (paume ou dos face à la caméra)
  - Comptage des doigts levés (fonctionne dans les deux orientations)

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
    def is_palm_facing(landmarks, handedness_label: str) -> bool:
        """
        Détermine si la PAUME fait face à la caméra (True)
        ou si c'est le DOS de la main (False).

        Méthode : produit vectoriel dans le plan image entre :
            - Vecteur A : poignet (0) → index MCP (5)
            - Vecteur B : poignet (0) → auriculaire MCP (17)

        La composante Z du produit vectoriel (Ax·By − Ay·Bx) indique
        l'orientation de la main dans le plan image :

            Main droite  →  cross_z > 0  =  paume face caméra
                            cross_z < 0  =  dos face caméra
            Main gauche  →  cross_z < 0  =  paume face caméra
                            cross_z > 0  =  dos face caméra

        Intuition géométrique :
            Dos  vers caméra : index MCP est à DROITE du auriculaire MCP
            Paume vers caméra : index MCP est à GAUCHE du auriculaire MCP
            → le signe du produit vectoriel capture ce changement de côté.
        """
        lm0  = landmarks[0]    # Poignet
        lm5  = landmarks[5]    # Index MCP (base de l'index)
        lm17 = landmarks[17]   # Auriculaire MCP (base de l'auriculaire)

        # Vecteurs depuis le poignet dans le plan image (x, y normalisés)
        ax = lm5.x  - lm0.x
        ay = lm5.y  - lm0.y
        bx = lm17.x - lm0.x
        by = lm17.y - lm0.y

        # Composante Z du produit vectoriel A × B
        cross_z = ax * by - ay * bx

        # Le signe s'interprète différemment selon la main (miroir horizontal)
        if handedness_label == "Right":
            return cross_z > 0   # Positif = paume face caméra pour main droite
        else:
            return cross_z < 0   # Négatif = paume face caméra pour main gauche

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
            │ Pouce         │ Comparaison sur l'axe X (déplacement latéral).     │
            │               │ La direction "extérieur" s'inverse quand on retourne│
            │               │ la main → on détecte l'orientation pour corriger.  │
            ├───────────────┼────────────────────────────────────────────────────┤
            │ Index → Aur.  │ TIP.y < BASE.y  (axe Y inversé en image).         │
            │               │ Valable quelle que soit l'orientation de la main.  │
            └───────────────┴────────────────────────────────────────────────────┘
        """
        count = 0

        # ── Détection de l'orientation de la main ─────────────────────────────
        # La logique du POUCE dépend de si la paume ou le dos fait face à la caméra.
        # Les 4 autres doigts (axe Y) ne sont pas affectés par l'orientation.
        palm_facing = HandAnalyzer.is_palm_facing(landmarks, handedness_label)

        # ── Pouce ──────────────────────────────────────────────────────────────
        thumb_tip_x  = landmarks[HandAnalyzer.FINGER_TIPS[0]].x
        thumb_base_x = landmarks[HandAnalyzer.FINGER_BASES[0]].x

        # Quand le DOS fait face à la caméra (position naturelle) :
        #   Main droite → pouce levé pointe vers la GAUCHE  (tip.x < base.x)
        #   Main gauche → pouce levé pointe vers la DROITE  (tip.x > base.x)
        #
        # Quand la PAUME fait face à la caméra, la main est retournée :
        #   La direction du pouce levé s'inverse dans les deux cas.
        #   → on inverse simplement la condition avec `not palm_facing`.

        if handedness_label == "Right":
            thumb_up = thumb_tip_x < thumb_base_x   # Règle pour dos face caméra
        else:
            thumb_up = thumb_tip_x > thumb_base_x   # Règle pour dos face caméra

        if palm_facing:
            thumb_up = not thumb_up   # Inversion quand paume face caméra

        if thumb_up:
            count += 1

        # ── Index, Majeur, Annulaire, Auriculaire ──────────────────────────────
        # Un doigt est levé si son TIP est AU-DESSUS de sa base.
        # En image, "au-dessus" = coordonnée Y plus petite (axe Y inversé).
        # Cette règle est valable que la paume ou le dos soit face à la caméra,
        # car un doigt levé pointe toujours vers le haut dans les deux cas.
        for i in range(1, 5):
            tip_y  = landmarks[HandAnalyzer.FINGER_TIPS[i]].y
            base_y = landmarks[HandAnalyzer.FINGER_BASES[i]].y
            if tip_y < base_y:
                count += 1

        return count
