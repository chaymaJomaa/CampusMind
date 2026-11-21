
"""
Gestion des rappels CampusMind.

Base de données séparée des données académiques.

Événements locaux Streamlit :
- Le navigateur doit rester ouvert.
- Aucune notification Windows, email ou push.
- Gestion des rappels affichés et différés.
- Historique des notifications.
- Protection contre les doublons.
"""

import sqlite3
from pathlib import Path
from contextlib import closing, contextmanager


# ============================================================
# 1. CONFIGURATION
# ============================================================

DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "campusmind_notifications.db"
)


# ============================================================
# 2. GESTION SÉCURISÉE DES CONNEXIONS SQLITE
# ============================================================

@contextmanager
def database_connection(db_path=None):
    """
    Ouvre une connexion SQLite et garantit sa fermeture.

    - Commit automatique si tout fonctionne.
    - Rollback en cas d'erreur.
    - Fermeture systématique de la connexion.
    """

    path = Path(db_path) if db_path is not None else DB_PATH

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with closing(
        sqlite3.connect(str(path), timeout=10)
    ) as conn:

        conn.row_factory = sqlite3.Row

        with conn:
            yield conn


# ============================================================
# 3. INITIALISATION DE LA BASE DE DONNÉES
# ============================================================

def initialize(db_path=None):
    """
    Crée la table des notifications si elle n'existe pas.
    """

    path = Path(db_path) if db_path is not None else DB_PATH

    with database_connection(path) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS notification_events (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                item_key TEXT NOT NULL UNIQUE,

                title TEXT NOT NULL,

                status TEXT NOT NULL
                CHECK (
                    status IN ('deferred', 'displayed')
                ),

                context_time TEXT NOT NULL,

                updated_at TEXT NOT NULL

            )
        """)

    return path


# ============================================================
# 4. DÉCISION CONTEXTUELLE ET ENREGISTREMENT
# ============================================================

def decide_and_save(
    recommendation,
    *,
    available,
    enabled,
    at,
    db_path=None
):
    """
    Détermine le comportement de la notification.

    Valeurs possibles :
    - disabled  : notifications désactivées
    - none      : aucune recommandation
    - deferred  : notification reportée
    - displayed : notification affichée
    - duplicate : notification déjà affichée

    Une même activité est affichée au maximum
    une fois par date simulée ou réelle.
    """

    if not enabled:
        return "disabled"

    if not recommendation:
        return "none"

    path = initialize(db_path)

    # Clé unique pour éviter les doublons
    key = "|".join([
        at.date().isoformat(),
        str(recommendation.get("type", "Activité")),
        str(recommendation["title"]),
        str(recommendation.get("days", ""))
    ])

    stamp = at.isoformat(timespec="minutes")

    with database_connection(path) as conn:

        old = conn.execute("""
            SELECT status
            FROM notification_events
            WHERE item_key = ?
        """, (key,)).fetchone()

        # Notification déjà affichée
        if old and old["status"] == "displayed":
            return "duplicate"

        # Étudiant indisponible
        if not available:

            if not old:

                conn.execute("""
                    INSERT INTO notification_events
                    (
                        item_key,
                        title,
                        status,
                        context_time,
                        updated_at
                    )
                    VALUES (?, ?, 'deferred', ?, ?)
                """, (
                    key,
                    recommendation["title"],
                    stamp,
                    stamp
                ))

            return "deferred"

        # Notification auparavant différée
        if old:

            conn.execute("""
                UPDATE notification_events

                SET status = 'displayed',
                    updated_at = ?

                WHERE item_key = ?
            """, (
                stamp,
                key
            ))

        # Nouvelle notification
        else:

            conn.execute("""
                INSERT INTO notification_events
                (
                    item_key,
                    title,
                    status,
                    context_time,
                    updated_at
                )
                VALUES (?, ?, 'displayed', ?, ?)
            """, (
                key,
                recommendation["title"],
                stamp,
                stamp
            ))

        return "displayed"


# ============================================================
# 5. RÉCUPÉRATION DES NOTIFICATIONS EN ATTENTE
# ============================================================

def pending_items(db_path=None, limit=10):
    """
    Retourne les notifications différées.
    """

    path = initialize(db_path)

    with database_connection(path) as conn:

        rows = conn.execute("""
            SELECT
                id,
                title,
                context_time

            FROM notification_events

            WHERE status = 'deferred'

            ORDER BY id ASC

            LIMIT ?
        """, (limit,)).fetchall()

        results = [
            dict(row)
            for row in rows
        ]

    return results


# ============================================================
# 6. LIBÉRATION DES NOTIFICATIONS DIFFÉRÉES
# ============================================================

def release_pending(
    *,
    available,
    enabled,
    at,
    db_path=None
):
    """
    Libère au maximum une notification différée
    de la date actuelle.

    Le rappel est libéré uniquement si :
    - l'étudiant est disponible ;
    - les notifications sont activées.
    """

    if not available or not enabled:
        return None

    path = initialize(db_path)

    with database_connection(path) as conn:

        row = conn.execute("""
            SELECT
                id,
                title

            FROM notification_events

            WHERE status = 'deferred'

              AND substr(item_key, 1, 10) = ?

            ORDER BY id ASC

            LIMIT 1
        """, (
            at.date().isoformat(),
        )).fetchone()

        if row is None:
            return None

        notification_id = row["id"]
        notification_title = row["title"]

        conn.execute("""
            UPDATE notification_events

            SET status = 'displayed',
                updated_at = ?

            WHERE id = ?
        """, (
            at.isoformat(timespec="minutes"),
            notification_id
        ))

    return notification_title


# ============================================================
# 7. HISTORIQUE DES NOTIFICATIONS
# ============================================================

def history(db_path=None, limit=30):
    """
    Retourne l'historique des notifications.

    Les notifications sont triées par ordre
    décroissant d'enregistrement.
    """

    path = initialize(db_path)

    with database_connection(path) as conn:

        rows = conn.execute("""
            SELECT
                title,
                status,
                context_time,
                updated_at

            FROM notification_events

            ORDER BY id DESC

            LIMIT ?
        """, (limit,)).fetchall()

        results = [
            dict(row)
            for row in rows
        ]

    return results
