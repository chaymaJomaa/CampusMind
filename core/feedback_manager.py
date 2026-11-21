
"""
CampusMind - Gestion des feedbacks étudiants.

Objectif :
Collecter des interactions réelles afin de préparer
l'apprentissage de préférences individuelles.

0 = recommandation refusée
1 = recommandation acceptée

Les connexions SQLite sont fermées explicitement.
"""

import sqlite3
from pathlib import Path
from contextlib import closing, contextmanager
from datetime import datetime


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "campusmind_feedback.db"
)


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with closing(
        sqlite3.connect(str(DB_PATH), timeout=10)
    ) as conn:
        conn.row_factory = sqlite3.Row

        with conn:
            yield conn


def initialize_feedback():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                interaction_key TEXT NOT NULL UNIQUE,

                title TEXT NOT NULL,
                activity_type INTEGER NOT NULL,

                days_remaining INTEGER NOT NULL,
                duration_minutes INTEGER NOT NULL,
                free_time_minutes INTEGER NOT NULL,

                hour INTEGER NOT NULL,
                weekday INTEGER NOT NULL,

                accepted INTEGER NOT NULL
                    CHECK (accepted IN (0, 1)),

                created_at TEXT NOT NULL
            )
        """)


def save_feedback(
    recommendation,
    free_time,
    accepted,
    at=None
):
    """
    Enregistre une réponse de l'étudiant.

    Une recommandation donnée est enregistrée
    une fois par jour. Une nouvelle réponse
    remplace la précédente pour cette journée.
    """

    initialize_feedback()

    now = at or datetime.now()

    title = str(recommendation["title"])

    days = int(recommendation["days"])
    duration = int(recommendation["minutes"])
    free_time = int(free_time)

    activity_type = int(
        recommendation.get("activity_type", 0)
    )

    if activity_type not in (0, 1):
        raise ValueError("Type d'activité invalide.")

    if accepted not in (0, 1, False, True):
        raise ValueError("Feedback invalide.")

    if duration <= 0 or free_time < 0:
        raise ValueError("Durée invalide.")

    interaction_key = "|".join([
        now.date().isoformat(),
        str(activity_type),
        title
    ])

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO feedbacks (
                interaction_key,
                title,
                activity_type,
                days_remaining,
                duration_minutes,
                free_time_minutes,
                hour,
                weekday,
                accepted,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(interaction_key)
            DO UPDATE SET
                days_remaining = excluded.days_remaining,
                duration_minutes = excluded.duration_minutes,
                free_time_minutes = excluded.free_time_minutes,
                hour = excluded.hour,
                weekday = excluded.weekday,
                accepted = excluded.accepted,
                created_at = excluded.created_at

        """, (
            interaction_key,
            title,
            activity_type,
            days,
            duration,
            free_time,
            now.hour,
            now.weekday(),
            int(accepted),
            now.isoformat(timespec="seconds")
        ))


def get_feedbacks():

    initialize_feedback()

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT *
            FROM feedbacks
            ORDER BY created_at DESC
        """).fetchall()

        return [
            dict(row)
            for row in rows
        ]


def get_feedback_stats():

    initialize_feedback()

    with get_connection() as conn:

        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(accepted), 0) AS accepted
            FROM feedbacks
        """).fetchone()

    total = row["total"]
    accepted = row["accepted"]

    return {
        "total": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "acceptance_rate": (
            round(100 * accepted / total, 2)
            if total > 0 else 0
        )
    }


if __name__ == "__main__":

    initialize_feedback()

    print("Base de feedbacks prête.")
    print(get_feedback_stats())
