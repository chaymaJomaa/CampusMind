
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "campusmind.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                day_of_week INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                exam_date TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                subject TEXT NOT NULL,
                deadline TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                completed INTEGER DEFAULT 0
            )
        """)


def add_course(subject, day, start, end):
    if start >= end:
        raise ValueError(
            "L'heure de fin doit être après l'heure de début."
        )

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO courses
               (subject, day_of_week, start_time, end_time)
               VALUES (?, ?, ?, ?)""",
            (subject, day, start, end)
        )


def add_exam(subject, exam_date):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO exams (subject, exam_date) VALUES (?, ?)",
            (subject, exam_date)
        )


def add_task(title, subject, deadline, duration):
    if duration <= 0:
        raise ValueError("La durée doit être positive.")

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO tasks
               (title, subject, deadline, duration_minutes)
               VALUES (?, ?, ?, ?)""",
            (title, subject, deadline, duration)
        )


def get_records(table):
    allowed_tables = {"courses", "exams", "tasks"}

    if table not in allowed_tables:
        raise ValueError("Table non autorisée.")

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY id DESC"
        ).fetchall()

    return [dict(row) for row in rows]
