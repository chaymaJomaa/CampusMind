
"""
CampusMind - Planificateur académique intelligent.

Fonctionnalités :
- Respect des horaires de cours
- Identification des créneaux libres
- Classement selon les échéances
- Découpage des grandes tâches
- Pauses entre les séances
- Construction d'un planning quotidien

Le moteur utilise des contraintes et des
heuristiques explicables. Il ne remplace
pas le modèle ML personnalisé.
"""

from datetime import datetime, date, time, timedelta


def to_minutes(value):
    """Convertit HH:MM en minutes depuis minuit."""
    h, m = map(int, value.split(":"))
    return h * 60 + m


def to_hhmm(minutes):
    """Convertit des minutes en HH:MM."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def priority_score(days, activity_type):
    """Score heuristique d'urgence."""

    if days < 0:
        return 130 if activity_type == "Tâche" else 0

    if days == 0:
        return 120

    if days == 1:
        return 110

    if days <= 3:
        return 95

    if days <= 7:
        return 75

    return 45


def free_intervals(busy, start, end):
    """
    Fusionne les cours qui se chevauchent
    et calcule les périodes libres.
    """

    merged = []

    for a, b in sorted(busy):
        a = max(a, start)
        b = min(b, end)

        if b <= a:
            continue

        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    free = []
    cursor = start

    for a, b in merged:
        if a > cursor:
            free.append((cursor, a))
        cursor = max(cursor, b)

    if cursor < end:
        free.append((cursor, end))

    return free


def generate_smart_plan(
    target_date,
    courses,
    tasks,
    exams,
    start_hour=8,
    end_hour=20,
    break_minutes=10,
    earliest_time=None
):
    """
    Génère un planning d'une journée.

    Les cours sont des contraintes obligatoires.
    Les activités sont planifiées par priorité.
    """

    if isinstance(target_date, datetime):
        target_date = target_date.date()

    if not isinstance(target_date, date):
        raise ValueError("Date invalide.")

    if not 0 <= start_hour < end_hour <= 24:
        raise ValueError("Plage horaire invalide.")

    start = start_hour * 60
    end = end_hour * 60

    if earliest_time is not None:
        current = (
            earliest_time.hour * 60
            + earliest_time.minute
        )
        start = max(start, current)

    if start >= end:
        return []

    plan = []
    busy = []

    # =====================================
    # 1. COURS DE LA JOURNÉE
    # =====================================

    for course in courses:

        day = int(
            course.get(
                "day",
                course.get("day_of_week", -1)
            )
        )

        if day != target_date.weekday():
            continue

        course_start = to_minutes(
            course.get("start", course.get("start_time"))
        )
        course_end = to_minutes(
            course.get("end", course.get("end_time"))
        )

        if course_end <= course_start:
            continue

        busy.append((course_start, course_end))

        visible_start = max(course_start, start)
        visible_end = min(course_end, end)

        if visible_start < visible_end:
            plan.append({
                "start": visible_start,
                "end": visible_end,
                "title": course["subject"],
                "type": "Cours",
                "priority": None
            })

    # =====================================
    # 2. ACTIVITÉS ACADÉMIQUES
    # =====================================

    activities = []

    for index, task in enumerate(tasks):

        deadline = task.get(
            "date",
            task.get("deadline")
        )

        if isinstance(deadline, str):
            deadline = date.fromisoformat(deadline)

        if deadline is None:
            continue

        if int(task.get("completed", 0)) == 1:
            continue

        duration = int(
            task.get(
                "minutes",
                task.get("duration_minutes", 30)
            )
        )

        if duration <= 0:
            continue

        days = (deadline - target_date).days

        activities.append({
            "id": f"task-{index}",
            "title": task["title"],
            "type": "Tâche",
            "remaining": duration,
            "score": priority_score(days, "Tâche"),
            "days": days
        })

    for index, exam in enumerate(exams):

        exam_date = exam.get(
            "date",
            exam.get("exam_date")
        )

        if isinstance(exam_date, str):
            exam_date = date.fromisoformat(exam_date)

        if exam_date is None:
            continue

        days = (exam_date - target_date).days

        if days < 0:
            continue

        activities.append({
            "id": f"exam-{index}",
            "title": f"Réviser {exam['subject']}",
            "type": "Révision",
            "remaining": 45,
            "score": priority_score(days, "Examen"),
            "days": days
        })

    activities.sort(
        key=lambda x: (-x["score"], x["days"])
    )

    # =====================================
    # 3. CONSTRUCTION DES SÉANCES
    # =====================================

    free_slots = free_intervals(
        busy, start, end
    )

    for slot_start, slot_end in free_slots:

        cursor = slot_start

        while cursor < slot_end:

            available = slot_end - cursor

            if available < 15:
                break

            eligible = [
                activity
                for activity in activities
                if activity["remaining"] > 0
            ]

            if not eligible:
                break

            activity = eligible[0]

            duration = min(
                45,
                available,
                activity["remaining"]
            )

            if duration < 15:
                break

            plan.append({
                "start": cursor,
                "end": cursor + duration,
                "title": activity["title"],
                "type": activity["type"],
                "priority": activity["score"]
            })

            activity["remaining"] -= duration
            cursor += duration

            # Pause entre deux séances
            if (
                cursor + break_minutes + 15
                <= slot_end
                and any(
                    a["remaining"] > 0
                    for a in activities
                )
            ):
                plan.append({
                    "start": cursor,
                    "end": cursor + break_minutes,
                    "title": "Pause",
                    "type": "Pause",
                    "priority": None
                })

                cursor += break_minutes

    plan.sort(
        key=lambda x: (x["start"], x["end"])
    )

    # Convertir les horaires pour l'interface
    for item in plan:
        item["start_time"] = to_hhmm(item["start"])
        item["end_time"] = to_hhmm(item["end"])
        item["duration"] = (
            item["end"] - item["start"]
        )

    return plan
