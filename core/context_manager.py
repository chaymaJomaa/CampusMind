
from datetime import datetime, timedelta


def analyze_context(
    courses,
    current_time=None,
    manual_status="Automatique"
):
    """
    Analyse le contexte temporel de l'étudiant.

    Retourne :
    - l'activité actuelle
    - le cours actuel
    - le prochain cours
    - le temps libre disponible
    - la disponibilité
    """

    now = current_time or datetime.now()

    current_day = now.weekday()
    current_course = None
    next_course = None

    today_courses = []

    # 1. Récupérer les cours du jour
    for course in courses:

        if course["day_of_week"] != current_day:
            continue

        start = datetime.combine(
            now.date(),
            datetime.strptime(
                course["start_time"], "%H:%M"
            ).time()
        )

        end = datetime.combine(
            now.date(),
            datetime.strptime(
                course["end_time"], "%H:%M"
            ).time()
        )

        today_courses.append({
            "subject": course["subject"],
            "start": start,
            "end": end
        })

    # 2. Trier les cours chronologiquement
    today_courses.sort(
        key=lambda course: course["start"]
    )

    # 3. Détecter le cours actuel
    for course in today_courses:

        if course["start"] <= now < course["end"]:
            current_course = course
            break

    # 4. Identifier le prochain cours
    for course in today_courses:

        if course["start"] > now:
            next_course = course
            break

    # 5. Calculer le temps libre
    end_of_day = datetime.combine(
        now.date() + timedelta(days=1),
        datetime.min.time()
    )

    if current_course:
        activity = "En cours"
        free_minutes = 0

    else:
        activity = "Libre"

        if next_course:
            end_of_slot = next_course["start"]
        else:
            end_of_slot = end_of_day

        free_minutes = max(
            0,
            int((end_of_slot - now).total_seconds() // 60)
        )

    # 6. Appliquer une éventuelle déclaration manuelle
    if manual_status == "En cours":
        activity = "En cours"
        free_minutes = 0

    elif manual_status == "En repos":
        activity = "En repos"
        free_minutes = 0

    elif manual_status == "Libre":
        activity = "Libre"

        # Une déclaration manuelle n'efface pas
        # les cours déjà présents dans l'agenda.
        if current_course:
            free_minutes = 0

    # 7. Déterminer la disponibilité
    available = (
        activity == "Libre"
        and current_course is None
        and free_minutes >= 15
    )

    return {
        "timestamp": now.strftime("%d/%m/%Y %H:%M"),
        "activity": activity,
        "current_course": (
            current_course["subject"]
            if current_course else None
        ),
        "next_course": (
            next_course["subject"]
            if next_course else None
        ),
        "free_minutes": free_minutes,
        "available": available
    }
