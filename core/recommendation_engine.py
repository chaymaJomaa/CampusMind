
from datetime import datetime, date


def calculate_urgency(days_remaining):
    """
    Calcule l'urgence d'une activité.
    Score compris entre 0 et 100.
    """
    if days_remaining < 0:
        return 0
    elif days_remaining == 0:
        return 100
    elif days_remaining == 1:
        return 95
    elif days_remaining <= 3:
        return 80
    elif days_remaining <= 7:
        return 60
    elif days_remaining <= 14:
        return 40
    else:
        return 20


def get_recommendations(
    context,
    tasks,
    exams,
    current_time=None
):
    """
    Génère et classe des recommandations
    en fonction du contexte académique.

    Retourne une liste de recommandations
    classées par score décroissant.
    """

    now = current_time or datetime.now()
    today = now.date()
    free_time = context["free_minutes"]

    recommendations = []

    # 1. Vérifier la disponibilité
    if not context["available"]:
        return []

    # 2. Analyser les tâches
    for task in tasks:

        if task["completed"] == 1:
            continue

        deadline = date.fromisoformat(
            task["deadline"]
        )

        days_remaining = (deadline - today).days

        # Ne pas proposer les tâches échues
        if days_remaining < 0:
            continue

        duration = task["duration_minutes"]

        # La tâche doit tenir dans le créneau
        if duration > free_time:
            continue

        urgency = calculate_urgency(days_remaining)

        # Bonus pour une tâche adaptée au créneau
        time_fit = duration / free_time * 100

        score = 0.8 * urgency + 0.2 * time_fit

        recommendations.append({
            "type": "Tâche",
            "title": task["title"],
            "subject": task["subject"],
            "duration": duration,
            "days_remaining": days_remaining,
            "score": round(score, 2),
            "reason": (
                f"Échéance dans {days_remaining} jour(s), "
                f"durée compatible avec votre temps libre."
            )
        })

    # 3. Analyser les examens
    for exam in exams:

        exam_date = date.fromisoformat(
            exam["exam_date"]
        )

        days_remaining = (exam_date - today).days

        if days_remaining < 0:
            continue

        # Séance de révision entre 15 et 45 min
        duration = min(45, free_time)

        if duration < 15:
            continue

        urgency = calculate_urgency(days_remaining)
        time_fit = duration / free_time * 100

        score = 0.8 * urgency + 0.2 * time_fit

        recommendations.append({
            "type": "Examen",
            "title": f"Réviser {exam['subject']}",
            "subject": exam["subject"],
            "duration": duration,
            "days_remaining": days_remaining,
            "score": round(score, 2),
            "reason": (
                f"Examen dans {days_remaining} jour(s). "
                f"Une révision de {duration} minutes "
                f"est possible."
            )
        })

    # 4. Classer les recommandations
    recommendations.sort(
        key=lambda item: (
            -item["score"],
            item["days_remaining"],
            item["title"]
        )
    )

    return recommendations
