
from datetime import datetime

from core.recommendation_engine import (
    get_recommendations
)


# Date simulée
now = datetime(2026, 11, 18, 16, 0)

# Contexte de l'étudiant
context = {
    "activity": "Libre",
    "free_minutes": 45,
    "available": True
}

# Tâches académiques
tasks = [
    {
        "title": "Préparer TP Réseaux",
        "subject": "Réseaux",
        "deadline": "2026-11-23",
        "duration_minutes": 40,
        "completed": 0
    },
    {
        "title": "Rédiger rapport IoT",
        "subject": "IoT",
        "deadline": "2026-11-28",
        "duration_minutes": 90,
        "completed": 0
    }
]

# Examens
exams = [
    {
        "subject": "Intelligence Artificielle",
        "exam_date": "2026-11-19"
    }
]

recommendations = get_recommendations(
    context,
    tasks,
    exams,
    current_time=now
)

print("\n=== RECOMMANDATIONS CAMPUSMIND ===")

for index, rec in enumerate(recommendations, 1):
    print(f"\n{index}. {rec['title']}")
    print("Type :", rec["type"])
    print("Durée :", rec["duration"], "min")
    print("Score :", rec["score"])
    print("Raison :", rec["reason"])

assert len(recommendations) == 2
assert recommendations[0]["title"] == (
    "Réviser Intelligence Artificielle"
)
assert recommendations[0]["score"] == 96.0

# Vérifier le comportement non intrusif
context["available"] = False

blocked = get_recommendations(
    context, tasks, exams, current_time=now
)

assert blocked == []

print("\nTous les tests ont réussi !")
