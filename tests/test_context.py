
from datetime import datetime
from core.context_manager import analyze_context


courses = [
    {
        "subject": "Intelligence artificielle",
        "day_of_week": 0,
        "start_time": "14:00",
        "end_time": "16:00"
    }
]

scenarios = [
    ("Avant le cours", "2026-11-16 13:30"),
    ("Pendant le cours", "2026-11-16 14:30"),
    ("Après le cours", "2026-11-16 16:15")
]

for name, simulated_time in scenarios:

    now = datetime.strptime(
        simulated_time,
        "%Y-%m-%d %H:%M"
    )

    result = analyze_context(
        courses,
        current_time=now
    )

    print(f"\n--- {name} ---")
    print("Activité :", result["activity"])
    print("Cours actuel :", result["current_course"])
    print("Temps libre :", result["free_minutes"])
    print("Disponible :", result["available"])
