
"""
CampusMind - Évaluation comparative.

Compare :
1. Le moteur à règles
2. Le modèle Random Forest

Mesures :
- Taux d'accord
- Nombre de désaccords
- Temps moyen de prédiction
- Respect des contraintes temporelles

Les données de test sont synthétiques.
"""

import csv
import json
from pathlib import Path
from time import perf_counter

from core.ml_engine import (
    load_model,
    predict_priority
)


# ============================================
# 1. CONFIGURATION
# ============================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "comparison.csv"
JSON_PATH = RESULTS_DIR / "metrics.json"

DAYS = [0, 1, 2, 3, 7, 8, 14, 30]
DURATIONS = [15, 30, 40, 60, 90]
FREE_TIMES = [15, 30, 45, 60, 120]
ACTIVITY_TYPES = [0, 1]


# ============================================
# 2. MOTEUR À RÈGLES
# ============================================

def rule_priority(days_remaining):
    """
    Reproduit les seuils de priorité
    du tableau de bord CampusMind.

    Retourne :
    Faible, Moyenne ou Élevée.
    """

    if days_remaining <= 1:
        return "Élevée"

    elif days_remaining <= 7:
        return "Moyenne"

    return "Faible"


# ============================================
# 3. GÉNÉRATION DES SCÉNARIOS
# ============================================

def generate_scenarios():
    """
    Génère des combinaisons déterministes.
    Aucun scénario n'est tiré au hasard.
    """

    scenarios = []

    for days in DAYS:
        for duration in DURATIONS:
            for free_time in FREE_TIMES:
                for activity_type in ACTIVITY_TYPES:

                    scenarios.append({
                        "days": days,
                        "duration": duration,
                        "free_time": free_time,
                        "activity_type": activity_type,
                        "compatible": duration <= free_time
                    })

    return scenarios


# ============================================
# 4. ÉVALUATION COMPARATIVE
# ============================================

def evaluate():

    print("\n=== CAMPUSMIND : ÉVALUATION IA ===")

    # Chargement initial exclu du chronométrage
    load_model()

    scenarios = generate_scenarios()

    results = []

    excluded = 0
    agreements = 0
    disagreements = 0

    rule_times = []
    ml_times = []

    for scenario in scenarios:

        days = scenario["days"]
        duration = scenario["duration"]
        free_time = scenario["free_time"]
        activity_type = scenario["activity_type"]

        # Ne comparer que les activités réalisables
        if not scenario["compatible"]:
            excluded += 1
            continue

        # Chronométrage du moteur à règles
        start = perf_counter()

        rule_result = rule_priority(days)

        rule_elapsed = (
            perf_counter() - start
        ) * 1000

        # Chronométrage de Random Forest
        start = perf_counter()

        ml_result = predict_priority(
            days_remaining=days,
            duration=duration,
            free_time=free_time,
            activity_type=activity_type
        )

        ml_elapsed = (
            perf_counter() - start
        ) * 1000

        ml_priority = ml_result["priority"]

        agreement = rule_result == ml_priority

        if agreement:
            agreements += 1
        else:
            disagreements += 1

        rule_times.append(rule_elapsed)
        ml_times.append(ml_elapsed)

        results.append({
            "days_remaining": days,
            "duration": duration,
            "free_time": free_time,
            "activity_type": (
                "Examen" if activity_type == 1
                else "Tâche"
            ),
            "rule_priority": rule_result,
            "ml_priority": ml_priority,
            "ml_confidence": ml_result["confidence"],
            "agreement": agreement,
            "rule_time_ms": round(rule_elapsed, 6),
            "ml_time_ms": round(ml_elapsed, 6)
        })

    # ========================================
    # 5. CALCUL DES MÉTRIQUES
    # ========================================

    total = len(results)

    agreement_rate = (
        agreements / total * 100
        if total > 0 else 0
    )

    avg_rule_time = (
        sum(rule_times) / len(rule_times)
        if rule_times else 0
    )

    avg_ml_time = (
        sum(ml_times) / len(ml_times)
        if ml_times else 0
    )

    metrics = {
        "total_scenarios": len(scenarios),
        "evaluated_scenarios": total,
        "excluded_scenarios": excluded,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": round(agreement_rate, 2),
        "avg_rule_time_ms": round(avg_rule_time, 6),
        "avg_ml_time_ms": round(avg_ml_time, 6)
    }

    # ========================================
    # 6. ENREGISTRER LES RÉSULTATS
    # ========================================

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "days_remaining",
                "duration",
                "free_time",
                "activity_type",
                "rule_priority",
                "ml_priority",
                "ml_confidence",
                "agreement",
                "rule_time_ms",
                "ml_time_ms"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================
    # 7. AFFICHAGE DES RÉSULTATS
    # ========================================

    print("\nScénarios générés :",
          metrics["total_scenarios"])

    print("Scénarios évalués :",
          metrics["evaluated_scenarios"])

    print("Scénarios incompatibles exclus :",
          excluded)

    print("Accords :", agreements)
    print("Désaccords :", disagreements)

    print(
        f"Taux d'accord : {agreement_rate:.2f} %"
    )

    print(
        f"Temps moyen règles : "
        f"{avg_rule_time:.6f} ms"
    )

    print(
        f"Temps moyen Random Forest : "
        f"{avg_ml_time:.6f} ms"
    )

    print("\nRésultats enregistrés :")
    print(CSV_PATH)
    print(JSON_PATH)

    print("\n=== ÉVALUATION TERMINÉE ===")

    return metrics


if __name__ == "__main__":
    evaluate()
