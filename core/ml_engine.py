
from pathlib import Path

import joblib


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "campusmind_rf.joblib"
)

MODEL = None

PRIORITY_LABELS = {
    0: "Faible",
    1: "Moyenne",
    2: "Élevée"
}


def load_model():
    """
    Charge le modèle entraîné.
    Uniquement depuis notre fichier local de confiance.
    """

    global MODEL

    if MODEL is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Modèle introuvable. "
                "Exécutez python -m core.train_model"
            )

        MODEL = joblib.load(MODEL_PATH)

    return MODEL


def predict_priority(
    days_remaining,
    duration,
    free_time,
    activity_type
):
    """
    Prédit la priorité académique d'une activité.

    activity_type :
        0 = tâche
        1 = examen
    """

    if activity_type not in (0, 1):
        raise ValueError("Type d'activité invalide.")

    if duration <= 0 or free_time < 0:
        raise ValueError("Durée ou temps libre invalide.")

    if days_remaining < 0:
        return {
            "priority": "Échue",
            "class_id": None,
            "confidence": None
        }

    if duration > free_time:
        return {
            "priority": "Non compatible",
            "class_id": None,
            "confidence": None
        }

    model = load_model()

    features = [[
        days_remaining,
        duration,
        free_time,
        activity_type
    ]]

    prediction = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]

    confidence = float(
        max(probabilities) * 100
    )

    return {
        "priority": PRIORITY_LABELS[prediction],
        "class_id": prediction,
        "confidence": round(confidence, 2)
    }
