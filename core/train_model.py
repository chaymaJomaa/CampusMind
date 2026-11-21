
from pathlib import Path

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "campusmind_rf.joblib"
)


def generate_dataset(n_samples=3000):
    """
    Génère des exemples simulés d'activités.

    Variables :
    0 : nombre de jours avant l'échéance
    1 : durée estimée de l'activité
    2 : temps libre de l'étudiant
    3 : type d'activité (0=tâche, 1=examen)

    Classes :
    0 : faible priorité
    1 : priorité moyenne
    2 : priorité élevée
    """

    rng = np.random.default_rng(42)

    X = []
    y = []

    for _ in range(n_samples):

        days = int(rng.integers(0, 31))
        duration = int(rng.integers(15, 121))
        free_time = int(rng.integers(15, 181))
        activity_type = int(rng.integers(0, 2))

        # Étiquetage pédagogique à partir de règles
        if duration > free_time:
            priority = 0

        elif days <= 1:
            priority = 2

        elif days <= 7:
            priority = 1

        else:
            priority = 0

        X.append([
            days,
            duration,
            free_time,
            activity_type
        ])

        y.append(priority)

    return np.array(X), np.array(y)


def train_model():

    print("=== Entraînement CampusMind AI ===")

    X, y = generate_dataset()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nAccuracy : {accuracy:.4f}")

    print("\nRapport de classification :")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            zero_division=0
        )
    )

    print("\nMatrice de confusion :")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2]
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("\nModèle enregistré :")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_model()
