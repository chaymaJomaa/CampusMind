
"""
CampusMind - Tests de robustesse.

Vérifie :
- les notifications différées
- le mode silencieux
- les doublons
- la libération des rappels
- les contraintes temporelles de l'IA
- les prédictions Random Forest
"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from core.notification_manager import (
    decide_and_save,
    release_pending,
    pending_items,
    history
)

from core.ml_engine import predict_priority


class TestCampusMind(unittest.TestCase):

    def setUp(self):
        """
        Crée une nouvelle base temporaire
        pour chaque test.
        """

        self.temp_dir = TemporaryDirectory()

        self.db_path = (
            Path(self.temp_dir.name)
            / "notifications.db"
        )

        self.now = datetime(
            2026, 11, 18, 14, 0
        )

        self.recommendation = {
            "type": "Tâche",
            "title": "Mini projet IA",
            "days": 2
        }

    def tearDown(self):
        """
        Supprime la base temporaire
        après le test.
        """

        self.temp_dir.cleanup()

    # =========================================
    # TEST 1 : ÉTUDIANT OCCUPÉ
    # =========================================

    def test_busy_student(self):

        result = decide_and_save(
            self.recommendation,
            available=False,
            enabled=True,
            at=self.now,
            db_path=self.db_path
        )

        self.assertEqual(
            result,
            "deferred"
        )

    # =========================================
    # TEST 2 : ÉTUDIANT DISPONIBLE
    # =========================================

    def test_available_student(self):

        result = decide_and_save(
            self.recommendation,
            available=True,
            enabled=True,
            at=self.now,
            db_path=self.db_path
        )

        self.assertEqual(
            result,
            "displayed"
        )

    # =========================================
    # TEST 3 : MODE SILENCIEUX
    # =========================================

    def test_silent_mode(self):

        result = decide_and_save(
            self.recommendation,
            available=True,
            enabled=False,
            at=self.now,
            db_path=self.db_path
        )

        self.assertEqual(
            result,
            "disabled"
        )

        self.assertEqual(
            len(history(self.db_path)),
            0
        )

    # =========================================
    # TEST 4 : ANTI-DUPLICATION
    # =========================================

    def test_duplicate_notification(self):

        first = decide_and_save(
            self.recommendation,
            available=True,
            enabled=True,
            at=self.now,
            db_path=self.db_path
        )

        second = decide_and_save(
            self.recommendation,
            available=True,
            enabled=True,
            at=self.now,
            db_path=self.db_path
        )

        self.assertEqual(first, "displayed")
        self.assertEqual(second, "duplicate")

        self.assertEqual(
            len(history(self.db_path)),
            1
        )

    # =========================================
    # TEST 5 : NOTIFICATION DIFFÉRÉE
    # =========================================

    def test_deferred_notification(self):

        result = decide_and_save(
            self.recommendation,
            available=False,
            enabled=True,
            at=self.now,
            db_path=self.db_path
        )

        pending = pending_items(self.db_path)

        self.assertEqual(result, "deferred")
        self.assertEqual(len(pending), 1)

    # =========================================
    # TEST 6 : LIBÉRATION DU RAPPEL
    # =========================================

    def test_release_pending(self):

        decide_and_save(
            self.recommendation,
            available=False,
            enabled=True,
            at=self.now,
            db_path=self.db_path
        )

        later = self.now + timedelta(hours=1)

        result = release_pending(
            available=True,
            enabled=True,
            at=later,
            db_path=self.db_path
        )

        self.assertEqual(
            result,
            "Mini projet IA"
        )

        self.assertEqual(
            len(pending_items(self.db_path)),
            0
        )

    # =========================================
    # TEST 7 : AUCUNE RECOMMANDATION
    # =========================================

    def test_no_recommendation(self):

        result = decide_and_save(
            None,
            available=True,
            enabled=True,
            at=self.now,
            db_path=self.db_path
        )

        self.assertEqual(result, "none")

    # =========================================
    # TEST 8 : DURÉE INCOMPATIBLE
    # =========================================

    def test_insufficient_time_ml(self):

        result = predict_priority(
            days_remaining=2,
            duration=90,
            free_time=30,
            activity_type=0
        )

        self.assertEqual(
            result["priority"],
            "Non compatible"
        )

    # =========================================
    # TEST 9 : TYPE D'ACTIVITÉ INVALIDE
    # =========================================

    def test_invalid_activity_type(self):

        with self.assertRaises(ValueError):

            predict_priority(
                days_remaining=2,
                duration=30,
                free_time=60,
                activity_type=5
            )

    # =========================================
    # TEST 10 : RANDOM FOREST
    # =========================================

    def test_ml_prediction(self):

        result = predict_priority(
            days_remaining=2,
            duration=40,
            free_time=120,
            activity_type=0
        )

        self.assertIn(
            result["priority"],
            ["Faible", "Moyenne", "Élevée"]
        )

        self.assertGreaterEqual(
            result["confidence"],
            0
        )

        self.assertLessEqual(
            result["confidence"],
            100
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
