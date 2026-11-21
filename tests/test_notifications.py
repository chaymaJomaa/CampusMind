from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from core.notification_manager import (
    decide_and_save, release_pending, history
)


def run_tests():
    with TemporaryDirectory() as folder:
        db = Path(folder) / 'notifications.db'
        rec = {'type': 'Tâche', 'title': 'mini projet', 'days': 2}
        blocked = datetime(2026, 11, 18, 9, 0)
        free = datetime(2026, 11, 18, 10, 30)
        assert decide_and_save(rec, available=False, enabled=True, at=blocked, db_path=db) == 'deferred'
        assert decide_and_save(rec, available=False, enabled=True, at=blocked, db_path=db) == 'deferred'
        assert len(history(db)) == 1
        assert release_pending(available=False, enabled=True, at=free, db_path=db) is None
        assert release_pending(available=True, enabled=True, at=free, db_path=db) == 'mini projet'
        assert release_pending(available=True, enabled=True, at=free, db_path=db) is None
        assert decide_and_save(rec, available=True, enabled=True, at=free, db_path=db) == 'duplicate'
        assert decide_and_save(rec, available=True, enabled=False, at=free, db_path=db) == 'disabled'
        assert len(history(db)) == 1
        tomorrow = datetime(2026, 11, 19, 10, 0)
        assert decide_and_save(rec, available=True, enabled=True, at=tomorrow, db_path=db) == 'displayed'
        assert len(history(db)) == 2
    print('Tests notifications CampusMind : OK')


if __name__ == '__main__':
    run_tests()
