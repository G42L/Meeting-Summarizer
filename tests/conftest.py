"""
Shared pytest setup. Some modules under test (pipeline.py) use PyQt5
QObject/pyqtSignal for their public API even though they contain no actual
widgets -- a QApplication instance makes signal/slot behavior consistent
without needing a real display.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
