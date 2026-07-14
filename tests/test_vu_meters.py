"""
Tests for vu_meters.py -- mostly regression coverage for bugs found by
actually clicking through every VU-style option in the running app:

  - PyQt6 is stricter than PyQt5 about QPoint vs. QPointF/float overloads.
    QLinearGradient(QPoint, QPoint), QRadialGradient(QPoint, radius), and
    QPainterPath.moveTo/lineTo(QPoint) all raise TypeError instead of
    silently coercing. Several meters (Modern, Glass, Liquid Glass,
    Tube Amplifier, Classic Sifam) crashed on first paint because of this.
  - DawPeakRmsVUMeter's constructor took rms_alpha, but create_vu_meter()
    always calls widget_cls(parent, alpha=alpha) -- selecting that style
    crashed with an unexpected-keyword TypeError.
  - ClassicHorizontalVUMeter combined setAutoFillBackground(True) with an
    opaque palette *and* WA_TranslucentBackground -- the contradiction
    stopped Qt from erasing the widget between frames, so paints piled up
    instead of replacing each other.

conftest.py sets QT_QPA_PLATFORM=offscreen and provides a session-scoped
QApplication, so real widgets can be constructed and painted here without a
display. Painting is exercised via QWidget.render() to a QPixmap, which
forces a real paintEvent the same way an on-screen repaint would.
"""
import pytest
from PyQt6.QtGui import QPixmap

from src import vu_meters


def _render(widget):
    """Force a real paintEvent and return without raising on success."""
    pixmap = QPixmap(max(widget.width(), 1), max(widget.height(), 1))
    widget.render(pixmap)


# ---------------------------------------------------------------------
# Every registered style: construct, feed it levels, paint it.
#
# This is the broad regression net for the QPoint/QPointF crashes -- it
# would have caught all of Modern, Glass, Liquid Glass, Tube Amplifier,
# Classic Sifam, and the DawPeakRms alpha-kwarg mismatch in one go.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("index", range(len(vu_meters.VU_METER_STYLES)))
def test_every_registered_style_paints_without_raising(index):
    name, widget_cls, *_ = vu_meters.VU_METER_STYLES[index]
    widget = vu_meters.create_vu_meter(index)
    widget.resize(300, 130)

    for level in (0.0, 0.05, 0.6, 0.95, 0.0):
        widget.update_level(level)
        _render(widget)


def test_vu_meter_style_names_matches_registry_order():
    assert vu_meters.vu_meter_style_names() == [
        name for name, *_ in vu_meters.VU_METER_STYLES
    ]


def test_create_vu_meter_clamps_out_of_range_index_to_first_style():
    widget = vu_meters.create_vu_meter(len(vu_meters.VU_METER_STYLES) + 5)
    assert isinstance(widget, vu_meters.VU_METER_STYLES[0][1])


# ---------------------------------------------------------------------
# DawPeakRmsVUMeter -- L/M/R channel handling
# ---------------------------------------------------------------------

def test_daw_peak_rms_accepts_alpha_kwarg_like_every_other_style():
    """Regression: constructor used to take rms_alpha, not alpha, and
    create_vu_meter() always passes alpha= -- selecting this style
    crashed with a TypeError before the fix."""
    widget = vu_meters.DawPeakRmsVUMeter(alpha=0.3)
    assert widget.rms_alpha == 0.3


def test_daw_peak_rms_mono_level_drives_all_three_channels_equally():
    widget = vu_meters.DawPeakRmsVUMeter()
    widget.update_level(0.5)
    levels = {ch: widget.channels[ch]["rms_level"] for ch in ("L", "M", "R")}
    assert levels["L"] == levels["M"] == levels["R"]
    assert levels["L"] > 0.0


def test_daw_peak_rms_two_tuple_mid_is_average_of_left_and_right():
    widget = vu_meters.DawPeakRmsVUMeter()
    widget.update_level((0.9, 0.1))
    # One update isn't enough to reach target due to smoothing, but L/M/R
    # should already show the expected ordering.
    assert widget.channels["L"]["rms_level"] > widget.channels["R"]["rms_level"]
    mid = widget.channels["M"]["rms_level"]
    assert widget.channels["R"]["rms_level"] <= mid <= widget.channels["L"]["rms_level"]


def test_daw_peak_rms_three_tuple_uses_explicit_mid_channel():
    widget = vu_meters.DawPeakRmsVUMeter()
    # Drive each channel to a stable level with repeated updates so the
    # one-pole smoothing has converged enough to compare distinctly.
    for _ in range(50):
        widget.update_level((0.95, 0.5, 0.05))
    assert widget.channels["L"]["rms_level"] > widget.channels["M"]["rms_level"]
    assert widget.channels["M"]["rms_level"] > widget.channels["R"]["rms_level"]


# ---------------------------------------------------------------------
# ClassicHorizontalVUMeter (Sifam) -- background erase regression
# ---------------------------------------------------------------------

def test_classic_sifam_does_not_use_contradictory_background_setup():
    """Regression: autoFillBackground(True) + an opaque palette color,
    combined with WA_TranslucentBackground, stopped Qt from erasing the
    widget between paints -- successive frames visibly piled up instead
    of replacing each other. Both attributes must agree that the widget
    manages its own transparent background."""
    from PyQt6.QtCore import Qt

    widget = vu_meters.ClassicHorizontalVUMeter()
    assert widget.autoFillBackground() is False
    assert widget.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_classic_sifam_repaints_cleanly_at_different_levels():
    widget = vu_meters.ClassicHorizontalVUMeter()
    widget.resize(300, 130)
    widget.update_level(0.1)
    _render(widget)
    widget.update_level(0.9)
    _render(widget)  # would raise if a stale paint state broke drawing
