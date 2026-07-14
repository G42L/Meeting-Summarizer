#!/usr/bin/env python3
"""
Meeting Transcriber - PyQt6 GUI
Records audio from one or more sources at once (mic + Teams/system-audio
loopback + ...), mixes them in real time via AudioMixerEngine, transcribes
the mix with Whisper, and summarizes with a local LLM.
All files are saved under ./transcripts/YYYY-MM-DD HH.MM.SS/
Includes a live mixed waveform + VU meter, and a small VU meter per source.

This file is the GUI/orchestration layer only. The actual logic lives in:
    audio_engine.py    multi-source capture + mixing
    vu_meters.py        waveform/VU-meter widgets
    whisper_engine.py   model catalogue, download, transcription
    llm_backend.py       LLM backend detection + summarization
    pipeline.py          Job / ProcessingWorker / QueueWorker
"""

import sys
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QProgressBar,
    QGroupBox, QFileDialog, QMessageBox, QCheckBox, QSizePolicy, QSlider,
    QPlainTextEdit, QListWidget, QListWidgetItem, QLineEdit, QDialog,
    QDialogButtonBox, QSystemTrayIcon, QMenu, QFrame,
    QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSettings, QPropertyAnimation, QEasingCurve, QPoint, QSize, QUrl, QEvent
from PyQt6.QtGui import QIcon, QFont, QKeySequence, QCursor, QPixmap, QPainter, QColor, QPalette, QTextDocument, QShortcut
from PyQt6.QtSvg import QSvgRenderer

from . import audio_engine
from . import vu_meters
from . import whisper_engine
from . import llm_backend
from . import pipeline
from . import sysmon
from . import resources

ICONS_DIR = os.path.join(os.path.dirname(__file__), "icons", "ui")
# Qt Style Sheet url() needs forward slashes even on Windows.
_ICONS_DIR_CSS = ICONS_DIR.replace(os.sep, "/")


def themed_icon(name, color, size=18):
    """
    Loads icons/ui/{name}.svg and tints it with `color` (a QColor), so one
    set of plain black-stroke SVGs can be recolored to match whatever the
    current OS palette actually is -- QSvgRenderer doesn't resolve CSS
    `currentColor`, so the SVGs are opaque black and the real color comes
    from here instead, via a SourceIn composite (keeps the icon's alpha
    shape, replaces every opaque pixel with `color`).
    """
    renderer = QSvgRenderer(os.path.join(ICONS_DIR, f"{name}.svg"))
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color)
    painter.end()
    return QIcon(pixmap)


# Single palette-driven stylesheet for the whole window -- every color is a
# palette(...) reference (the widget's actual current QPalette role), never
# a hardcoded hex, so this automatically matches the OS's light/dark theme
# and accent color instead of committing to one explicit look. `cls` is a
# plain Qt dynamic property (set via _set_button_class), not a real CSS
# class -- QSS supports selecting on arbitrary widget properties this way.
# The one deliberate exception is [cls="danger"]: recording-in-progress is
# conventionally red regardless of theme (like a hardware REC light), so
# that one case only uses a fixed color instead of palette(highlight).
BASE_STYLESHEET = """
QPushButton {
    background-color: palette(button);
    color: palette(buttontext);
    border: 1px solid palette(mid);
    border-radius: 8px;
    padding: 6px 14px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: palette(light);
}
QPushButton:pressed {
    background-color: palette(dark);
}
QPushButton:disabled {
    color: palette(mid);
    border-color: palette(midlight);
}
QPushButton[cls="primary"] {
    background-color: palette(highlight);
    color: palette(highlightedtext);
    border: 1px solid palette(highlight);
    font-weight: 600;
}
QPushButton[cls="primary"]:hover {
    background-color: palette(highlight);
}
QPushButton[cls="danger"] {
    background-color: #e74c3c;
    color: white;
    border: 1px solid #c0392b;
    font-weight: 600;
}
QPushButton[cls="danger"]:hover {
    background-color: #c0392b;
}

QGroupBox {
    border: 1px solid palette(mid);
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}

QComboBox, QLineEdit, QPlainTextEdit {
    border: 1px solid palette(mid);
    border-radius: 6px;
    padding: 4px 6px;
    min-height: 20px;
}
QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus {
    border: 1px solid palette(highlight);
}

/* Without this, the drop-down button keeps the native style's square
   corners even though QComboBox itself is rounded above -- the box looks
   rounded but the arrow sits in a squared-off patch cut out of its right
   edge (seen on both macOS and Linux). Rounding only the two corners that
   sit flush with the outer border, and leaving the arrow glyph itself to
   the platform style, keeps the fix minimal and still theme-native. */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid palette(mid);
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}
QComboBox::drop-down:hover {
    background-color: palette(light);
}
QComboBox::down-arrow {
    /* Styling ::drop-down at all makes Qt stop drawing its native arrow
       glyph inside it (verified directly -- it renders as a blank box
       otherwise), so this needs an explicit image. QSS can't tint url()
       images per-palette like themed_icon() does, so the color is baked
       into the SVG itself as a neutral gray already verified >=3:1
       contrast against both light and dark (see LOG_ICON_COLORS). */
    image: url(__ICONS_DIR__/chevron-down.svg);
    width: 10px;
    height: 10px;
}

QListWidget {
    border: 1px solid palette(mid);
    border-radius: 6px;
}
QListWidget::item:selected {
    background-color: palette(highlight);
    color: palette(highlightedtext);
}
""".replace("__ICONS_DIR__", _ICONS_DIR_CSS)

# Maps each status emoji used in append_log() calls (across main.py and the
# log(...) callables in pipeline.py/llm_backend.py/whisper_engine.py/
# diarization.py, all of which funnel into append_log) to one of the
# bundled icons/ui/*.svg names. Emoji glyph rendering isn't guaranteed
# correct across platforms/Qt versions/fonts -- verified in this very
# environment, where Qt renders these as blank tofu boxes despite a color
# emoji font being installed -- so append_log() substitutes each of these
# for the same theme-tinted vector icon already used on buttons, via a
# Markdown image reference resolved against a QTextDocument resource (see
# MainWindow._register_log_icons()).
LOG_ICONS = {
    "✅": "check-circle-outside",
    "❌": "cross-circle",
    "⚠️": "alert-triangle",
    "🛑": "alert-octagon",
    "🎤": "mic",
    "🌊": "audio-wave",  # "Starting transcription..." -- distinct from mic/Recording above
    "⏹": "stop-circle",
    "📁": "folder",
    "📂": "folder",
    "⏳": "clock",
    "📝": "edit-3",
    "📊": "bar-chart-2",
    "🤖": "cpu",
    "🗣️": "users",
    "🔄": "refresh-cw",
    "📥": "inbox",
    "🗑️": "trash-2",
    "➖": "minus",
    "➕": "plus",
}

# Semantic accent colors for the log icons above, echoing the color coding
# the original emoji carried (green check, red x, amber warning, ...)
# instead of the neutral text-color tint buttons use. Each was picked with
# a fixed hex (not palette-driven, unlike themed_icon()'s normal use) and
# verified to hold a WCAG contrast ratio >= 3:1 against both a white and a
# #1e1e1e dark background, so they stay legible in either OS theme without
# needing separate light/dark variants.
LOG_ICON_COLORS = {
    "check-circle-outside": "#1e8e3e",  # green -- success
    "plus": "#1e8e3e",            # green -- added
    "cross-circle": "#d93025",    # red -- error
    "alert-octagon": "#d93025",   # red -- stopped/cancelled
    "alert-triangle": "#b8860b",  # amber -- warning
    "refresh-cw": "#1a73e8",      # blue -- in progress
    "inbox": "#1a73e8",           # blue -- queued
    "mic": "#1a73e8",             # blue -- active recording
    "audio-wave": "#1a73e8",      # blue -- transcribing
    "cpu": "#9350c4",             # purple -- LLM
    "users": "#9350c4",           # purple -- speakers
    "edit-3": "#0e8a7d",          # teal -- review
    "bar-chart-2": "#0e8a7d",     # teal -- info/stats
    "folder": "#b8730a",          # amber -- folder
    "clock": "#757575",          # gray -- waiting
    "stop-circle": "#757575",     # gray -- stopped (neutral, not an error)
    "minus": "#d93025",           # red -- removed
    "trash-2": "#757575",         # gray -- clear/delete (not alarming)
}


# ----------------------------------------------------------------------
# A small, borderless icon-only button that swaps to a different color on
# hover -- QSS alone can't recolor an already-rendered icon pixmap for a
# :hover pseudo-state, so this just pre-renders both and switches between
# them directly. Used where a full bordered QPushButton (BASE_STYLESHEET's
# default) would be too heavy for a minor, low-emphasis action.
# ----------------------------------------------------------------------
class FlatIconButton(QPushButton):
    """
    Borderless, background-less, fixed-size icon-only button base -- QSS
    can't give a plain QPushButton this "just an icon, nothing else" look
    reliably across platforms, so it's set directly here instead. Shared
    base for HoverColorIconButton (e.g. remove-source, recolors on hover)
    and MuteButton (mute/unmute, swaps icon shape on click vs. long-press),
    so both get identical sizing/style and only differ in what actually
    changes (color vs. icon).
    """
    def __init__(self, size=18, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(size, size))
        self.setFixedSize(size + 8, size + 8)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton { border: none; background: transparent; padding: 0px; }"
            "QPushButton:hover { border: none; background: transparent; }"
            "QPushButton:checked { border: none; background: transparent; }"
        )


class HoverColorIconButton(FlatIconButton):
    """A FlatIconButton that swaps to a different color on hover -- QSS
    alone can't recolor an already-rendered icon pixmap for a :hover
    pseudo-state, so this pre-renders both and switches between them
    directly."""
    def __init__(self, icon_name, hover_color, normal_color=None, size=18, parent=None):
        super().__init__(size, parent)
        self._normal_color = normal_color
        self._icon_name = icon_name
        self._hover_icon = themed_icon(icon_name, QColor(hover_color), size)
        self.refresh_normal_icon()

    def refresh_normal_icon(self):
        """Re-renders the non-hovered icon against the current palette's
        text color (or the fixed normal_color if one was given) -- call
        this if the palette changes after construction."""
        color = self._normal_color or self.palette().color(QPalette.ColorRole.WindowText)
        self._normal_icon = themed_icon(self._icon_name, QColor(color) if isinstance(color, str) else color, self.iconSize().width())
        self.setIcon(self._normal_icon)

    def enterEvent(self, event):
        self.setIcon(self._hover_icon)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setIcon(self._normal_icon)
        super().leaveEvent(event)


class MuteButton(FlatIconButton):
    """
    A three-state mute control distinguishing a quick click from a
    press-and-hold, since Qt has no built-in long-press gesture:

      - short click : toggle "none" <-> "short" (full mute -- excluded
        from the mix AND this source's own dedicated VU meter goes silent)
      - long press  : toggle "none" <-> "long" (mix-only mute -- excluded
        from the mix, but the VU meter keeps showing real activity; this
        is the mute behavior the app had before short/long were split out)

    Cross-mode gestures are deliberately asymmetric (confirmed as the
    intended UX, not an oversight): a short click while long-muted just
    unmutes (has to be a deliberate second click to reach short-mute from
    there), but a long-press while short-muted switches directly to
    long-mute in one gesture.
    """
    LONG_PRESS_MS = 500
    ICON_FOR_MODE = {"none": "volume-high", "short": "volume-muted", "long": "volume-disabled"}
    # Fixed (not palette-driven) colors, each verified >=3:1 contrast
    # against both a white and a #1e1e1e dark background, same bar as
    # LOG_ICON_COLORS. Deliberately inverted from the usual red=bad/
    # green=good convention: red on "none" flags a hot mic ("you're being
    # heard, recording-ready"), green on "short" reassures the opposite
    # ("fully muted, safe to talk freely").
    COLOR_FOR_MODE = {"none": "#d93025", "short": "#1e8e3e", "long": "#cc6600"}
    mode_changed = pyqtSignal(str)   # "none" | "short" | "long"

    def __init__(self, size=18, parent=None):
        super().__init__(size, parent)
        self._mode = "none"
        self._long_press_fired = False
        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.setInterval(self.LONG_PRESS_MS)
        self._long_press_timer.timeout.connect(self._trigger_long_press)
        self._update_icon()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._long_press_fired = False
            self._long_press_timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._long_press_timer.stop()
            if not self._long_press_fired and self.rect().contains(event.pos()):
                self._set_mode("short" if self._mode == "none" else "none")
        super().mouseReleaseEvent(event)

    def _trigger_long_press(self):
        self._long_press_fired = True
        self._set_mode("none" if self._mode == "long" else "long")

    def _set_mode(self, mode):
        self._mode = mode
        self._update_icon()
        self.mode_changed.emit(mode)

    def _update_icon(self):
        color = QColor(self.COLOR_FOR_MODE[self._mode])
        self.setIcon(themed_icon(self.ICON_FOR_MODE[self._mode], color, self.iconSize().width()))
        self.setToolTip({
            "none": "Click to mute, hold to mute (keep meter active)",
            "short": "Muted -- click to unmute",
            "long": "Muted (meter still active) -- click to unmute",
        }[self._mode])


# ----------------------------------------------------------------------
# One row in the "Audio Sources" panel: name, gain, mute, mini VU meter,
# remove button. Purely a UI widget -- AudioMixerEngine doesn't know this
# class exists; MainWindow is the only thing that wires the two together.
# ----------------------------------------------------------------------
class SourceRow(QWidget):
    remove_clicked = pyqtSignal(str)   # emits the source name to remove
    gain_changed = pyqtSignal(str, float)     # name, gain (0.0 .. 2.0)
    mute_changed = pyqtSignal(str, bool, bool)      # name, muted, full_mute

    def __init__(self, name, display_label, icon=None, parent=None):
        super().__init__(parent)
        self.source_name = name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # Exposed as attributes (not local vars) so MainWindow can register
        # them with _track_icon() for the OS theme-change refresh.
        self.icon_label = None
        if icon is not None:
            self.icon_label = QLabel()
            self.icon_label.setPixmap(icon.pixmap(16, 16))
            layout.addWidget(self.icon_label)

        label = QLabel(display_label)
        label.setMinimumWidth(220)
        layout.addWidget(label, stretch=2)

        layout.addWidget(QLabel("Gain:"))
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 200)  # percent
        self.gain_slider.setValue(100)
        self.gain_slider.setMaximumWidth(110)
        layout.addWidget(self.gain_slider)
        self.gain_value_label = QLabel("100%")
        self.gain_value_label.setMinimumWidth(38)
        layout.addWidget(self.gain_value_label)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)

        self.mute_btn = MuteButton()
        self.mute_btn.mode_changed.connect(self._on_mute_mode_changed)
        layout.addWidget(self.mute_btn)

        # Small, fixed-size VU meter just for this one source.
        self.vu = vu_meters.MiniLEDHorizontalVUMeter(alpha=0.10)
        self.vu.setMinimumHeight(20)
        self.vu.setMaximumHeight(24)
        self.vu.setMinimumWidth(90)
        self.vu.setMaximumWidth(140)
        layout.addWidget(self.vu)

        self.remove_btn = HoverColorIconButton("cross-circle", hover_color="#d93025")
        self.remove_btn.setToolTip("Remove this source")
        self.remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self.source_name))
        layout.addWidget(self.remove_btn)

    def _on_gain_changed(self, percent):
        self.gain_value_label.setText(f"{percent}%")
        self.gain_changed.emit(self.source_name, percent / 100.0)

    def _on_mute_mode_changed(self, mode):
        self.mute_changed.emit(self.source_name, mode != "none", mode == "short")


# ----------------------------------------------------------------------
# One row in the "System" panel per GPU detected by sysmon.list_gpus():
# name, load bar, VRAM bar. Bars go disabled/"N/A" when a stat isn't
# available for that GPU/platform -- see sysmon.py for which backends
# expose what.
# ----------------------------------------------------------------------
class GpuRow(QWidget):
    def __init__(self, gpu, parent=None):
        super().__init__(parent)
        self.gpu = gpu

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # Two even columns: GPU name on the left, Load+VRAM grouped on the
        # right -- equal stretch factors keep them at half the row each
        # regardless of how long the GPU's name is.
        label = QLabel(f"GPU ({gpu['vendor']}): {gpu['name']}")
        layout.addWidget(label, stretch=1)

        stats_layout = QHBoxLayout()
        stats_layout.addWidget(QLabel("Load:"))
        self.load_bar = QProgressBar()
        self.load_bar.setRange(0, 100)
        self.load_bar.setFormat("%p%")
        stats_layout.addWidget(self.load_bar, stretch=1)

        stats_layout.addWidget(QLabel("VRAM:"))
        self.vram_bar = QProgressBar()
        self.vram_bar.setRange(0, 100)
        self.vram_bar.setFormat("%p%")
        stats_layout.addWidget(self.vram_bar, stretch=1)
        self.vram_value_label = QLabel("N/A")
        self.vram_value_label.setMinimumWidth(90)
        stats_layout.addWidget(self.vram_value_label)

        layout.addLayout(stats_layout, stretch=1)

    def update_stats(self, stats):
        if stats["load_percent"] is None:
            self.load_bar.setEnabled(False)
            self.load_bar.setValue(0)
        else:
            self.load_bar.setEnabled(True)
            self.load_bar.setValue(int(stats["load_percent"]))

        vram_total = stats["vram_total_mb"]
        vram_used = stats["vram_used_mb"]
        if not vram_total or vram_used is None:
            self.vram_bar.setEnabled(False)
            self.vram_bar.setValue(0)
            self.vram_value_label.setText("N/A")
        else:
            self.vram_bar.setEnabled(True)
            self.vram_bar.setValue(int(100 * vram_used / vram_total))
            self.vram_value_label.setText(f"{vram_used / 1024:.1f}/{vram_total / 1024:.1f} GB")


# ----------------------------------------------------------------------
# Modal dialog shown mid-job when Job.review_transcript is True. The
# ProcessingWorker that requested this is paused in a polling loop on the
# queue thread (see pipeline.ProcessingWorker.process) -- Continue/Cancel
# here just set plain attributes on that worker instance to unblock it,
# the same attribute-polling idiom already used for Cancel-the-whole-job.
# ----------------------------------------------------------------------
class TranscriptReviewDialog(QDialog):
    def __init__(self, transcript, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Transcript")
        self.resize(700, 500)
        self.setStyleSheet(BASE_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Edit the transcript below if needed, then continue to summarization:"))

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlainText(transcript)
        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox()
        continue_btn = buttons.addButton("Continue", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn = buttons.addButton("Cancel Job", QDialogButtonBox.ButtonRole.RejectRole)
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        continue_btn.setIcon(themed_icon("check-circle-outside", text_color))
        cancel_btn.setIcon(themed_icon("cross-circle", text_color))
        continue_btn.setProperty("cls", "primary")
        continue_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def text(self):
        return self.text_edit.toPlainText()


# ----------------------------------------------------------------------
# Floating history sidebar, styled after the collapsible conversation-
# history sidebars in some LLM chat UIs: a thin handle stays permanently
# visible at the left edge; hovering it slides the full panel out over the
# window content, and moving the mouse away (after a short grace period,
# so briefly crossing the edge doesn't flicker) collapses it again.
#
# This widget is deliberately never added to a layout -- it's a plain
# child of `central`, positioned by hand and raised above its siblings.
# Qt clips child widgets to the parent's rectangle automatically, so a
# height much taller than any real window just gets cropped for free;
# that means this never has to track window-resize events to stay full
# height, and its floating-above-everything-else behavior costs nothing
# more than not calling layout.addWidget(...) on it.
# ----------------------------------------------------------------------
class HistorySidebar(QWidget):
    PANEL_WIDTH = 280
    HANDLE_WIDTH = 14
    ANIMATION_MS = 180
    COLLAPSE_DELAY_MS = 250
    TALL_ENOUGH_HEIGHT = 3000  # cropped to the real window height by Qt's child clipping

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(self.PANEL_WIDTH, self.TALL_ENOUGH_HEIGHT)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QFrame()
        content.setFrameShape(QFrame.Shape.StyledPanel)
        content.setAutoFillBackground(True)
        content_layout = QVBoxLayout(content)

        # Opens the general ./transcripts/ output root (or the most recent
        # job's folder) -- placed above the "History" label and named
        # distinctly from the list's own "Open Folder" (which opens the
        # folder for whichever *specific* past session is selected below),
        # so the two aren't confused for one another.
        self.open_output_folder_btn = QPushButton("Open Outputs Folder")
        content_layout.addWidget(self.open_output_folder_btn)

        content_layout.addWidget(QLabel("History (past sessions)"))

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        btn_row.addWidget(self.refresh_btn)
        self.open_folder_btn = QPushButton("Open Folder")
        btn_row.addWidget(self.open_folder_btn)
        content_layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        self.list_widget.setToolTip("Double-click a session to open its summary.md")
        content_layout.addWidget(self.list_widget)
        outer.addWidget(content, stretch=1)

        handle = QFrame()
        handle.setFrameShape(QFrame.Shape.StyledPanel)
        handle.setAutoFillBackground(True)
        handle.setFixedWidth(self.HANDLE_WIDTH)
        handle_layout = QVBoxLayout(handle)
        handle_layout.setContentsMargins(0, 8, 0, 0)
        self.handle_label = QLabel()
        self.handle_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        handle_layout.addWidget(self.handle_label)
        handle_layout.addStretch()
        outer.addWidget(handle)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(3, 0)
        self.setGraphicsEffect(shadow)

        self._collapse_timer = QTimer(self)
        self._collapse_timer.setSingleShot(True)
        self._collapse_timer.setInterval(self.COLLAPSE_DELAY_MS)
        self._collapse_timer.timeout.connect(self._collapse)

        self._animation = QPropertyAnimation(self, b"pos", self)
        self._animation.setDuration(self.ANIMATION_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Polling the real cursor position instead of relying on
        # enterEvent/leaveEvent: Qt's enter/leave delivery for a raised,
        # partly-off-screen overlay like this one is unreliable exactly at
        # window-focus/re-entry boundaries (e.g. alt-tabbing back in can
        # synthesize a spurious Enter regardless of where the cursor
        # actually is), which was popping the sidebar open on its own.
        # Checking the true cursor position directly sidesteps that whole
        # class of platform quirk. Same "one continuous QTimer" pattern
        # already used for monitor_timer/sysmon_timer elsewhere.
        self._hovered = False
        self._hover_poll_timer = QTimer(self)
        self._hover_poll_timer.setInterval(100)
        self._hover_poll_timer.timeout.connect(self._check_hover)
        self._hover_poll_timer.start()

        self.reposition(expanded=False)

    def reposition(self, expanded, animate=False):
        target_x = 0 if expanded else -(self.PANEL_WIDTH - self.HANDLE_WIDTH)
        target = QPoint(target_x, 0)
        if animate:
            self._animation.stop()
            self._animation.setStartValue(self.pos())
            self._animation.setEndValue(target)
            self._animation.start()
        else:
            self.move(target)

    def _collapse(self):
        self.reposition(expanded=False, animate=True)

    def _check_hover(self):
        window = self.window()
        local_pos = self.mapFromGlobal(QCursor.pos())
        is_hovered = window.isActiveWindow() and self.rect().contains(local_pos)

        if is_hovered and not self._hovered:
            self._hovered = True
            self._collapse_timer.stop()
            self.raise_()
            self.reposition(expanded=True, animate=True)
        elif not is_hovered and self._hovered:
            self._hovered = False
            self._collapse_timer.start()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Transcriber")
        self.setMinimumSize(960, 995)

        self._icon_cache = {}
        self._icon_refreshers = []   # zero-arg callables re-applying a palette-tinted icon
        self._record_icon_name = "mic"

        # ---- Multi-source recording state ----
        self.mixer = audio_engine.AudioMixerEngine()
        self.source_rows = {}      # name -> SourceRow
        self.is_recording = False
        self._reported_source_errors = {}  # name -> last-logged error, avoids spamming the log every tick

        # ---- Loaded-file playback state (unchanged from before) ----
        self.loaded_samples = None
        self.loaded_sr = None
        self.playhead = 0
        self.playback_timer = None

        icon_path = str(resources.resource_path("src", "icon.svg"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setup_ui()
        self.setup_tray_icon(icon_path)
        self.setup_shortcuts()
        self.refresh_source_picker()
        self.refresh_whisper_models()
        self.refresh_backends()
        self.refresh_history()

        self.queue_worker = pipeline.QueueWorker()
        self.queue_thread = QThread()
        self.queue_worker.moveToThread(self.queue_thread)
        self.queue_thread.started.connect(self.queue_worker.run)
        self.queue_thread.start()

        self.queue_worker.log.connect(self.append_log)
        self.queue_worker.progress.connect(self.progress_bar.setValue)
        self.queue_worker.summary_chunk.connect(self.append_summary)
        self.queue_worker.job_finished.connect(self.on_job_finished)
        self.queue_worker.job_error.connect(self.on_error)
        self.queue_worker.job_started.connect(self.on_job_started)
        self.queue_worker.transcript_ready.connect(self.on_transcript_ready)

        self.whisper_combo.currentIndexChanged.connect(self.on_whisper_model_changed)

        self.job_count = 0
        self.update_config_lock()

        # Drives live VU meters and the combined waveform continuously, for
        # as long as the app runs -- not just while Record is pressed. Runs
        # for every source from the moment it's added (see
        # AudioMixerEngine.add_source), independent of self.is_recording.
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_visualization)
        self.monitor_timer.start(33)

        # CPU/RAM/GPU stats change slowly compared to audio -- 1s is plenty
        # responsive and avoids re-querying NVML/sysfs at 30fps for nothing.
        self.sysmon_timer = QTimer()
        self.sysmon_timer.timeout.connect(self.update_system_monitor)
        self.sysmon_timer.start(1000)
        self.update_system_monitor()  # populate immediately instead of waiting 1s

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        # Reserve the collapsed sidebar handle's width as permanent left
        # margin, so its ever-visible strip sits in its own space instead
        # of overlapping the leftmost few pixels of every section's text.
        margins = layout.contentsMargins()
        layout.setContentsMargins(
            margins.left() + HistorySidebar.HANDLE_WIDTH, margins.top(), margins.right(), margins.bottom()
        )

        central.setStyleSheet(BASE_STYLESHEET)

        # ----- Audio Sources (multi-source picker + list of active sources) -----
        self.dev_group = QGroupBox("Audio Sources")
        dev_layout = QVBoxLayout()

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Add source:"))
        self.source_picker_combo = QComboBox()
        self.source_picker_combo.setToolTip(
            "Microphones, plus system-audio / Teams loopback devices "
            "(BlackHole on macOS, PulseAudio Monitor on Linux, WASAPI loopback on Windows)"
        )
        picker_row.addWidget(self.source_picker_combo, stretch=1)
        add_source_btn = QPushButton(self._icon("plus"), "Add")
        add_source_btn.clicked.connect(self.add_selected_source)
        self._track_icon(lambda b=add_source_btn: b.setIcon(self._icon("plus")))
        picker_row.addWidget(add_source_btn)
        refresh_dev_btn = QPushButton(self._icon("refresh-cw"), "Refresh")
        refresh_dev_btn.clicked.connect(self.refresh_source_picker)
        self._track_icon(lambda b=refresh_dev_btn: b.setIcon(self._icon("refresh-cw")))
        picker_row.addWidget(refresh_dev_btn)
        dev_layout.addLayout(picker_row)

        self.sources_container = QWidget()
        self.sources_layout = QVBoxLayout(self.sources_container)
        self.sources_layout.setContentsMargins(0, 0, 0, 0)
        dev_layout.addWidget(self.sources_container)

        self.no_sources_label = QLabel("No sources added yet — pick one above and click Add.")
        self.no_sources_label.setStyleSheet("color: gray; font-style: italic;")
        self.sources_layout.addWidget(self.no_sources_label)

        self.dev_group.setLayout(dev_layout)
        layout.addWidget(self.dev_group)

        # ----- Whisper model -----
        self.whisper_group = QGroupBox("Whisper Model")
        whisper_group_layout = QVBoxLayout()

        whisper_layout = QHBoxLayout()
        whisper_layout.addWidget(QLabel("Model:"))
        self.whisper_combo = QComboBox()
        self.whisper_combo.setToolTip("Select a Whisper model size")
        whisper_layout.addWidget(self.whisper_combo)
        self.use_cli_check = QCheckBox("Use whisper-cli (requires built binary)")
        self.use_cli_check.setToolTip("If unchecked, uses faster-whisper (Python)")
        self.use_cli_check.toggled.connect(self.on_use_cli_toggled)
        whisper_layout.addWidget(self.use_cli_check)
        refresh_whisper_btn = QPushButton(self._icon("refresh-cw"), "Refresh")
        refresh_whisper_btn.clicked.connect(self.refresh_whisper_models)
        self._track_icon(lambda b=refresh_whisper_btn: b.setIcon(self._icon("refresh-cw")))
        whisper_layout.addWidget(refresh_whisper_btn)
        whisper_group_layout.addLayout(whisper_layout)

        # Review + diarization share one row (rather than a row each) to
        # save vertical space -- both are secondary, occasional-use options
        # for the same "before summarizing" step.
        options_layout = QHBoxLayout()
        self.review_transcript_check = QCheckBox("Review transcript before summarizing")
        self.review_transcript_check.setIcon(self._icon("edit-3"))
        self._track_icon(lambda: self.review_transcript_check.setIcon(self._icon("edit-3")))
        self.review_transcript_check.setToolTip(
            "Pause after transcription so you can read/edit the text before it's sent to the LLM"
        )
        options_layout.addWidget(self.review_transcript_check)

        self.diarization_check = QCheckBox("Label speakers (diarization)")
        self.diarization_check.setIcon(self._icon("users"))
        self._track_icon(lambda: self.diarization_check.setIcon(self._icon("users")))
        self.diarization_check.setToolTip(
            "Requires the faster-whisper backend (not whisper-cli), pyannote.audio installed, "
            "and a Hugging Face token with the gated 'pyannote/speaker-diarization-3.1' model's "
            "terms accepted at huggingface.co"
        )
        self.diarization_check.toggled.connect(self.on_diarization_toggled)
        options_layout.addWidget(self.diarization_check)
        self.hf_token_label = QLabel("HF Token:")
        self.hf_token_label.setEnabled(False)
        options_layout.addWidget(self.hf_token_label)
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setPlaceholderText("hf_...")
        self.hf_token_edit.setEnabled(False)
        options_layout.addWidget(self.hf_token_edit, stretch=1)
        whisper_group_layout.addLayout(options_layout)

        self.whisper_group.setLayout(whisper_group_layout)
        layout.addWidget(self.whisper_group)

        self._load_diarization_settings()

        # ----- LLM Backend and Model (+ summary style, folded in below to save a row) -----
        self.llm_group = QGroupBox("LLM Backend")
        llm_group_layout = QVBoxLayout()

        llm_layout = QHBoxLayout()
        llm_layout.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.setToolTip("Detected running LLM servers")
        self.backend_combo.currentIndexChanged.connect(self.on_backend_changed)
        llm_layout.addWidget(self.backend_combo)
        llm_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Models available on the selected backend")
        llm_layout.addWidget(self.model_combo)
        refresh_backend_btn = QPushButton(self._icon("refresh-cw"), "Refresh")
        refresh_backend_btn.clicked.connect(self.refresh_backends)
        self._track_icon(lambda b=refresh_backend_btn: b.setIcon(self._icon("refresh-cw")))
        llm_layout.addWidget(refresh_backend_btn)
        llm_group_layout.addLayout(llm_layout)

        prompt_style_row = QHBoxLayout()
        prompt_style_row.addWidget(QLabel("Summary Style:"))
        self.prompt_style_combo = QComboBox()
        self.prompt_style_combo.addItems(list(llm_backend.PROMPT_TEMPLATES.keys()) + ["Custom..."])
        self.prompt_style_combo.setToolTip("How the LLM should summarize the transcript")
        self.prompt_style_combo.currentTextChanged.connect(self.on_prompt_style_changed)
        prompt_style_row.addWidget(self.prompt_style_combo, stretch=1)
        llm_group_layout.addLayout(prompt_style_row)

        self.custom_prompt_edit = QPlainTextEdit()
        self.custom_prompt_edit.setMaximumHeight(80)
        self.custom_prompt_edit.setPlaceholderText(
            "Write your own prompt. Must include {transcript} where the transcript should be inserted."
        )
        self.custom_prompt_edit.setToolTip("The literal text {transcript} will be replaced with the full transcript.")
        self.custom_prompt_edit.setVisible(False)
        llm_group_layout.addWidget(self.custom_prompt_edit)

        self.llm_group.setLayout(llm_group_layout)
        layout.addWidget(self.llm_group)

        self._load_prompt_style_settings()

        # ----- Audio Visualization (mixed waveform + combined VU) -----
        vis_group = QGroupBox("Audio Monitor (mixed output)")
        vis_main_layout = QVBoxLayout()

        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("VU Style:"))
        self.vu_style_combo = QComboBox()
        self.vu_style_combo.addItems(vu_meters.vu_meter_style_names())
        self.vu_style_combo.currentIndexChanged.connect(self.switch_vu_style)
        style_layout.addWidget(self.vu_style_combo)
        style_layout.addStretch()
        vis_main_layout.addLayout(style_layout)

        h_layout = QHBoxLayout()
        self.waveform = vu_meters.WaveformDisplay()
        h_layout.addWidget(self.waveform, stretch=4)

        self.vu_container = QWidget()
        self.vu_container_layout = QVBoxLayout(self.vu_container)
        self.vu_container_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(self.vu_container, stretch=1)

        # self.vumeter is created by switch_vu_style itself (see the None
        # guard in that method) -- setting the combo index below fires the
        # currentIndexChanged signal we already connected above, which
        # builds and adds the widget exactly once. Don't also build it
        # here, or it gets added to the layout twice.
        self.vumeter = None
        default_index = 4  # "Analog VU-meter" -- matches the previous default
        self.vu_style_combo.setCurrentIndex(default_index)

        vis_main_layout.addLayout(h_layout)
        vis_group.setLayout(vis_main_layout)
        layout.addWidget(vis_group)

        # ----- System resource monitor (CPU/RAM/GPU/VRAM) -----
        sys_group = QGroupBox("System")
        sys_layout = QVBoxLayout()

        cpu_ram_row = QHBoxLayout()
        cpu_ram_row.addWidget(QLabel("CPU:"))
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setFormat("%p%")
        cpu_ram_row.addWidget(self.cpu_bar)

        cpu_ram_row.addWidget(QLabel("RAM:"))
        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setFormat("%p%")
        cpu_ram_row.addWidget(self.ram_bar)
        self.ram_value_label = QLabel("0.0/0.0 GB")
        self.ram_value_label.setMinimumWidth(90)
        cpu_ram_row.addWidget(self.ram_value_label)

        sys_layout.addLayout(cpu_ram_row)

        # GPUs are detected once at startup (sysmon.list_gpus()) since
        # enumerating them isn't free and the set of GPUs doesn't change
        # while the app runs; only their load/VRAM are re-sampled per tick.
        self.gpus = sysmon.list_gpus()
        self.gpu_rows = []
        for gpu in self.gpus:
            row = GpuRow(gpu)
            sys_layout.addWidget(row)
            self.gpu_rows.append(row)
        if not self.gpus:
            sys_layout.addWidget(QLabel("No GPU detected"))

        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)

        # ----- Record/Stop button + progress -----
        control_layout = QHBoxLayout()
        self.record_btn = QPushButton(self._icon("mic"), "Record")
        self.record_btn.clicked.connect(self.toggle_recording)
        self._set_button_class(self.record_btn, "primary")
        # record_btn's icon alternates mic/stop-circle with recording state
        # (see start_recording/reset_ui), not just theme -- refresh
        # whichever name is current rather than hardcoding "mic" here.
        self._track_icon(lambda: self.record_btn.setIcon(self._icon(self._record_icon_name)))
        control_layout.addWidget(self.record_btn)

        self.load_btn = QPushButton(self._icon("upload"), "Load Audio")
        self.load_btn.clicked.connect(self.load_audio_file)
        self._track_icon(lambda: self.load_btn.setIcon(self._icon("upload")))
        control_layout.addWidget(self.load_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        self.cancel_btn = QPushButton(self._icon("x"), "Cancel")
        self.cancel_btn.clicked.connect(self.cancel_current_job)
        self.cancel_btn.setEnabled(False)
        self._track_icon(lambda: self.cancel_btn.setIcon(self._icon("x")))
        control_layout.addWidget(self.cancel_btn)

        self.clear_log_btn = QPushButton(self._icon("trash-2"), "Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self._track_icon(lambda: self.clear_log_btn.setIcon(self._icon("trash-2")))
        control_layout.addWidget(self.clear_log_btn)

        layout.addLayout(control_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("monospace"))
        layout.addWidget(QLabel("Log / Summary Output:"))
        layout.addWidget(self.log_text, stretch=1)
        self._register_log_icons()

        # Console is rendered as Markdown (Qt's built-in CommonMark-ish parser,
        # available since Qt 5.14). We keep the raw Markdown source ourselves
        # in _console_md and re-render the whole document on a short debounce
        # timer rather than on every streamed token, since QTextEdit has no
        # incremental-append API for Markdown -- see append_log/append_summary.
        self._console_md = ""
        self._console_last_was_summary = False
        self._console_render_timer = QTimer(self)
        self._console_render_timer.setSingleShot(True)
        self._console_render_timer.setInterval(200)
        self._console_render_timer.timeout.connect(self._flush_console)

        # Whether the console should keep pinned to the bottom as new lines
        # arrive, vs. leaving it alone because the user scrolled up to read
        # something. Tracked live off the scrollbar's own valueChanged signal
        # rather than comparing scrollbar.value()/maximum() immediately
        # before and after each setHtml() re-render (the previous approach)
        # -- that comparison turned out unreliable on macOS specifically,
        # where the native Cocoa scrollbar doesn't report a stable maximum()
        # at the exact moment this code reads it, so "was at bottom" kept
        # evaluating False and the console appeared to jump to the top on
        # every new line even though it hadn't actually been scrolled up.
        self._log_autoscroll = True
        self.log_text.verticalScrollBar().valueChanged.connect(self._on_log_scroll_changed)

        self.last_md_path = None

        # ----- History (past sessions from ./transcripts/) -----
        # Floating overlay, not part of `layout` -- see HistorySidebar's
        # docstring. Built last so it raises above every widget already
        # added above.
        self.history_sidebar = HistorySidebar(central)
        self.history_sidebar.open_output_folder_btn.setIcon(self._icon("folder"))
        self.history_sidebar.open_output_folder_btn.clicked.connect(self.open_folder)
        self._track_icon(lambda: self.history_sidebar.open_output_folder_btn.setIcon(self._icon("folder")))
        self.history_sidebar.refresh_btn.setIcon(self._icon("refresh-cw"))
        self.history_sidebar.refresh_btn.clicked.connect(self.refresh_history)
        self._track_icon(lambda: self.history_sidebar.refresh_btn.setIcon(self._icon("refresh-cw")))
        self.history_sidebar.open_folder_btn.setIcon(self._icon("folder"))
        self.history_sidebar.open_folder_btn.clicked.connect(self.open_selected_history_folder)
        self._track_icon(lambda: self.history_sidebar.open_folder_btn.setIcon(self._icon("folder")))
        self.history_sidebar.list_widget.itemDoubleClicked.connect(self.open_history_summary)
        self.history_sidebar.handle_label.setPixmap(self._icon("clock").pixmap(16, 16))
        self._track_icon(lambda: self.history_sidebar.handle_label.setPixmap(self._icon("clock").pixmap(16, 16)))
        self.history_sidebar.raise_()

    # ------------------------------------------------------------------
    # System tray + shortcuts
    # ------------------------------------------------------------------
    def setup_tray_icon(self, icon_path):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = None
            return

        icon = QIcon(icon_path) if os.path.exists(icon_path) else self.windowIcon()
        self.tray_icon = QSystemTrayIcon(icon, self)
        self.tray_icon.setToolTip("Meeting Transcriber")

        menu = QMenu()
        self.tray_show_action = menu.addAction("Show/Hide")
        self.tray_show_action.triggered.connect(self.toggle_window_visibility)
        self.tray_record_action = menu.addAction(self._icon("mic"), "Record")
        self.tray_record_action.triggered.connect(self.toggle_recording)
        menu.addSeparator()
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

        # Sync the label/icon with self.record_btn ("Record"/"Stop") right
        # before the menu opens, rather than updating it from every place
        # that changes record_btn's text.
        def _sync_tray_record_action():
            self.tray_record_action.setText(self.record_btn.text())
            self.tray_record_action.setIcon(self.record_btn.icon())
        menu.aboutToShow.connect(_sync_tray_record_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:  # single left-click
            self.toggle_window_visibility()

    def toggle_window_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def setup_shortcuts(self):
        # Qt.ApplicationShortcut fires regardless of which widget inside the
        # app currently has focus -- not a true OS-wide global hotkey (which
        # would need a new cross-platform dependency and doesn't reliably
        # work on Wayland anyway), just "works anywhere in this app".
        self.record_shortcut = QShortcut(QKeySequence("Ctrl+Alt+R"), self)
        self.record_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.record_shortcut.activated.connect(self.toggle_recording)
        self.record_btn.setToolTip("Start/stop recording (Ctrl+Alt+R)")

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------
    def _icon(self, name, size=18):
        """themed_icon(), cached per (name, size, current text color) so
        repeated calls (e.g. rebuilding the source picker) don't re-render
        the same SVG over and over."""
        color = self.palette().color(QPalette.ColorRole.WindowText)
        key = (name, size, color.rgba())
        if key not in self._icon_cache:
            self._icon_cache[key] = themed_icon(name, color, size)
        return self._icon_cache[key]

    def _set_button_class(self, button, cls):
        """
        Sets the 'cls' dynamic property the stylesheet's
        QPushButton[cls="..."] selectors key off (see BASE_STYLESHEET),
        then forces Qt to re-evaluate the stylesheet for this widget --
        changing a property alone doesn't trigger a repaint with the new
        rule applied.
        """
        button.setProperty("cls", cls)
        button.style().unpolish(button)
        button.style().polish(button)

    def _track_icon(self, apply_fn):
        """
        Registers a zero-arg callable that re-applies a palette-tinted
        icon, so changeEvent() can redo it when the OS theme flips while
        the app is running -- our baked icon bitmaps (themed_icon()/
        self._icon()) don't repaint themselves on a palette change;
        nothing was listening for the theme change at all before this.
        """
        self._icon_refreshers.append(apply_fn)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._icon_cache.clear()
            # A refresher can point at a widget that no longer exists
            # (e.g. a removed SourceRow) -- that's a real case, not
            # speculative, so drop it instead of letting it break every
            # other icon's refresh for the rest of the session.
            for refresh in list(self._icon_refreshers):
                try:
                    refresh()
                except RuntimeError:
                    self._icon_refreshers.remove(refresh)
            # BASE_STYLESHEET's palette(...) references look dynamic, but
            # Qt's QSS engine actually caches the resolved colors per
            # widget and only recomputes them on an explicit repolish --
            # verified directly: without this, buttons/combos/line edits
            # kept rendering with the *old* theme's colors indefinitely
            # after a live palette change, while native-drawn widgets
            # (QGroupBox fill, QLabel text) updated fine on their own.
            self._repolish_widget_tree()

    def _repolish_widget_tree(self):
        # Re-assigning the stylesheet string on the two widgets that actually
        # own BASE_STYLESHEET (self and the central widget) forces Qt to
        # fully rebuild its cached style-sheet render rules for every
        # descendant, rather than relying solely on each widget's own
        # unpolish()/polish() below.
        central = self.centralWidget()
        for owner in (self, central):
            if owner is not None:
                owner.setStyleSheet(owner.styleSheet())
        # QPushButton's own text label isn't actually painted through the
        # "color: palette(buttontext)" CSS rule above -- on macOS in
        # particular, QMacStyle's native button-label painter reads
        # buttonText() straight off the *widget's own* QPalette object
        # (option->palette), not a live lookup against the QApplication's
        # palette. Reassigning each button's palette from the just-changed
        # app palette (rather than trusting Qt's normal inheritance
        # cascade, which this bridged native rendering path apparently
        # doesn't always respect) is what actually keeps text color in
        # sync, alongside the background/border colors the repolish below
        # already fixes. Scoped to QPushButton specifically -- unlike the
        # generic unpolish()/polish() loop below, blanket-resetting every
        # widget's palette this way would also stomp on the deliberate
        # fixed (non-theme) colors some custom widgets set for themselves,
        # e.g. WaveformDisplay's dark background in vu_meters.py.
        app_palette = QApplication.instance().palette()
        for button in self.findChildren(QPushButton):
            try:
                button.setPalette(app_palette)
            except RuntimeError:
                pass
        for widget in [self] + self.findChildren(QWidget):
            try:
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
            except RuntimeError:
                pass

    def _register_log_icons(self):
        """
        Registers every LOG_ICONS entry as a named image resource on the
        log console's own QTextDocument, once. append_log() then embeds
        `![i](icon-name)` Markdown image references (not raw HTML -- Qt's
        setMarkdown() silently drops inline HTML, verified directly),
        which setMarkdown() resolves against these resources on every
        re-render. Resources persist across repeated setMarkdown() calls
        on the same document, so this only needs to run once.

        Unlike _icon()'s buttons (tinted with the palette's text color),
        each of these gets its LOG_ICON_COLORS accent -- a fixed hex, not
        palette-driven, since the whole point is restoring the semantic
        color the original emoji carried. Verified contrast-safe against
        both light and dark separately (see LOG_ICON_COLORS), so a fixed
        color is fine here without needing a palette-based recompute.
        """
        document = self.log_text.document()
        for icon_name in set(LOG_ICONS.values()):
            color = QColor(LOG_ICON_COLORS[icon_name])
            pixmap = themed_icon(icon_name, color, size=14).pixmap(14, 14)
            document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(icon_name), pixmap)

    def closeEvent(self, event):
        self._save_prompt_style_settings()
        self._save_diarization_settings()
        self.mixer.shutdown()  # stop() alone would leave sources monitoring; we're closing for good
        if self.queue_worker:
            self.queue_worker.stop()
            self.queue_thread.quit()
            self.queue_thread.wait()
        sysmon.shutdown()
        event.accept()

    # ------------------------------------------------------------------
    # Audio Sources panel
    # ------------------------------------------------------------------
    def refresh_source_picker(self):
        """
        Repopulate the 'Add source' combo. Devices already part of the mix
        are shown but greyed out (disabled) so the same physical device
        can't be added twice -- Qt renders disabled combo items in grey
        automatically, no extra styling needed.
        """
        current_selection_name = self.source_picker_combo.currentData()
        current_selection_name = current_selection_name["name"] if current_selection_name else None

        self.source_picker_combo.clear()
        try:
            devices = audio_engine.list_all_sources()
        except Exception as e:
            self.append_log(f"Error listing audio devices: {e}")
            return

        model = self.source_picker_combo.model()
        restore_index = None
        first_enabled_index = None
        for d in devices:
            icon = self._icon("monitor" if d["is_loopback"] else "mic")
            already_added = d["name"] in self.mixer.sources
            label = d["name"] + ("  (already added)" if already_added else "")
            self.source_picker_combo.addItem(icon, label, d)
            row = self.source_picker_combo.count() - 1

            if already_added:
                item = model.item(row)
                if item is not None:
                    item.setEnabled(False)
            elif first_enabled_index is None:
                first_enabled_index = row

            if d["name"] == current_selection_name and not already_added:
                restore_index = row

        # Keep the previous selection if it's still pickable, otherwise land
        # on the first enabled (not-yet-added) device rather than a greyed-out one.
        if restore_index is not None:
            self.source_picker_combo.setCurrentIndex(restore_index)
        elif first_enabled_index is not None:
            self.source_picker_combo.setCurrentIndex(first_enabled_index)

    def add_selected_source(self):
        idx = self.source_picker_combo.currentIndex()
        if idx < 0:
            return

        model = self.source_picker_combo.model()
        item = model.item(idx)
        if item is not None and not item.isEnabled():
            QMessageBox.information(self, "Already added", "That device is already part of the mix.")
            return

        dev = self.source_picker_combo.itemData(idx)
        if dev is None:
            return

        name = dev["name"]
        if name in self.mixer.sources:
            # Shouldn't happen (the picker greys these out), but guard anyway.
            QMessageBox.information(self, "Already added", "That device is already part of the mix.")
            return

        try:
            self.mixer.add_source(
                name=name,
                device_id=dev["device_id"],
                samplerate=dev["samplerate"],
                channels=dev["channels"],
                is_loopback=dev["is_loopback"],
                wasapi_loopback=dev["wasapi_loopback"],
            )
        except Exception as e:
            QMessageBox.warning(self, "Could not add source", str(e))
            return

        icon_name = "monitor" if dev["is_loopback"] else "mic"
        row = SourceRow(name, name, icon=self._icon(icon_name))
        row.remove_clicked.connect(self.remove_source)
        row.gain_changed.connect(lambda n, g: self.mixer.set_gain(n, g))
        row.mute_changed.connect(lambda n, m, f: self.mixer.set_muted(n, m, f))
        if row.icon_label is not None:
            self._track_icon(lambda label=row.icon_label, n=icon_name: label.setPixmap(self._icon(n).pixmap(16, 16)))
        self._track_icon(row.remove_btn.refresh_normal_icon)
        self.source_rows[name] = row
        self.sources_layout.addWidget(row)
        self.no_sources_label.setVisible(False)
        self.append_log(f"➕ Added source: {name}")

        self.refresh_source_picker()  # grey out the one we just added

    def remove_source(self, name):
        if self.is_recording:
            QMessageBox.information(self, "Recording in progress", "Stop recording before removing a source.")
            return
        self.mixer.remove_source(name)
        self._reported_source_errors.pop(name, None)
        row = self.source_rows.pop(name, None)
        if row is not None:
            self.sources_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self.no_sources_label.setVisible(not self.source_rows)
        self.append_log(f"➖ Removed source: {name}")
        self.refresh_source_picker()  # re-enable it in the picker

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------
    def toggle_recording(self):
        if self.record_btn.text() == "Record":
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.mixer.sources:
            QMessageBox.warning(self, "No sources", "Add at least one audio source before recording.")
            return

        self.load_btn.setEnabled(False)
        self.record_btn.setText("Stop")
        self._record_icon_name = "stop-circle"
        self.record_btn.setIcon(self._icon("stop-circle"))
        self._set_button_class(self.record_btn, "danger")
        self.progress_bar.setValue(0)
        self.append_log("🎤 Recording... (press Stop to finish)")

        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()
            self.playback_timer = None
        self.loaded_samples = None  # hand the combined waveform/VU back to live monitoring

        self.is_recording = True
        self.dev_group.setEnabled(False)

        try:
            errors = self.mixer.start()
        except Exception as e:
            self.append_log(f"❌ Recording error: {e}")
            self.is_recording = False
            self.dev_group.setEnabled(True)
            self.reset_ui()
            return

        for err in errors:
            self.append_log(f"⚠️ Source failed to start: {err}")
        # No per-recording timer to start here -- self.monitor_timer already
        # runs continuously and mixer.tick() now accumulates automatically
        # because self.mixer.start() just flipped is_running to True.

    def stop_recording(self):
        mixed_audio = self.mixer.stop()  # sources keep running; monitoring continues
        self.is_recording = False
        self.dev_group.setEnabled(True)

        if mixed_audio.size == 0:
            self.append_log("⏹ No audio recorded.")
            self.load_btn.setEnabled(True)
            self.update_config_lock()
            self.reset_ui()
            return

        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()
            self.playback_timer = None
        self.loaded_samples = None

        base_dir = Path.cwd() / "transcripts"
        base_dir.mkdir(exist_ok=True)
        folder_name = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        output_dir = base_dir / folder_name
        output_dir.mkdir(exist_ok=True)
        audio_file = output_dir / "meeting.wav"
        sf.write(str(audio_file), mixed_audio, audio_engine.ENGINE_SAMPLE_RATE)
        self.append_log(f"✅ Recorded ({len(self.mixer.sources)} source(s) mixed) to {audio_file}")

        self._add_job_from_audio(str(audio_file), output_dir=output_dir)

        # Deliberately NOT blanking the waveform/VU meters here -- live
        # monitoring keeps running via self.monitor_timer, so they should
        # carry straight on showing current levels instead of flashing to zero.

        self.load_btn.setEnabled(True)
        self.update_config_lock()
        self.reset_ui()

    def update_visualization(self):
        """
        Advance the mixer by one tick and refresh the combined + per-source
        VU meters. Runs continuously (from self.monitor_timer, started once
        in __init__) regardless of whether we're recording -- that's what
        makes the VU meters live as soon as a source is added, not just
        while Record is held down. mixer.tick() itself only accumulates
        into the saved recording while self.is_recording is True; here we
        just always ask it to tick and always refresh the meters.
        """
        try:
            self.mixer.tick()

            for name, row in self.source_rows.items():
                row.vu.update_level(self.mixer.get_source_level(name))

            for name, err in self.mixer.get_source_errors().items():
                if self._reported_source_errors.get(name) != err:
                    self.append_log(f"⚠️ [{name}] {err}")
                    self._reported_source_errors[name] = err

            # The big combined waveform/VU meter is shared with the
            # loaded-file playback preview -- only drive it from live
            # monitoring when a file isn't currently being previewed.
            if self.loaded_samples is None:
                preview = self.mixer.get_mixed_preview()
                if preview:
                    self.waveform.update_buffer(preview)
                    arr = np.array(preview, dtype=np.float32)
                    rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
                    self.vumeter.update_level(rms)
                else:
                    self.waveform.update_buffer([])
                    self.vumeter.update_level(0)
        except Exception:
            pass

    def update_system_monitor(self):
        cpu_ram = sysmon.sample_cpu_ram()
        self.cpu_bar.setValue(int(cpu_ram["cpu_percent"]))
        self.ram_bar.setValue(int(cpu_ram["ram_percent"]))
        self.ram_value_label.setText(f"{cpu_ram['ram_used_gb']:.1f}/{cpu_ram['ram_total_gb']:.1f} GB")

        for gpu, row in zip(self.gpus, self.gpu_rows):
            row.update_stats(sysmon.sample_gpu(gpu))

    # ------------------------------------------------------------------
    # File loading + playback visualisation (unchanged behaviour)
    # ------------------------------------------------------------------
    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aac);;All Files (*)"
        )
        if not file_path:
            return

        try:
            data, sr = sf.read(file_path, dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            self.loaded_samples = data
            self.loaded_sr = sr
            self.playhead = 0
            if self.playback_timer:
                self.playback_timer.stop()
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self.update_playback_visualisation)
            self.playback_timer.start(33)
        except Exception as e:
            self.append_log(f"Could not load audio for visualisation: {e}")
            self.loaded_samples = None

        self._add_job_from_audio(file_path)

    def update_playback_visualisation(self):
        if self.loaded_samples is None or self.loaded_sr is None:
            return

        dt = 1.0 / 30.0
        samples_per_frame = int(self.loaded_sr * dt)
        self.playhead += samples_per_frame

        total_samples = len(self.loaded_samples)
        if self.playhead >= total_samples:
            self.playback_timer.stop()
            self.playback_timer = None
            self.waveform.update_buffer([])
            self.vumeter.update_level(0)
            return

        window_size = 16000
        start = max(0, self.playhead - window_size)
        end = min(total_samples, self.playhead)
        window = self.loaded_samples[start:end]

        if len(window) > 0:
            self.waveform.update_buffer(window.tolist())
            frame_start = max(0, self.playhead - samples_per_frame)
            frame_end = self.playhead
            frame = self.loaded_samples[frame_start:frame_end]
            if len(frame) > 0:
                rms = np.sqrt(np.mean(frame ** 2))
                self.vumeter.update_level(rms)
            else:
                self.vumeter.update_level(0)

    def clear_loaded_audio_visualization(self):
        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()
            self.playback_timer = None
        self.waveform.update_buffer([])
        self.vumeter.update_level(0)
        self.loaded_samples = None
        self.loaded_sr = None
        self.playhead = 0

    # ------------------------------------------------------------------
    # VU style switching
    # ------------------------------------------------------------------
    def switch_vu_style(self, index):
        old = self.vumeter
        if old is not None:
            self.vu_container_layout.removeWidget(old)
            old.setParent(None)
            old.deleteLater()

        self.vumeter = vu_meters.create_vu_meter(index)
        self.vu_container_layout.addWidget(self.vumeter)
        self.vu_container_layout.activate()
        self.vu_container.updateGeometry()
        self.vumeter.update()
        self.vumeter.repaint()

        if self.loaded_samples is not None:
            pass  # playback timer will refresh it
        elif self.is_recording:
            preview = self.mixer.get_mixed_preview()
            if preview:
                arr = np.array(preview, dtype=np.float32)
                rms = float(np.sqrt(np.mean(arr ** 2)))
                self.vumeter.update_level(rms)
        else:
            self.vumeter.update_level(0)

    # ------------------------------------------------------------------
    # Whisper model refresh / selection (unchanged behaviour, now backed by whisper_engine)
    # ------------------------------------------------------------------
    def refresh_whisper_models(self):
        # Rebuilding the combo fires currentIndexChanged the instant the very
        # first item is added (Qt auto-selects index 0 as soon as it goes
        # from empty to non-empty) -- and that happens *before* setItemData
        # runs for it, so on_whisper_model_changed would see incomplete data,
        # sometimes misfiring its "model not downloaded, download it?" prompt
        # for whatever model happens to be first in the list (tiny), and
        # occasionally leaving that as the selection. Block signals for the
        # whole rebuild so only our own explicit setCurrentIndex below can
        # ever trigger that handler, with the combo fully populated.
        self.whisper_combo.blockSignals(True)
        try:
            self.whisper_combo.clear()
            models_info = whisper_engine.get_whisper_models_info()

            for info in models_info:
                status = "✅ Downloaded" if info["downloaded"] else "⬇️ Not downloaded"
                display = f"{info['name']} ({info['disk_size']} disk, {info['mem_usage']} mem) {status}"
                self.whisper_combo.addItem(display)
                idx = self.whisper_combo.count() - 1
                self.whisper_combo.setItemData(idx, info, Qt.ItemDataRole.UserRole)
                tooltip = (f"Model: {info['name']}\n"
                           f"Disk: {info['disk_size']}\n"
                           f"Memory: {info['mem_usage']}\n"
                           f"Language: {info['language']}\n"
                           f"Speed: {info['speed']}\n"
                           f"Accuracy: {info['accuracy']}\n"
                           f"Usage: {info['usage']}\n"
                           f"Status: {'Downloaded' if info['downloaded'] else 'Not downloaded'}")
                self.whisper_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)

            default_name = whisper_engine.pick_default_model(models_info)
            default_index = 0
            for i in range(self.whisper_combo.count()):
                info = self.whisper_combo.itemData(i, Qt.ItemDataRole.UserRole)
                if info and info["name"] == default_name:
                    default_index = i
                    break
        finally:
            self.whisper_combo.blockSignals(False)
        self.whisper_combo.setCurrentIndex(default_index)

    def refresh_backends(self):
        self.backend_combo.clear()
        self.backends = llm_backend.detect_backends(log=self.append_log)
        if not self.backends:
            self.backend_combo.addItem("No backend detected")
            self.model_combo.clear()
            return
        for name in self.backends.keys():
            self.backend_combo.addItem(name)
        if self.backend_combo.count() > 0:
            self.on_backend_changed(0)

    def on_backend_changed(self, index):
        DEFAULT_LLM_MODEL = "gemma4:26b"
        self.model_combo.clear()
        if index < 0:
            return
        backend_name = self.backend_combo.currentText()
        if backend_name not in self.backends:
            return

        backend = self.backends[backend_name]
        all_models = backend.get("all_models") or [{"id": m, "usable": True} for m in backend["models"]]
        if not all_models:
            self.model_combo.addItem("(no models)")
            return

        item_model = self.model_combo.model()
        first_usable_index = None
        default_index = None
        for i, m in enumerate(all_models):
            label = m["id"] if m["usable"] else f"{m['id']}  (not loaded)"
            self.model_combo.addItem(label, m["id"])
            if not m["usable"]:
                item = item_model.item(i)
                if item is not None:
                    item.setEnabled(False)
            else:
                if first_usable_index is None:
                    first_usable_index = i
                if m["id"] == DEFAULT_LLM_MODEL:
                    default_index = i

        if default_index is not None:
            self.model_combo.setCurrentIndex(default_index)
        elif first_usable_index is not None:
            self.model_combo.setCurrentIndex(first_usable_index)
        else:
            # Nothing usable. Qt auto-selects index 0 as soon as the first
            # item is added, which would make the greyed-out model look
            # selected. -1 blanks the displayed text while leaving every
            # (disabled) model visible in the dropdown when clicked open.
            self.model_combo.setCurrentIndex(-1)

    def on_whisper_model_changed(self, index):
        if index < 0:
            return
        if hasattr(self, '_downloading') and self._downloading:
            return

        info = self.whisper_combo.itemData(index, Qt.ItemDataRole.UserRole)
        if not isinstance(info, dict):
            text = self.whisper_combo.currentText()
            model_name = text.split()[0] if text else "unknown"
            info = {"name": model_name, "downloaded": False, "disk_size": "?", "mem_usage": "?"}
            self.append_log(f"⚠️ Could not retrieve model info; using fallback for '{model_name}'.")

        if info.get("downloaded", False):
            return

        reply = QMessageBox.question(
            self, "Model not downloaded",
            f"The model '{info['name']}' is not present on disk.\n"
            f"Disk size: {info['disk_size']}, Memory: {info['mem_usage']}\n\n"
            "Do you want to download it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.download_model(info["name"])
        else:
            all_infos = [self.whisper_combo.itemData(i, Qt.ItemDataRole.UserRole) for i in range(self.whisper_combo.count())]
            all_infos = [i for i in all_infos if isinstance(i, dict)]
            revert_name = whisper_engine.pick_default_model(all_infos) if all_infos else None

            target_index = None
            for i in range(self.whisper_combo.count()):
                info2 = self.whisper_combo.itemData(i, Qt.ItemDataRole.UserRole)
                if isinstance(info2, dict) and info2.get("name") == revert_name and info2.get("downloaded", False):
                    target_index = i
                    break
            if target_index is not None:
                self.whisper_combo.setCurrentIndex(target_index)

    # ------------------------------------------------------------------
    # Diarization settings
    # ------------------------------------------------------------------
    def on_use_cli_toggled(self, checked):
        # Diarization needs faster-whisper's per-segment timestamps;
        # whisper-cli's plain -otxt output has none, so grey both
        # diarization and its HF token field out rather than letting the
        # user pick a combination that can't work. Remembers whether
        # diarization was checked so unchecking "Use whisper-cli" restores
        # it (and the token field along with it, via on_diarization_toggled)
        # instead of leaving it force-disabled.
        if checked:
            self._diarization_was_checked = self.diarization_check.isChecked()
            self.diarization_check.setChecked(False)
        self.diarization_check.setEnabled(not checked)
        if not checked and getattr(self, "_diarization_was_checked", False):
            self.diarization_check.setChecked(True)

    def on_diarization_toggled(self, checked):
        self.hf_token_edit.setEnabled(checked)
        self.hf_token_label.setEnabled(checked)

    def _load_diarization_settings(self):
        settings = QSettings("MeetingTranscriber", "MeetingTranscriber")
        self.hf_token_edit.setText(settings.value("diarization/hf_token", ""))

    def _save_diarization_settings(self):
        settings = QSettings("MeetingTranscriber", "MeetingTranscriber")
        settings.setValue("diarization/hf_token", self.hf_token_edit.text())

    # ------------------------------------------------------------------
    # Model download
    # ------------------------------------------------------------------
    def download_model(self, model_name):
        if hasattr(self, '_downloading') and self._downloading:
            return
        self._downloading = True
        self.update_config_lock()

        self.record_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)

        self.download_thread = QThread()
        self.download_worker = whisper_engine.DownloadWorker(model_name)
        self.download_worker.moveToThread(self.download_thread)

        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.log.connect(self.append_log)
        self.download_worker.finished.connect(self.on_download_finished)
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_worker.finished.connect(self.download_worker.deleteLater)
        self.download_thread.finished.connect(self.download_thread.deleteLater)

        self.download_thread.start()

    def on_download_finished(self, success, model_name):
        self._downloading = False
        self.update_config_lock()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.record_btn.setEnabled(True)
        self.load_btn.setEnabled(True)

        self.refresh_whisper_models()

        self.whisper_combo.blockSignals(True)
        try:
            if success:
                self.append_log(f"✅ Model '{model_name}' is now available.")
                for i in range(self.whisper_combo.count()):
                    info = self.whisper_combo.itemData(i, Qt.ItemDataRole.UserRole)
                    if info and info["name"] == model_name:
                        self.whisper_combo.setCurrentIndex(i)
                        break
            else:
                self.append_log(f"❌ Download of '{model_name}' failed. Please download manually.")
                for i in range(self.whisper_combo.count()):
                    info = self.whisper_combo.itemData(i, Qt.ItemDataRole.UserRole)
                    if info and info["downloaded"]:
                        self.whisper_combo.setCurrentIndex(i)
                        break
        finally:
            self.whisper_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Job / queue plumbing
    # ------------------------------------------------------------------
    def _add_job_from_audio(self, audio_file_path, output_dir=None):
        current_idx = self.whisper_combo.currentIndex()
        item_data = self.whisper_combo.itemData(current_idx, Qt.ItemDataRole.UserRole)
        whisper_model = item_data["name"] if item_data and "name" in item_data else self.whisper_combo.currentText().split()[0]

        backend_name = self.backend_combo.currentText()
        if backend_name not in self.backends:
            QMessageBox.warning(self, "Backend Error", "No valid LLM backend selected.")
            return
        backend_info = self.backends[backend_name].copy()
        backend_info["name"] = backend_name

        model_idx = self.model_combo.currentIndex()
        model_item = self.model_combo.model().item(model_idx) if model_idx >= 0 else None
        if model_idx < 0 or model_item is None or not model_item.isEnabled():
            QMessageBox.warning(self, "Model Error", "No usable LLM model selected -- load a model in the backend, then hit Refresh.")
            return
        llm_model = self.model_combo.itemData(model_idx)
        if not llm_model:
            QMessageBox.warning(self, "Model Error", "No LLM model selected.")
            return

        use_cli = self.use_cli_check.isChecked()
        prompt_template = self._current_prompt_template()

        job = pipeline.Job(
            audio_file_path, whisper_model, backend_info, llm_model, use_cli,
            output_dir=output_dir, prompt_template=prompt_template,
            review_transcript=self.review_transcript_check.isChecked(),
            enable_diarization=self.diarization_check.isChecked(),
            hf_token=self.hf_token_edit.text() or None,
        )
        self.queue_worker.add_job(job)
        self.job_count += 1
        self.update_config_lock()
        self.append_log(f"📥 Job #{job.id} added to queue.")
        self.cancel_btn.setEnabled(True)

    def on_job_finished(self, md_path):
        self.job_count -= 1
        self.update_config_lock()

        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()
        self.waveform.update_buffer([])
        self.vumeter.update_level(0)
        self.loaded_samples = None

        self.clear_loaded_audio_visualization()

        self.last_md_path = md_path
        self.append_log(f"✅ Markdown saved to: {md_path}")
        self.append_log(f"📁 All files in: {os.path.dirname(md_path)}")
        self.progress_bar.setValue(100)

        self.cancel_btn.setEnabled(False)
        self.refresh_history()

    def on_job_started(self, job_id):
        self.append_log(f"🔄 Processing job #{job_id}...")
        self.cancel_btn.setEnabled(True)

    def cancel_current_job(self):
        if self.queue_worker:
            self.queue_worker.stop_current_job()
            self.append_log("⏹ Cancelling current job...")
            self.cancel_btn.setEnabled(False)

    def on_transcript_ready(self, transcript):
        """
        The active ProcessingWorker is paused, polling review_result/
        review_cancelled on itself (see pipeline.ProcessingWorker.process)
        -- Continue/Cancel here just set one of those to unblock it.
        """
        processor = self.queue_worker.current_processor
        if processor is None:
            return  # job finished/cancelled before the dialog could open
        dialog = TranscriptReviewDialog(transcript, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            processor.review_result = dialog.text()
        else:
            processor.review_cancelled = True

    def on_error(self, msg):
        self.append_log(f"❌ Error: {msg}")
        self.job_count -= 1
        self.update_config_lock()
        self.reset_ui()

    def append_log(self, text):
        """Add a diagnostic log line to the console. Markdown-special characters
        are escaped so paths/messages (e.g. 'a_b*c') don't get misread as
        emphasis -- unlike append_summary, this text isn't meant to be Markdown.
        Status emoji are then swapped for `![i](icon-name)` Markdown image
        references (after escaping, so this injected syntax itself isn't
        escaped) -- see LOG_ICONS/_register_log_icons for why."""
        escaped = re.sub(r'([\\`*_\[\]#])', r'\\\1', text)
        for emoji, icon_name in LOG_ICONS.items():
            if emoji in escaped:
                # Alt text must be non-empty -- Qt's Markdown parser silently
                # ignores `![](name)` (verified directly) and only resolves
                # the image resource when there's alt text, e.g. `![i](name)`.
                escaped = escaped.replace(emoji, f"![i]({icon_name})")
        if self._console_md:
            self._console_md += "\n\n"
        self._console_md += escaped
        self._console_last_was_summary = False
        self._schedule_console_render()

    def append_summary(self, chunk):
        """Append a streamed LLM token/chunk. Chunks are concatenated raw
        (no escaping, no separators) since together they form one Markdown
        document the LLM is generating -- only the *start* of a summary run
        gets a blank-line separator from whatever preceded it."""
        if self._console_md and not self._console_last_was_summary:
            self._console_md += "\n\n"
        self._console_md += chunk
        self._console_last_was_summary = True
        self._schedule_console_render()

    def _schedule_console_render(self):
        if not self._console_render_timer.isActive():
            self._console_render_timer.start()

    def _on_log_scroll_changed(self, value):
        scrollbar = self.log_text.verticalScrollBar()
        self._log_autoscroll = value >= scrollbar.maximum() - 4

    def _flush_console(self):
        scrollbar = self.log_text.verticalScrollBar()
        was_at_bottom = self._log_autoscroll
        # setHtml() below can itself reset the scrollbar (e.g. to 0) while
        # rebuilding the document layout, which would otherwise feed back
        # into _on_log_scroll_changed and corrupt _log_autoscroll before
        # we get a chance to restore the real position -- block that.
        scrollbar.blockSignals(True)
        # Not self.log_text.setMarkdown(...) directly -- verified directly
        # that Qt's Markdown parser produces the correct `<img src="...">`
        # reference in the resulting document structure, but its inline-
        # image *painting* path fails to resolve pre-registered
        # QTextDocument.addResource() images for Markdown-sourced content
        # specifically (confirmed with a minimal repro; the identical
        # resource resolves fine via setHtml()). Parsing on a throwaway
        # document and feeding the resulting HTML to the real widget's
        # setHtml() sidesteps that bug while keeping Markdown formatting
        # (bold/italic/etc., also confirmed) and the icon images intact.
        parser = QTextDocument()
        parser.setMarkdown(self._console_md)
        self.log_text.setHtml(parser.toHtml())
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        scrollbar.blockSignals(False)
        self._log_autoscroll = was_at_bottom

    def reset_ui(self):
        self.record_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.record_btn.setText("Record")
        self._record_icon_name = "mic"
        self.record_btn.setIcon(self._icon("mic"))
        self._set_button_class(self.record_btn, "primary")

        self.progress_bar.setValue(0)

        self.clear_loaded_audio_visualization()
        self.cancel_btn.setEnabled(False)
        # No manual blanking or timer bookkeeping needed here -- self.monitor_timer
        # runs continuously and will resume driving the waveform/VU meters from
        # live source monitoring now that loaded_samples is back to None.

    # ---------- Save / Open ----------
    def open_folder(self):
        """Always the general ./transcripts/ root -- opening a *specific*
        session's folder is the History sidebar's own "Open Folder"
        button's job (open_selected_history_folder), not this one's."""
        transcripts_dir = Path.cwd() / "transcripts"
        folder = str(transcripts_dir) if transcripts_dir.exists() else os.getcwd()
        self._open_path(folder)

    def _open_path(self, path):
        """Open a file or folder with whatever the OS considers its default app."""
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def clear_log(self):
        self._console_render_timer.stop()
        self._console_md = ""
        self._console_last_was_summary = False
        self.log_text.clear()
        # QTextEdit.clear() wipes the document's registered image resources
        # too (verified directly), not just its text -- without this,
        # every log icon would silently fall back to a broken-image
        # placeholder for the rest of the session after one Clear Log click.
        self._register_log_icons()

    # ---------- Summary Style (prompt templates) ----------
    def on_prompt_style_changed(self, style_name):
        self.custom_prompt_edit.setVisible(style_name == "Custom...")

    def _load_prompt_style_settings(self):
        settings = QSettings("MeetingTranscriber", "MeetingTranscriber")
        selected_text = settings.value("prompt_style/selected_text", "Standard Minutes")
        custom_text = settings.value("prompt_style/custom_text", "")

        self.custom_prompt_edit.setPlainText(custom_text)
        index = self.prompt_style_combo.findText(selected_text)
        self.prompt_style_combo.setCurrentIndex(index if index >= 0 else 0)
        self.custom_prompt_edit.setVisible(self.prompt_style_combo.currentText() == "Custom...")

    def _save_prompt_style_settings(self):
        settings = QSettings("MeetingTranscriber", "MeetingTranscriber")
        settings.setValue("prompt_style/selected_text", self.prompt_style_combo.currentText())
        settings.setValue("prompt_style/custom_text", self.custom_prompt_edit.toPlainText())

    def _current_prompt_template(self):
        style_name = self.prompt_style_combo.currentText()
        if style_name != "Custom...":
            return llm_backend.PROMPT_TEMPLATES.get(style_name)

        custom_text = self.custom_prompt_edit.toPlainText().strip()
        if custom_text and "{transcript}" not in custom_text:
            self.append_log(
                "⚠️ Custom summary template has no {transcript} placeholder -- "
                "the transcript will not be included in the prompt."
            )
        return custom_text or None

    # ---------- History (past sessions) ----------
    def refresh_history(self):
        self.history_sidebar.list_widget.clear()
        transcripts_dir = Path.cwd() / "transcripts"
        for session in llm_backend.list_past_sessions(transcripts_dir):
            text = session["timestamp"]
            if session["summary_snippet"]:
                text += f"  —  {session['summary_snippet']}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, session)
            self.history_sidebar.list_widget.addItem(item)

    def open_selected_history_folder(self):
        item = self.history_sidebar.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "No selection", "Select a session first.")
            return
        session = item.data(Qt.ItemDataRole.UserRole)
        self._open_path(str(session["folder"]))

    def open_history_summary(self, item):
        session = item.data(Qt.ItemDataRole.UserRole)
        if not session["md_path"]:
            QMessageBox.information(self, "No summary", "This session has no summary.md (the job may not have completed).")
            return
        self._open_path(session["md_path"])

    # ---------- Lock configuration ----------
    def update_config_lock(self):
        locked = self.is_recording or self.job_count > 0 or getattr(self, '_downloading', False)
        enabled = not locked
        self.dev_group.setEnabled(enabled)
        self.whisper_group.setEnabled(enabled)
        self.llm_group.setEnabled(enabled)


# ----------------------------------------------------------------------
def _dark_palette():
    """Same color values already used (and visually verified via screenshots
    throughout development) to simulate a dark OS theme for testing."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.Text, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.Button, QColor(60, 60, 63))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(80, 140, 220))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Mid, QColor(90, 90, 95))
    palette.setColor(QPalette.ColorRole.Light, QColor(75, 75, 80))
    palette.setColor(QPalette.ColorRole.Dark, QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(70, 70, 75))
    return palette


def apply_linux_color_scheme(app):
    """
    Even on Qt 6.5+, QStyleHints.colorScheme() only reflects whatever
    platform theme integration plugin (qt5ct, qgnomeplatform, KDE's native
    integration, ...) is configured, and most Linux setups don't have one,
    so apps default to a static light palette regardless of the OS setting.
    This queries the standardized XDG Desktop Portal instead (org.freedesktop.portal.
    Settings, supported by GNOME/KDE and, via xdg-desktop-portal-cosmic,
    COSMIC too), which is desktop-environment-agnostic.

    Every failure path here (no portal service running, D-Bus unavailable,
    a malformed reply) leaves the app exactly as it behaves today -- a
    silent no-op, the same "optional feature degrades gracefully"
    convention already used for nvidia-ml-py/pyannote.audio elsewhere in
    this app. Whether the portal is actually running varies by machine
    (some Linux sandboxes/CI environments have no xdg-desktop-portal
    service at all), so both the fallback and success paths are covered
    with mocked D-Bus objects in tests/test_main.py rather than depending
    on any particular host's real D-Bus state.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        from PyQt6.QtDBus import QDBusConnection, QDBusInterface

        bus = QDBusConnection.sessionBus()
        if not bus.isConnected():
            return
        iface = QDBusInterface(
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.Settings",
            bus,
        )
        if not iface.isValid():
            return

        def read_color_scheme():
            reply = iface.call("Read", "org.freedesktop.appearance", "color-scheme")
            args = reply.arguments()
            if not args:
                return None
            value = args[0]
            # Settings.Read wraps its reply in a variant, and the setting
            # itself is stored as a variant too -- unwrap up to two layers.
            for _ in range(2):
                if hasattr(value, "variant"):
                    value = value.variant()
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        if read_color_scheme() == 1:  # 1 == prefer dark, 2 == prefer light, 0 == no preference
            app.setPalette(_dark_palette())

        def on_setting_changed(namespace, key, value):
            if namespace == "org.freedesktop.appearance" and key == "color-scheme":
                scheme = read_color_scheme()
                if scheme == 1:
                    app.setPalette(_dark_palette())
                elif scheme == 2:
                    app.setPalette(app.style().standardPalette())

        bus.connect(
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.Settings",
            "SettingChanged",
            on_setting_changed,
        )
    except Exception:
        pass  # any failure here must never block the app from starting


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Transcriber")
    app.setWindowIcon(QIcon(str(resources.resource_path("src", "icons", "icon.svg"))))
    apply_linux_color_scheme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
