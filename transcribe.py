#!/usr/bin/env python3
"""
Meeting Transcriber - PyQt5 GUI
Records audio OR loads an existing audio file,
transcribes with Whisper, summarizes with a local LLM.
All files are saved under ./transcripts/YYYY-MM-DD HH.MM.SS/
Includes live VU meter and waveform display (only for recording).
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from collections import deque

import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QProgressBar,
    QGroupBox, QFileDialog, QMessageBox, QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QTimer, QPoint, QRect, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QLinearGradient, QConicalGradient, QBrush, QPainterPath, QIcon, QFont, QRadialGradient

# ----------------------------------------------------------------------
# Custom Widgets for Audio Visualization
# ----------------------------------------------------------------------

class WaveformDisplay(QWidget):
    """Draws a scrolling waveform of the audio buffer."""
    def __init__(self, parent=None, max_samples=16000):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.audio_buffer = deque(maxlen=max_samples)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(pal)

        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(True)
        self.setStyleSheet("background: transparent;")

    def update_buffer(self, buffer):
        """Set the buffer (list or deque) to display."""
        self.audio_buffer = buffer if isinstance(buffer, deque) else deque(buffer, maxlen=16000)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # Background
        # painter.fillRect(rect, QColor(20, 20, 20))
        painter.fillRect(rect, QColor(20, 20, 20, 75))  # alpha = 191 (~75% opaque) - last number

        if not self.audio_buffer:
            # Draw "No input" text
            painter.setPen(Qt.gray)
            painter.drawText(rect, Qt.AlignCenter, "Waiting for audio...")
            return

        # Get samples
        samples = list(self.audio_buffer)
        if not samples:
            return

        # Normalize to [-1, 1] (samples are float32)
        arr = np.array(samples, dtype=np.float32)
        # Clamp to avoid spikes
        arr = np.clip(arr, -1.0, 1.0)

        # Draw waveform as a continuous path
        painter.setPen(QPen(QColor(0, 200, 255), 1.5))
        points = []
        step = max(1, len(arr) // w)
        for x in range(w):
            idx = int(x * step)
            if idx < len(arr):
                value = arr[idx]  # -1..1
                y = h // 2 - value * (h // 2 - 4)
                points.append((x, y))

        if len(points) > 1:
            path = QPainterPath()
            path.moveTo(points[0][0], points[0][1])
            for x, y in points[1:]:
                path.lineTo(x, y)
            painter.drawPath(path)

        # Draw zero line
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
        painter.drawLine(0, h//2, w, h//2)

        # Draw peak indicator (optional)
        peak = np.max(np.abs(arr))
        if peak > 0.01:
            peak_y = h // 2 - peak * (h // 2 - 4)
            painter.setPen(QPen(QColor(255, 100, 100), 2))
            painter.drawLine(w-20, int(peak_y), w-5, int(peak_y))

class BasicVUMeter(QWidget):
    """Vertical VU meter with peak hold."""
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumWidth(30)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.level = 0.0  # 0..1
        self.peak_hold = 0.0
        self.hold_counter = 0

        # Smoothing for the needle
        self.smooth_level = 0.0
        self.alpha = alpha          # lower = more inertia

        # dB scale limits
        self.db_min = -50.0
        self.db_max = 6.0
        
        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(pal)

    def update_level(self, rms):
        """Update the current RMS level (0..1)."""
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))
        
        # Normalise to 0..1
        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level)) # scale up for visibility

        # Smooth the level (inertia)
        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level   # this is what the needle draws

        # Peak hold
        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 20  # hold for ~20 frames
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995  # slow decay
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # Draw solid background (instead of transparent)
        painter.fillRect(rect, QColor(30, 30, 30, 75))

        # Draw gradient bar (green -> yellow -> red)
        bar_height = int(self.level * (h - 8))
        if bar_height > 0:
            grad = QLinearGradient(0, h, 0, 0)
            grad.setColorAt(0, QColor(0, 200, 0))
            grad.setColorAt(0.5, QColor(255, 255, 0))
            grad.setColorAt(1, QColor(255, 0, 0))
            painter.fillRect(4, h - 4 - bar_height, w - 8, bar_height, QBrush(grad))

        # Draw border
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))

        # Peak hold line
        if self.peak_hold > 0.02:
            peak_y = h - 4 - int(self.peak_hold * (h - 8))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(2, peak_y, w - 2, peak_y)

        # Draw dB labels
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.setFont(self.font())
        db_labels = [-50, -40, -30, -20, -15, -10, -5, 0, 3, 6]
        for db in db_labels:
            # Normalise db to position (0 = bottom, 1 = top)
            norm = (db - self.db_min) / (self.db_max - self.db_min)
            y = h - 4 - int(norm * (h - 8))
            if 0 <= y <= h:
                painter.drawText(2, y, f"{db}")

class RetroLEDVerticalVUMeter(QWidget):
    """
    Professional LED-style vertical audio level meter.
    Scale: -50 dBFS .. 6 dBFS
    ┌─────────────────────────────────────────────────────────────┐
    │  dBFS                                                       │
    │  ┌──────────────────────────────────────────────┐           │
    │  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │   6       │
    │  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │   3       │
    │  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │   0       │
    │  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │ -10       │
    │  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │ -20       │
    │  │  █████████████████████████████████████████   │ -30       │
    │  │  █████████████████████████████████████████   │ -40       │
    │  │  █████████████████████████████████████████   │ -50       │
    │  └──────────────────────────────────────────────┘    dB     │
    └─────────────────────────────────────────────────────────────┘
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)

        self.setMinimumWidth(80)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        
        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        self.level = 0.0
        self.display_level = 0.0

        self.peak_hold = 0.0
        self.hold_counter = 0

        self.clip_counter = 0

        self.db_min = -50.0
        self.db_max = 6.0

        # Smoothing for the needle
        self.smooth_level = 0.0
        self.alpha = alpha          # lower = more inertia

    def update_level(self, rms):
        """
        rms: 0..1
        """

        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20.0 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        level = max(0.0, min(1.0, level))

        # Fast attack / slow release
        if level > self.display_level:
            self.display_level = (self.display_level * 0.30 + level * 0.70)
        else:
            self.display_level = (self.display_level * 0.95 + level * 0.05)

        self.level = self.display_level

         # Smooth the level (inertia)
        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level   # this is what the needle draws

        # Peak hold
        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 40
        elif self.hold_counter > 0:
            self.hold_counter -= 1
        else:
            self.peak_hold *= 0.995

        # Clip detection
        if db > -0.5:
            self.clip_counter = 60
        elif self.clip_counter > 0:
            self.clip_counter -= 1

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()

        # ==================================================
        # Background
        # ==================================================
        bg = self.rect()  # Use the full widget area
        # bg = rect.adjusted(2, 2, -2, -2) # Use the less than full widget area
        grad = QLinearGradient(0, bg.top(), 0, bg.bottom())
        grad.setColorAt(0.0, QColor(45, 45, 45))
        grad.setColorAt(1.0, QColor(18, 18, 18))
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(bg, 6, 6)

        # ==================================================
        # Layout
        # ==================================================
        scale_width = 35
        meter_rect = QRect(scale_width, 15, rect.width() - scale_width - 10, rect.height() - 35)

        # ==================================================
        # Scale
        # ==================================================

        db_marks = [-50, -40, -30, -20, -10, -5, 0, 3, 6]
        painter.setPen(QColor(180, 180, 180))

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in db_marks:
            t = (db - self.db_min) / (self.db_max - self.db_min)
            y = int(meter_rect.bottom() - t * meter_rect.height())
            painter.drawLine(scale_width - 8, y, scale_width - 2, y)
            painter.drawText( 2, y + 4, 26, 10, Qt.AlignRight, str(db))

        # ==================================================
        # LEDs
        # ==================================================
        available_height = meter_rect.height()
        segments = min(40, max(10, available_height // 6))
        gap = 2
        seg_height = (available_height - (segments - 1) * gap ) / segments
        seg_height = max(2, int(seg_height))
        lit_segments = int(self.level * segments)

        for i in range(segments):
            total_height = (segments * seg_height + (segments - 1) * gap)
            start_y = (meter_rect.bottom() - total_height)
            y = start_y + (segments - 1 - i) * (seg_height + gap)
            position = i / segments

            if position < 0.75:
                on_color = QColor(0, 220, 0)

            elif position < 0.92:
                on_color = QColor(255, 220, 0)

            else:
                on_color = QColor(255, 60, 60)

            off_color = QColor(35, 35, 35)

            painter.setPen(Qt.NoPen)

            if i < lit_segments:
                painter.setBrush(on_color)
            else:
                painter.setBrush(off_color)

            painter.drawRoundedRect(meter_rect.left(), y, meter_rect.width(), seg_height, 1, 1)

        # ==================================================
        # Meter Border
        # ==================================================
        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(meter_rect)

        # ==================================================
        # Peak Hold
        # ==================================================
        if self.peak_hold > 0.01:
            peak_y = int(meter_rect.bottom() - self.peak_hold * meter_rect.height())
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(meter_rect.left(), peak_y, meter_rect.right(), peak_y)

        # ==================================================
        # Clip LED
        # ==================================================
        painter.setPen(Qt.NoPen)

        if self.clip_counter > 0:
            painter.setBrush(QColor(255, 0, 0))
        else:
            painter.setBrush(QColor(60, 0, 0))
        painter.drawEllipse(rect.width() - 18, 6, 10, 10)

        # ==================================================
        # Labels
        # ==================================================
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(4, 12, "dBFS")
        painter.drawText(rect.width() - 24, rect.height() - 5, "PK")
    
class RetroLEDHorizontalVUMeter(QWidget):
    """
    Professional LED-style vertical audio level meter.
    Scale: -50 dBFS .. 6 dBFS
    ┌─────────────────────────────────────────────────────────────┐
    │  dBFS                                                 PK    │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │█ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ ░░░░░░░░░░░░░░░░░░░░│  │  │
    │  │█ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ ░░░░░░░░░░░░░░░░░░░░│  │  │
    │  │█ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ ░░░░░░░░░░░░░░░░░░░░│  │  │
    │  └───────────────────────────────────────────────────────┘  │
    │    -50   -40   -30   -20   -10    -5     0     3     6      │
    └─────────────────────────────────────────────────────────────┘
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)

        self.setMinimumWidth(80)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        
        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        self.level = 0.0
        self.display_level = 0.0

        self.peak_hold = 0.0
        self.hold_counter = 0

        self.clip_counter = 0

        self.db_min = -50.0
        self.db_max = 6.0

        # Smoothing for the needle
        self.smooth_level = 0.0
        self.alpha = alpha          # lower = more inertia

    def update_level(self, rms):
        """
        rms: 0..1
        """

        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20.0 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        level = max(0.0, min(1.0, level))

        # Fast attack / slow release
        if level > self.display_level:
            self.display_level = (self.display_level * 0.30 + level * 0.70)
        else:
            self.display_level = (self.display_level * 0.95 + level * 0.05)

        self.level = self.display_level

        # Smooth the level (inertia)
        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        # Peak hold
        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 40
        elif self.hold_counter > 0:
            self.hold_counter -= 1
        else:
            self.peak_hold *= 0.995

        # Clip detection
        if db > -0.5:
            self.clip_counter = 60
        elif self.clip_counter > 0:
            self.clip_counter -= 1

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()

        # ==================================================
        # Background
        # ==================================================
        bg = self.rect()
        grad = QLinearGradient(0, bg.top(), 0, bg.bottom())
        grad.setColorAt(0.0, QColor(45, 45, 45))
        grad.setColorAt(1.0, QColor(18, 18, 18))
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(bg, 6, 6)

        # ==================================================
        # Layout - HORIZONTAL NOW
        # ==================================================
        meter_height = 30  # Height of the meter bars
        scale_height = 30  # Height for scale labels below
        
        # Meter rectangle now spans the width, positioned at top
        meter_rect = QRect(15, 15, rect.width() - 30, meter_height)

        # ==================================================
        # Scale - now below the meter
        # ==================================================
        db_marks = [-50, -40, -30, -20, -10, -5, 0, 3, 6]
        painter.setPen(QColor(180, 180, 180))

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in db_marks:
            t = (db - self.db_min) / (self.db_max - self.db_min)
            x = int(meter_rect.left() + t * meter_rect.width())
            y = meter_rect.bottom() + 10
            painter.drawLine(x, y - 5, x, y + 2)
            painter.drawText(x - 15, y + 15, 30, 12, Qt.AlignHCenter, str(db))

        # ==================================================
        # LEDs - HORIZONTAL now
        # ==================================================
        available_width = meter_rect.width()
        segments = min(40, max(10, available_width // 8))
        gap = 2
        seg_width = (available_width - (segments - 1) * gap) / segments
        seg_width = max(2, int(seg_width))
        
        lit_segments = int(self.level * segments)

        for i in range(segments):
            total_width = (segments * seg_width + (segments - 1) * gap)
            start_x = meter_rect.left()
            x = start_x + i * (seg_width + gap)
            
            position = i / segments

            if position < 0.75:
                on_color = QColor(0, 220, 0)
            elif position < 0.92:
                on_color = QColor(255, 220, 0)
            else:
                on_color = QColor(255, 60, 60)

            off_color = QColor(35, 35, 35)

            painter.setPen(Qt.NoPen)

            if i < lit_segments:
                painter.setBrush(on_color)
            else:
                painter.setBrush(off_color)

            painter.drawRoundedRect(x, meter_rect.top(), seg_width, meter_rect.height(), 1, 1)

        # ==================================================
        # Meter Border
        # ==================================================
        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(meter_rect)

        # ==================================================
        # Peak Hold - now vertical line
        # ==================================================
        if self.peak_hold > 0.01:
            peak_x = int(meter_rect.left() + self.peak_hold * meter_rect.width())
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(peak_x, meter_rect.top(), peak_x, meter_rect.bottom())

        # ==================================================
        # Clip LED
        # ==================================================
        painter.setPen(Qt.NoPen)

        if self.clip_counter > 0:
            painter.setBrush(QColor(255, 0, 0))
        else:
            painter.setBrush(QColor(60, 0, 0))
        painter.drawEllipse(rect.width() - 18, 6, 10, 10)

        # ==================================================
        # Labels
        # ==================================================
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(4, 12, "dBFS")
        painter.drawText(rect.width() - 24, rect.height() - 5, "PK")

class ModernVUMeter(QWidget):
    """
    Horizontal LED-style VU meter with individual segments.
    Scale: -50 dB .. +6 dB.
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.setMinimumWidth(80)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0
        self.smooth_level = 0.0
        self.alpha = alpha

        self.db_min = -50.0
        self.db_max = 6.0
        self.db_values = [-50, -40, -30, -20, -10, 0, 3, 6]
        self.segments = 24  # Number of LED segments

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

    def update_level(self, rms):
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level))

        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        # ---- Dark background with rounded corners ----
        bg_grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        bg_grad.setColorAt(0.0, QColor(35, 35, 40, 230))
        bg_grad.setColorAt(1.0, QColor(18, 18, 22, 230))
        painter.setPen(QPen(QColor(60, 60, 65), 1))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 8, 8)

        # ---- Meter area ----
        left_padding = 15
        right_padding = 15
        top_padding = 20
        bottom_padding = 30
        meter_rect = rect.adjusted(left_padding, top_padding, -right_padding, -bottom_padding)
        h = meter_rect.height()
        w = meter_rect.width()

        # ---- Draw LED segments (horizontal) ----
        segment_width = (w - (self.segments - 1) * 2) / self.segments
        segment_width = max(3, segment_width)
        lit_segments = int(self.level * self.segments)

        for i in range(self.segments):
            x = meter_rect.left() + i * (segment_width + 2)
            segment_rect = QRect(int(x), meter_rect.top(), int(segment_width), h)
            
            if i < lit_segments:
                # Determine colour based on position
                position = i / self.segments
                if position < 0.6:
                    color = QColor(0, 200, 0)      # green
                elif position < 0.85:
                    color = QColor(200, 200, 0)    # yellow
                else:
                    color = QColor(255, 50, 0)     # red
                
                # Glow effect
                glow_color = QColor(color)
                glow_color.setAlpha(40)
                painter.fillRect(segment_rect.adjusted(-1, -1, 1, 1), glow_color)
                painter.fillRect(segment_rect, color)
            else:
                # Off segment
                painter.fillRect(segment_rect, QColor(30, 30, 35))

            # Border
            painter.setPen(QPen(QColor(15, 15, 18), 1))
            painter.drawRect(segment_rect)

        # ---- Peak hold (bright white line) ----
        if self.peak_hold > 0.02:
            peak_x = int(meter_rect.left() + self.peak_hold * w)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
            painter.drawLine(peak_x, meter_rect.top() - 2, peak_x, meter_rect.bottom() + 2)

        # ---- Scale ticks (below the LED bar) ----
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in self.db_values:
            norm = (db - self.db_min) / (self.db_max - self.db_min)
            x = int(meter_rect.left() + norm * w)
            painter.drawLine(x, meter_rect.bottom() + 2, x, meter_rect.bottom() + 8)
            label = f"{db:+d}" if db > 0 else str(db)
            painter.drawText(x - 12, meter_rect.bottom() + 20, 24, 14, Qt.AlignCenter, label)

        # ---- Labels ----
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawText(rect.left() + 6, rect.top() + 14, "LED")
        painter.drawText(rect.right() - 30, rect.bottom() - 4, "dB")

class AnalogStyleVUMeter(QWidget):
    """
    Classic VU meter with a 180° semicircle.
    Scale: -60 dB (left) to +6 dB (right), with 0 at top centre.
    With smoothing (inertia) for the needle movement.
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0

        # Smoothing for the needle
        self.smooth_level = 0.0
        self.alpha = alpha          # lower = more inertia

        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        # Set background colours
        # self.setAutoFillBackground(True)
        # pal = self.palette()
        # pal.setColor(self.backgroundRole(), QColor(20, 20, 20))
        # self.setPalette(pal)

        # VU scale: -60 to +6 dB
        self.db_min = -50
        self.db_max = 6
        self.db_values = [-50, -40, -30, -20, -10, 0, 3, 6]

    def update_level(self, rms):
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        self.level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, self.level))

        # Smooth the level (inertia)
        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level   # this is what the needle draws

        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # ---- Full‑width background with rounded corners ----
        bg_margin = 0
        bg_rect = rect.adjusted(bg_margin, bg_margin, -bg_margin, -bg_margin)
        painter.setPen(QPen(QColor(60, 60, 60), 0))
        painter.setBrush(QBrush(QColor(25, 25, 25, 75)))
        painter.drawRoundedRect(bg_rect, 8, 8)

        # ---- Semicircle geometry ----
        center_x = bg_rect.center().x()
        center_y = bg_rect.bottom() - 10
        radius = min(bg_rect.width(), bg_rect.height() * 2) // 2 - 20

        # ---- Draw the coloured arc ----
        start_angle = 180 * 16          # left (9 o'clock)
        span_angle = -180 * 16          # clockwise to right (3 o'clock)

        arc_rect = QRect(center_x - radius, center_y - radius, radius * 2, radius * 2)

        # ---- Conical gradient ----
        # 0.0 = left (180°), 0.5 = right (0°)
        grad = QConicalGradient(center_x, center_y, 180)

        # Create a gradient for the arc
        grad = QConicalGradient(center_x, center_y, 225)
        grad.setColorAt(0.0, QColor(255, 50, 50))  # -20 dB
        grad.setColorAt(0.6, QColor(255, 220, 0))  # -10 dB
        grad.setColorAt(0.8, QColor(0, 220, 0))    # -3 dB
        grad.setColorAt(0.9, QColor(0, 200, 0))    # +3 dB
        grad.setColorAt(1.0, QColor(255, 50, 50))  # +6 dB

        painter.setPen(QPen(QBrush(grad), 12))       # thicker pen for visibility
        painter.drawArc(arc_rect, start_angle, span_angle)

        # ---- Scale ticks and labels ----
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in self.db_values:
            t = (db - self.db_min) / (self.db_max - self.db_min)
            angle = 180 - t * 180
            rad = np.radians(angle)

            inner_r = radius - 18
            outer_r = radius - 8
            x1 = int(center_x + inner_r * np.cos(rad))
            y1 = int(center_y - inner_r * np.sin(rad))
            x2 = int(center_x + outer_r * np.cos(rad))
            y2 = int(center_y - outer_r * np.sin(rad))
            painter.drawLine(x1, y1, x2, y2)

            label_r = radius - 28
            label_x = int(center_x + label_r * np.cos(rad))
            label_y = int(center_y - label_r * np.sin(rad))
            label = f"{db:+d}" if db > 0 else str(db)
            painter.drawText(label_x - 8, label_y - 4, label)

        # ---- Needle (simple line) ----
        needle_angle = 180 - self.level * 180
        rad = np.radians(needle_angle)
        needle_len = radius - 12
        tip_x = int(center_x + needle_len * np.cos(rad))
        tip_y = int(center_y - needle_len * np.sin(rad))

        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(center_x, center_y, tip_x, tip_y)

        # ---- Centre pivot ----
        painter.setBrush(QBrush(QColor(80, 80, 80)))
        painter.drawEllipse(center_x - 6, center_y - 6, 12, 12)

        # ---- Peak hold (red cross) ----
        if self.peak_hold > 0.02:
            peak_angle = 180 - self.peak_hold * 180
            rad = np.radians(peak_angle)
            peak_r = radius - 10
            peak_x = int(center_x + peak_r * np.cos(rad))
            peak_y = int(center_y - peak_r * np.sin(rad))
            painter.setPen(QPen(QColor(255, 50, 50), 3))
            painter.drawLine(peak_x - 4, peak_y - 4, peak_x + 4, peak_y + 4)
            painter.drawLine(peak_x + 4, peak_y - 4, peak_x - 4, peak_y + 4)

        # ---- Labels ----
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawText(rect.left() + 15, rect.top() + 25, "VU")
        painter.drawText(rect.right() - 70, rect.bottom() - 10, "dB")

class ClassicHorizontalVUMeter(QWidget):
    """
    Classic horizontal VU meter with logarithmic scale:
    60, 50, 40, 30 20, 10, 7, 5, 3, 0, 3, 6 (VU) and a moving needle.
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(20, 20, 20))
        self.setPalette(pal)

        # Smoothing for the needle
        self.smooth_level = 0.0
        self.alpha = alpha          # lower = more inertia

        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setAutoFillBackground(False)
        # self.setStyleSheet("background: transparent;")


        # Use the same min/max as the scale
        self.db_min = -50.0
        self.db_max = 6.0

        # Predefined Sifam scale: (db_value, normalized_x)
        self.scale_points = [
            (-50, 0.00), (-40, 0.10), (-30, 0.25), (-20, 0.40),
            (-10, 0.55), (-0, 0.70), (3, 0.85), (6, 1.00)
        ]
        self.db_values = [-50, -40, -30, -20, -10, 0, 3, 6]

    def update_level(self, rms):
        # Get min/max automatically from the scale
        db_min = self.scale_points[0][0]
        db_max = self.scale_points[-1][0]

        if rms < 1e-10:
            db = db_min
        else:
            db = 20 * np.log10(rms)
            db = max(db_min, min(db_max, db))

        self.level = self._db_to_norm(db)
        self.level = max(0, min(1, self.level))

        # Smoothing (inertia)
        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        # Peak hold logic (unchanged)
        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def _db_to_norm(self, db):
        """Interpolate db to normalized x (0..1) using the Sifam scale."""
        if db <= self.scale_points[0][0]:
            return self.scale_points[0][1]
        if db >= self.scale_points[-1][0]:
            return self.scale_points[-1][1]
        for i in range(len(self.scale_points) - 1):
            db0, x0 = self.scale_points[i]
            db1, x1 = self.scale_points[i + 1]
            if db0 <= db <= db1:
                t = (db - db0) / (db1 - db0)
                return x0 + t * (x1 - x0)
        return 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # Background with rounded corners
        # bg_rect = self.rect()  # Use the full widget area
        bg_rect = rect.adjusted(8, 8, -8, -8) # Use the less than full widget area
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.setBrush(QBrush(QColor(25, 25, 25, 75)))
        painter.drawRoundedRect(bg_rect, 8, 8)

        left = bg_rect.left() + 20
        right = bg_rect.right() - 20
        top = bg_rect.top() + 20
        bottom = bg_rect.bottom() - 20
        scale_height = bottom - top

        # Draw the coloured zone (green -> yellow -> red)
        grad = QLinearGradient(left, 0, right, 0)
        grad.setColorAt(0.0, QColor(0, 180, 0))
        grad.setColorAt(0.55, QColor(0, 200, 0))
        grad.setColorAt(0.70, QColor(255, 255, 0))
        grad.setColorAt(0.85, QColor(255, 150, 0))
        grad.setColorAt(1.0, QColor(255, 0, 0))
        bar_y = top + scale_height // 2 - 4
        painter.fillRect(left, bar_y, right - left, 8, QBrush(grad))

        # Scale ticks and labels
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in self.db_values:
            x = int(left + self._db_to_norm(db) * (right - left))
            # Major tick
            painter.drawLine(x, bottom, x, bottom + 8)
            # Label
            label = f"{db:+d}" if db > 0 else str(db)
            painter.drawText(x - 12, bottom + 18, 24, 14, Qt.AlignCenter, label)

        # Minor ticks (optional, between major ones)
        minor_db = [-15, -12, -8, -6, -1, 1, 4, 7]
        for db in minor_db:
            x = int(left + self._db_to_norm(db) * (right - left))
            painter.drawLine(x, bottom, x, bottom + 4)

        # Needle
        needle_x = int(left + self.level * (right - left))
        # Draw needle as a triangle pointing up
        needle_tip = QPoint(needle_x, top)
        needle_base_left = QPoint(needle_x - 6, top + 22)
        needle_base_right = QPoint(needle_x + 6, top + 22)
        path = QPainterPath()
        path.moveTo(needle_tip)
        path.lineTo(needle_base_left)
        path.lineTo(needle_base_right)
        path.closeSubpath()
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QBrush(QColor(255, 200, 50)))
        painter.drawPath(path)

        # Pivot dot
        pivot = QPoint(needle_x, top + 22)
        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.drawEllipse(pivot, 4, 4)

        # Peak hold (red dot / line)
        if self.peak_hold > 0.02:
            peak_x = int(left + self.peak_hold * (right - left))
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawLine(peak_x, top - 4, peak_x, top + 4)

        # Labels
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawText(bg_rect.left() + 4, bg_rect.top() + 14, "VU")
        painter.drawText(bg_rect.right() - 30, bg_rect.bottom() - 4, "dB")
        # Optional: "dreamstime" style branding
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawText(bg_rect.right() - 70, bg_rect.bottom() - 4, "Sifam")

class GlassVUMeter(QWidget):
    """
    Sleek glass‑style horizontal VU meter with a glowing needle.
    Scale: -50 dB .. +6 dB.
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0

        # Smoothing
        self.smooth_level = 0.0
        self.alpha = alpha

        # dB range (same as others)
        self.db_min = -50.0
        self.db_max = 6.0

        # Scale tick values to display
        self.db_values = [-50, -40, -30, -20, -10, 0, 3, 6]

        # Enable transparency and set background
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        # Give it a dark background (like the others)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(20, 20, 20))
        self.setPalette(pal)

    def update_level(self, rms):
        """Update the meter with a new RMS value (0..1)."""
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        # Normalise to 0..1
        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level))

        # Smoothing (inertia)
        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        # Peak hold
        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()  # Use the full widget area
        # rect = rect.adjusted(8, 8, -8, -8) # Use the less than full widget area

        # ---- Background: glass gradient ----
        bg_grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        bg_grad.setColorAt(0.0, QColor(50, 50, 55))
        bg_grad.setColorAt(1.0, QColor(20, 20, 25))
        painter.setPen(QPen(QColor(70, 70, 80), 1))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 8, 8)

        # ---- Inner layout ----
        left = rect.left() + 25
        right = rect.right() - 25
        top = rect.top() + 10
        bottom = rect.bottom() - 10
        center_y = (top + bottom) // 2
        scale_height = bottom - top

        # ---- Scale ticks and labels ----
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in self.db_values:
            norm = (db - self.db_min) / (self.db_max - self.db_min)
            x = int(left + norm * (right - left))
            # Major tick (line above the centre)
            painter.drawLine(x, center_y - 12, x, center_y - 4)
            # Label
            label = f"{db:+d}" if db > 0 else str(db)
            painter.drawText(x - 12, center_y - 16, 24, 14, Qt.AlignCenter, label)

        # Minor ticks (optional)
        minor_db = [-45, -35, -25, -15, -5, -1, 1, 4]
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        for db in minor_db:
            norm = (db - self.db_min) / (self.db_max - self.db_min)
            x = int(left + norm * (right - left))
            painter.drawLine(x, center_y - 8, x, center_y - 4)

        # ---- Coloured bar behind the needle (optional) ----
        # A thin gradient bar that fills up to the current level
        bar_height = 6
        bar_y = center_y + 4
        bar_left = left
        bar_right = int(left + self.level * (right - left))
        if bar_right > bar_left:
            grad = QLinearGradient(bar_left, 0, bar_right, 0)
            grad.setColorAt(0.0, QColor(0, 180, 0))
            grad.setColorAt(0.6, QColor(255, 220, 0))
            grad.setColorAt(0.9, QColor(255, 150, 0))
            grad.setColorAt(1.0, QColor(255, 0, 0))
            painter.fillRect(bar_left, bar_y, bar_right - bar_left, bar_height, QBrush(grad))

        # ---- Needle with glow ----
        needle_x = int(left + self.level * (right - left))
        needle_top = center_y + 4
        needle_bottom = bottom - 2

        # Glow (soft shadow) – draw a thicker, transparent line
        glow_color = QColor(255, 140, 50, 60)
        painter.setPen(QPen(glow_color, 6, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(needle_x, needle_top, needle_x, needle_bottom)

        # Main needle (thin, bright)
        painter.setPen(QPen(QColor(255, 120, 50), 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(needle_x, needle_top, needle_x, needle_bottom)

        # ---- Peak hold (yellow dot) ----
        if self.peak_hold > 0.02:
            peak_x = int(left + self.peak_hold * (right - left))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 0))
            painter.drawEllipse(peak_x - 4, center_y - 16, 8, 8)

        # ---- Labels ----
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawText(rect.left() + 6, rect.top() + 14, "dBFS")
        painter.drawText(rect.right() - 24, rect.bottom() - 4, "Peak")

class LiquidGlassVUMeter(QWidget):
    """
    Smooth liquid-glass style VU meter with fluid-like movement.
    Scale: -50 dB .. +6 dB.
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.setMinimumWidth(50)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0
        self.smooth_level = 0.0
        self.alpha = alpha

        self.db_min = -50.0
        self.db_max = 6.0
        self.db_values = [-50, -40, -30, -20, -10, 0, 3, 6]

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

    def update_level(self, rms):
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level))

        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        # ---- Background: dark with subtle gradient ----
        bg_grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        bg_grad.setColorAt(0.0, QColor(30, 35, 40, 230))
        bg_grad.setColorAt(1.0, QColor(15, 18, 22, 230))
        painter.setPen(QPen(QColor(60, 65, 70), 1))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 8, 8)

        # ---- Meter area (the "glass tube") ----

        left_padding = 10
        right_padding = 35
        top_padding = 25
        bottom_padding = 25
        meter_rect = rect.adjusted(left_padding, top_padding, -right_padding, -bottom_padding)
        h = meter_rect.height()
        w = meter_rect.width()

        # ---- Glass tube with rounded ends ----
        tube_rect = QRect(meter_rect.left(), meter_rect.top(), w, h)
        
        # Glass tube background (semi-transparent)
        painter.setPen(QPen(QColor(80, 85, 90, 100), 1))
        painter.setBrush(QBrush(QColor(40, 45, 50, 60)))
        painter.drawRoundedRect(tube_rect, 6, 6)

        # ---- Liquid level (gradient bar with glow) ----
        bar_height = int(self.level * (h - 4))
        if bar_height > 2:
            bar_rect = QRect(
                meter_rect.left() + 2,
                meter_rect.bottom() - 2 - bar_height,
                w - 4,
                bar_height
            )
            
            # Liquid gradient (cyan/blue to green)
            grad = QLinearGradient(0, bar_rect.bottom(), 0, bar_rect.top())
            grad.setColorAt(0.0, QColor(0, 200, 255, 200))    # cyan at bottom
            grad.setColorAt(0.4, QColor(0, 180, 255, 200))    # bright blue
            grad.setColorAt(0.7, QColor(0, 220, 200, 200))    # teal
            grad.setColorAt(1.0, QColor(0, 255, 150, 200))    # green at top
            
            # Draw liquid with rounded bottom corners
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(grad))
            
            # Create rounded rectangle for liquid (only bottom rounded)
            path = QPainterPath()
            path.addRoundedRect(QRectF(bar_rect), 4, 4)
            painter.drawPath(path)
            
            # ---- Liquid glow (soft glow at the bottom) ----
            glow_rect = QRect(
                bar_rect.left(),
                bar_rect.bottom() - min(20, bar_height),
                bar_rect.width(),
                min(20, bar_height)
            )
            glow_grad = QLinearGradient(0, glow_rect.bottom(), 0, glow_rect.top())
            glow_grad.setColorAt(0.0, QColor(0, 200, 255, 0))
            glow_grad.setColorAt(1.0, QColor(0, 200, 255, 60))
            painter.fillRect(glow_rect, glow_grad)
            
            # ---- Glass reflection (shiny highlight) ----
            highlight_rect = QRect(
                bar_rect.left() + 3,
                bar_rect.top() + 2,
                bar_rect.width() // 4,
                max(2, bar_height // 2)
            )
            highlight_grad = QLinearGradient(0, highlight_rect.top(), 0, highlight_rect.bottom())
            highlight_grad.setColorAt(0.0, QColor(255, 255, 255, 40))
            highlight_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(highlight_rect, highlight_grad)

        # ---- Glass tube border highlight ----
        painter.setPen(QPen(QColor(150, 200, 255, 40), 1))
        painter.drawRoundedRect(tube_rect, 6, 6)

        # ---- Scale ticks (right side) ----
        painter.setPen(QPen(QColor(180, 190, 200), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in self.db_values:
            norm = (db - self.db_min) / (self.db_max - self.db_min)
            y = int(meter_rect.bottom() - norm * h)
            painter.drawLine(meter_rect.right(), y, meter_rect.right() + 8, y)
            label = f"{db:+d}" if db > 0 else str(db)
            painter.drawText(meter_rect.right() + 12, y + 4, label)

        # ---- Peak hold (glowing dot) ----
        if self.peak_hold > 0.02:
            peak_y = int(meter_rect.bottom() - self.peak_hold * (h - 4) - 2)
            peak_x = meter_rect.left() + w // 2
            
            # Glow
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 200, 255, 60))
            painter.drawEllipse(peak_x - 10, peak_y - 10, 20, 20)
            # Core
            painter.setBrush(QColor(0, 200, 255))
            painter.drawEllipse(peak_x - 4, peak_y - 4, 8, 8)
            # White center
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(peak_x - 2, peak_y - 2, 4, 4)

        # ---- Labels ----
        painter.setPen(QPen(QColor(180, 190, 200), 1))
        painter.drawText(rect.left() + 6, rect.top() + 16, "LIQUID")
        painter.drawText(rect.right() - 24, rect.bottom() - 4, "dB")

class NeonRetroVUMeter(QWidget):
    """
    80s Synthwave style VU meter with neon glow.
    Scale: -50..+6 dB.
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0
        self.smooth_level = 0.0
        self.alpha = alpha

        # Enable transparency and set background
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        self.db_min = -50.0
        self.db_max = 6.0
        self.db_values = [-50, -40, -30, -20, -10, 0, 3, 6]

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(10, 5, 20))
        self.setPalette(pal)

    def update_level(self, rms):
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level))

        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()  # Use the full widget area
        # rect = rect.adjusted(8, 8, -8, -8) # Use the less than full widget area

        # Dark background
        painter.setPen(QPen(QColor(50, 30, 70), 1))
        painter.setBrush(QBrush(QColor(10, 5, 20, 200)))  # Semi-transparent
        painter.drawRoundedRect(rect, 8, 8) 

        # Grid lines (subtle)
        painter.setPen(QPen(QColor(30, 20, 50), 1))
        for x in range(rect.left() + 20, rect.right() - 20, 20):
            painter.drawLine(x, rect.top() + 10, x, rect.bottom() - 10)

        # Meter area
        left = rect.left() + 15
        right = rect.right() - 15
        top = rect.top() + 10
        bottom = rect.bottom() - 10
        center_y = (top + bottom) // 2

        # Neon bar (cyan glow)
        bar_height = 8
        bar_y = center_y - bar_height // 2
        bar_left = left
        bar_right = int(left + self.level * (right - left))
        
        if bar_right > bar_left:
            # Glow effect (thick, transparent)
            glow_color = QColor(0, 255, 255, 40)
            painter.setPen(QPen(glow_color, 20, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(bar_left, center_y, bar_right, center_y)
            
            # Main bar (cyan)
            grad = QLinearGradient(bar_left, 0, bar_right, 0)
            grad.setColorAt(0.0, QColor(255, 0, 200))    # neon pink
            grad.setColorAt(0.5, QColor(0, 255, 255))    # cyan
            grad.setColorAt(1.0, QColor(255, 0, 200))    # neon pink
            painter.fillRect(bar_left, bar_y, bar_right - bar_left, bar_height, QBrush(grad))

        # Scale ticks (neon pink)
        painter.setPen(QPen(QColor(255, 0, 200, 150), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in self.db_values:
            norm = (db - self.db_min) / (self.db_max - self.db_min)
            x = int(left + norm * (right - left))
            painter.drawLine(x, center_y - 6, x, center_y - 2)
            label = f"{db:+d}" if db > 0 else str(db)
            painter.drawText(x - 12, center_y - 8, 24, 14, Qt.AlignCenter, label)

        # Neon needle (pink with glow)
        needle_x = int(left + self.level * (right - left))
        
        # Glow
        glow_color = QColor(255, 0, 200, 60)
        painter.setPen(QPen(glow_color, 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(needle_x, center_y + 10, needle_x, bottom - 4)
        
        # Main needle
        painter.setPen(QPen(QColor(255, 0, 200), 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(needle_x, center_y + 10, needle_x, bottom - 4)

        # Peak hold (white dot with glow)
        if self.peak_hold > 0.02:
            peak_x = int(left + self.peak_hold * (right - left))
            # Glow
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 60))
            painter.drawEllipse(peak_x - 6, center_y - 20, 12, 12)
            # Core
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(peak_x - 3, center_y - 17, 6, 6)

        # Labels
        painter.setPen(QPen(QColor(255, 0, 200, 200), 1))
        painter.drawText(rect.left() + 6, rect.top() + 14, "80s")
        painter.drawText(rect.right() - 24, rect.bottom() - 4, "dB")

class TubeAmplifierVUMeter(QWidget):
    """
    Vintage tube amplifier style VU meter with warm amber glow.
    Scale: -50..+6 dB.
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0
        self.smooth_level = 0.0
        self.alpha = alpha

        # Enable transparency and set background
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        self.db_min = -50.0
        self.db_max = 6.0
        self.db_values = [-50, -40, -30, -20, -10, 0, 3, 6]

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(30, 20, 10))
        self.setPalette(pal)

    def update_level(self, rms):
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level))

        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()  # Use the full widget area
        # rect = rect.adjusted(8, 8, -8, -8) # Use the less than full widget area

        # Warm amber background with vignette
        bg_grad = QRadialGradient(rect.center(), rect.width() * 0.7)
        bg_grad.setColorAt(0.0, QColor(60, 40, 20, 230))
        bg_grad.setColorAt(0.7, QColor(30, 20, 10, 230))
        bg_grad.setColorAt(1.0, QColor(15, 10, 5, 230))
        painter.setPen(QPen(QColor(80, 60, 30), 1))
        painter.setBrush(QBrush(bg_grad))
        painter.drawRoundedRect(rect, 8, 8)

        # Inner glow (warm ring)
        inner_rect = rect.adjusted(10, 10, -10, -10)
        grad = QRadialGradient(inner_rect.center(), inner_rect.width() * 0.5)
        grad.setColorAt(0.0, QColor(100, 70, 30, 0))
        grad.setColorAt(0.8, QColor(100, 70, 30, 20))
        grad.setColorAt(1.0, QColor(100, 70, 30, 80))
        painter.fillRect(inner_rect, grad)

        # Meter area (horizontal)
        left = rect.left() + 15
        right = rect.right() - 15
        top = rect.top() + 10
        bottom = rect.bottom() - 10
        center_y = (top + bottom) // 2

        # Warm glow bar (amber)
        bar_height = 10
        bar_y = center_y - bar_height // 2
        bar_left = left
        bar_right = int(left + self.level * (right - left))
        
        if bar_right > bar_left:
            # Glow
            glow_color = QColor(255, 150, 50, 40)
            painter.setPen(QPen(glow_color, 30, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(bar_left, center_y, bar_right, center_y)
            
            # Main bar (amber gradient)
            grad = QLinearGradient(bar_left, 0, bar_right, 0)
            grad.setColorAt(0.0, QColor(200, 100, 0))
            grad.setColorAt(0.5, QColor(255, 180, 50))
            grad.setColorAt(1.0, QColor(255, 80, 0))
            painter.fillRect(bar_left, bar_y, bar_right - bar_left, bar_height, QBrush(grad))

        # Scale ticks (warm white)
        painter.setPen(QPen(QColor(200, 180, 150), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in self.db_values:
            norm = (db - self.db_min) / (self.db_max - self.db_min)
            x = int(left + norm * (right - left))
            painter.drawLine(x, center_y - 8, x, center_y - 2)
            label = f"{db:+d}" if db > 0 else str(db)
            painter.drawText(x - 12, center_y - 10, 24, 14, Qt.AlignCenter, label)

        # Amber needle
        needle_x = int(left + self.level * (right - left))
        
        # Glow
        glow_color = QColor(255, 150, 50, 60)
        painter.setPen(QPen(glow_color, 8, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(needle_x, center_y + 8, needle_x, bottom - 4)
        
        # Main needle
        painter.setPen(QPen(QColor(255, 200, 100), 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(needle_x, center_y + 8, needle_x, bottom - 4)

        # Peak hold (amber dot)
        if self.peak_hold > 0.02:
            peak_x = int(left + self.peak_hold * (right - left))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 200, 100, 200))
            painter.drawEllipse(peak_x - 4, center_y - 18, 8, 8)
            # Glow
            painter.setBrush(QColor(255, 200, 100, 40))
            painter.drawEllipse(peak_x - 8, center_y - 22, 16, 16)

        # Labels
        painter.setPen(QPen(QColor(200, 180, 150), 1))
        painter.drawText(rect.left() + 6, rect.top() + 16, "TUBE")
        painter.drawText(rect.right() - 24, rect.bottom() - 4, "VU")

class ClassicBBCPPM(QWidget):
    """
    Classic BBC PPM (Peak Program Meter) style.
    Scale: 1..7 (mapped to -50..+6 dBFS).
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.setMinimumWidth(50)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0
        self.smooth_level = 0.0
        self.alpha = alpha

        # Enable transparency and set background
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        # dB range mapping
        self.db_min = -50.0
        self.db_max = 6.0
        
        # BBC PPM scale: PPM value -> dB
        self.ppm_scale = [
            (1, -50), (2, -40), (3, -30), (4, -20), 
            (5, -10), (6, 0), (7, 6)
        ]
        self.ppm_values = [1, 2, 3, 4, 5, 6, 7]

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(40, 40, 45))
        self.setPalette(pal)

    def update_level(self, rms):
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level))

        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def _db_to_ppm(self, db):
        """Convert dB to PPM value."""
        for i, (ppm, ppm_db) in enumerate(self.ppm_scale):
            if db <= ppm_db:
                return ppm + (db - ppm_db) / (self.ppm_scale[i+1][1] - ppm_db) if i < len(self.ppm_scale)-1 else ppm
        return self.ppm_scale[-1][0]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect() # Use the full widget area
        # rect = self.rect().adjusted(6, 6, -6, -6) # Use the less than full widget area

        # Off-white background (retro)
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.setBrush(QBrush(QColor(220, 218, 210, 230)))  # Semi-transparent
        painter.drawRoundedRect(rect, 8, 8)

        # Meter area
        left_padding = 55
        right_padding = 15
        top_padding = 10
        bottom_padding = 20
        meter_rect = rect.adjusted(left_padding, top_padding, -right_padding, -bottom_padding)
        h = meter_rect.height()

        # Draw the coloured zone (green/yellow/red) as background
        grad = QLinearGradient(0, meter_rect.bottom(), 0, meter_rect.top())
        grad.setColorAt(0.0, QColor(0, 180, 0))
        grad.setColorAt(0.6, QColor(255, 220, 0))
        grad.setColorAt(0.85, QColor(255, 150, 0))
        grad.setColorAt(1.0, QColor(255, 0, 0))
        painter.fillRect(meter_rect, QBrush(grad))

        # Black bar overlay (showing current level)
        bar_height = int(self.level * h)
        if bar_height > 0:
            black_rect = QRect(
                meter_rect.left(),
                meter_rect.bottom() - bar_height,
                meter_rect.width(),
                bar_height
            )
            painter.fillRect(black_rect, QColor(0, 0, 0, 200))

        # PPM scale ticks (on the left)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for ppm in self.ppm_values:
            # Find dB for this PPM value
            db = next((d for p, d in self.ppm_scale if p == ppm), None)
            if db is not None:
                norm = (db - self.db_min) / (self.db_max - self.db_min)
                y = int(meter_rect.bottom() - norm * h)
                painter.drawLine(meter_rect.left() - 8, y, meter_rect.left(), y)
                painter.drawText(meter_rect.left() - 20, y + 4, str(ppm))

        # Peak hold line
        if self.peak_hold > 0.02:
            peak_y = int(meter_rect.bottom() - self.peak_hold * h)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawLine(meter_rect.left(), peak_y, meter_rect.right(), peak_y)

        # Labels
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawText(rect.left() + 8, rect.top() + 16, "PPM")
        painter.drawText(rect.right() - 30, rect.bottom() - 4, "BBC")

class NordicVUMeter(QWidget):
    """
    Clean Nordic-style VU meter with white background and thin needle.
    Scale: -50..+6 dB.
    """
    def __init__(self, parent=None, alpha=0.15):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.setMinimumWidth(50)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0
        self.smooth_level = 0.0
        self.alpha = alpha

        # Enable transparency and set background
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

        self.db_min = -50.0
        self.db_max = 6.0
        self.db_values = [-50, -40, -30, -20, -10, 0, 3, 6]

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(245, 242, 235))
        self.setPalette(pal)

    def update_level(self, rms):
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level))

        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect() # Use the full widget area
        # rect = self.rect().adjusted(6, 6, -6, -6) # Use the less than full widget area

        # White background with thin border
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.setBrush(QBrush(QColor(245, 242, 235, 230)))  # Semi-transparent
        painter.drawRoundedRect(rect, 8, 8)

        # Meter area
        left_padding = 20
        right_padding = 45
        top_padding = 25
        bottom_padding = 25
        meter_rect = rect.adjusted(left_padding, top_padding, -right_padding, -bottom_padding)
        h = meter_rect.height()
        w = meter_rect.width()

        # Red zone (above 0 dB)
        # 0 dB corresponds to norm = (0 - (-50)) / 56 ≈ 0.893
        red_zone_start = (0 - self.db_min) / (self.db_max - self.db_min)
        red_y = int(meter_rect.bottom() - red_zone_start * h)
        if red_y > meter_rect.top():
            red_rect = QRect(meter_rect.left(), meter_rect.top(), w, red_y - meter_rect.top())
            painter.fillRect(red_rect, QColor(255, 200, 200))

        # Black bar (current level)
        bar_height = int(self.level * h)
        if bar_height > 0:
            bar_rect = QRect(
                meter_rect.left(),
                meter_rect.bottom() - bar_height,
                w,
                bar_height
            )
            painter.fillRect(bar_rect, QColor(0, 0, 0, 180))

        # Scale ticks (right side)
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for db in self.db_values:
            norm = (db - self.db_min) / (self.db_max - self.db_min)
            y = int(meter_rect.bottom() - norm * h)
            painter.drawLine(meter_rect.right(), y, meter_rect.right() + 8, y)
            label = f"{db:+d}" if db > 0 else str(db)
            painter.drawText(meter_rect.right() + 12, y + 4, label)

        # Needle (thin line)
        needle_x = int(meter_rect.left() + self.level * w)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawLine(needle_x, meter_rect.top(), needle_x, meter_rect.bottom())

        # Peak hold (red dot)
        if self.peak_hold > 0.02:
            peak_x = int(meter_rect.left() + self.peak_hold * w)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(peak_x - 3, meter_rect.bottom() + 6, 6, 6)

        # Labels
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawText(rect.left() + 6, rect.top() + 18, "VU")
        painter.drawText(rect.right() - 24, rect.bottom() - 4, "dB")

class LEDMatrixBarMeter(QWidget):
    """
    Rectangular LED matrix style VU meter.
    A grid of small square LEDs that light up from bottom to top.
    Scale: -50 dBFS .. +6 dBFS.
    """
    def __init__(self, parent=None, alpha=0.15, cols=16, rows=8):
        super().__init__(parent)
        self.setMinimumSize(120, 80)
        self.setMaximumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.cols = cols              # number of columns (horizontal)
        self.rows = rows              # number of rows (vertical)
        self.level = 0.0
        self.peak_hold = 0.0
        self.hold_counter = 0
        self.smooth_level = 0.0
        self.alpha = alpha

        self.db_min = -50.0
        self.db_max = 6.0

        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

    def update_level(self, rms):
        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20 * np.log10(rms)
            db = max(self.db_min, min(self.db_max, db))

        level = (db - self.db_min) / (self.db_max - self.db_min)
        self.level = max(0.0, min(1.0, level))

        self.smooth_level = self.smooth_level * (1.0 - self.alpha) + self.level * self.alpha
        self.smooth_level = max(0.0, min(1.0, self.smooth_level))
        self.level = self.smooth_level

        if self.level > self.peak_hold:
            self.peak_hold = self.level
            self.hold_counter = 30
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1
            else:
                self.peak_hold *= 0.995
                if self.peak_hold < 0.01:
                    self.peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        # ---- Background ----
        margin = 8
        bg_rect = rect.adjusted(margin, margin, -margin, -margin)
        painter.setPen(QPen(QColor(50, 50, 55), 1))
        painter.setBrush(QBrush(QColor(15, 15, 20)))
        painter.drawRoundedRect(bg_rect, 4, 4)

        # ---- LED grid layout ----
        inner_margin = 4
        grid_rect = bg_rect.adjusted(inner_margin, inner_margin, -inner_margin, -inner_margin)
        cell_width = grid_rect.width() / self.cols
        cell_height = grid_rect.height() / self.rows
        gap = 2   # spacing between LEDs

        # Total number of LEDs = cols * rows
        total_leds = self.cols * self.rows
        lit_count = int(self.level * total_leds)
        lit_count = min(lit_count, total_leds)

        # We fill from bottom-left upwards, column by column (or row by row)
        # For a classic bar, we fill from bottom to top.
        # Let's fill row by row, from bottom row upwards.
        led_index = 0
        for row in range(self.rows - 1, -1, -1):   # bottom to top
            for col in range(self.cols):           # left to right
                x = grid_rect.left() + col * cell_width + gap/2
                y = grid_rect.top() + row * cell_height + gap/2
                w = cell_width - gap
                h = cell_height - gap

                # Determine colour based on position in the matrix
                # Position 0..1 (normalised by total LEDs)
                pos = led_index / total_leds if total_leds > 0 else 0

                if led_index < lit_count:
                    # Colour gradient: green -> yellow -> red
                    if pos < 0.6:
                        color = QColor(0, 200, 0)      # green
                    elif pos < 0.85:
                        color = QColor(200, 200, 0)    # yellow
                    else:
                        color = QColor(255, 50, 0)     # red

                    # Glow effect
                    glow_color = QColor(color)
                    glow_color.setAlpha(60)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(glow_color)
                    painter.drawRect(int(x-1), int(y-1), int(w+2), int(h+2))

                    # Main LED
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
                    painter.drawRect(int(x), int(y), int(w), int(h))
                else:
                    # Off LED – dark grey
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(QColor(30, 30, 35)))
                    painter.drawRect(int(x), int(y), int(w), int(h))

                led_index += 1

        # ---- Peak hold (bright white dot) ----
        if self.peak_hold > 0.02:
            peak_led_index = int(self.peak_hold * total_leds)
            peak_led_index = min(peak_led_index, total_leds - 1)
            # Find row and column for this index
            # Since we fill bottom-up, we need to map index to (row, col)
            # row = (total_leds - 1 - peak_led_index) // cols
            # col = (total_leds - 1 - peak_led_index) % cols
            # But we filled bottom-up, so the last LED is at top-right.
            # Let's just compute the position from the fraction.
            peak_frac = self.peak_hold
            # Map fraction to a position in the grid: x = fraction * cols, y = fraction * rows
            # We'll place it on the right edge at the appropriate height.
            peak_x = grid_rect.left() + peak_frac * grid_rect.width()
            peak_y = grid_rect.bottom() - 4   # near the bottom
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(int(peak_x-4), int(peak_y-4), 8, 8)

        # ---- Labels ----
        painter.setPen(QPen(QColor(150, 150, 160), 1))
        painter.drawText(bg_rect.left() + 4, bg_rect.top() + 14, "VU")
        painter.drawText(bg_rect.right() - 24, bg_rect.bottom() - 4, "dB")

class BroadcastStereoVUMeter(QWidget):
    """
    Stereo LED broadcast VU meter.

    update_levels(left_rms, right_rms)

    RMS values expected:
        0.0 .. 1.0
    """

    def __init__(self, parent=None, alpha = 0.15):
        super().__init__(parent)

        self.setMinimumHeight(80)
        self.setMaximumHeight(150)
        self.setMinimumWidth(300)

        self.db_min = -50.0
        self.db_max = 6.0

        self.left_level = 0.0
        self.right_level = 0.0

        self.left_peak = 0.0
        self.right_peak = 0.0

        self.left_hold = 0
        self.right_hold = 0

        self.attack = 0.65
        self.release = 0.08

        self.segment_count = 32

        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

    # ---------------------------------------------------------
    # Audio update
    # ---------------------------------------------------------

    def _rms_to_norm(self, rms):

        if rms < 1e-10:
            db = self.db_min
        else:
            db = 20.0 * np.log10(rms)

        db = max(self.db_min, min(self.db_max, db))

        return (db - self.db_min) / (self.db_max - self.db_min)

    def update_level(self, rms):

        if isinstance(rms, (tuple, list)):
            left_rms, right_rms = rms
        else:
            left_rms = rms
            right_rms = rms

        target_l = self._rms_to_norm(left_rms)
        target_r = self._rms_to_norm(right_rms)

        if target_l > self.left_level:
            self.left_level += (target_l - self.left_level) * self.attack
        else:
            self.left_level += (target_l - self.left_level) * self.release

        if target_r > self.right_level:
            self.right_level += (target_r - self.right_level) * self.attack
        else:
            self.right_level += (target_r - self.right_level) * self.release

        # Peak hold LEFT

        if self.left_level > self.left_peak:
            self.left_peak = self.left_level
            self.left_hold = 25
        elif self.left_hold > 0:
            self.left_hold -= 1
        else:
            self.left_peak *= 0.995

        # Peak hold RIGHT

        if self.right_level > self.right_peak:
            self.right_peak = self.right_level
            self.right_hold = 25
        elif self.right_hold > 0:
            self.right_hold -= 1
        else:
            self.right_peak *= 0.995

        self.update()

    # ---------------------------------------------------------
    # Colours
    # ---------------------------------------------------------

    def segment_colour(self, position):

        db = self.db_min + position * (self.db_max - self.db_min)

        if db < -3:
            return QColor(140, 255, 40)

        elif db < 0:
            return QColor(255, 220, 0)

        else:
            return QColor(255, 40, 40)

    # ---------------------------------------------------------
    # Draw channel
    # ---------------------------------------------------------

    def draw_channel(self, painter, rect, label, level, peak):
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(rect.left(), rect.center().y() + 5, label)

        meter_left = rect.left() + 22
        meter_width = rect.width() - 28
        gap = 2

        seg_width = (
            meter_width -
            gap * (self.segment_count - 1)
        ) / self.segment_count

        seg_height = rect.height() - 8
        lit = int(level * self.segment_count)

        for i in range(self.segment_count):
            x = meter_left + i * (seg_width + gap)
            seg_rect = QRect(int(x), rect.top() + 4, int(seg_width), seg_height)

            pos = i / self.segment_count
            if i < lit:
                painter.fillRect(seg_rect, self.segment_colour(pos))
            else:
                painter.fillRect(seg_rect, QColor(35, 20, 20))

        # peak hold
        peak_x = meter_left + peak * meter_width
        painter.setPen(QPen(QColor(255, 255, 255), 2))

        painter.drawLine(int(peak_x), rect.top() + 2, int(peak_x), rect.bottom() - 2)

    # ---------------------------------------------------------
    # Paint
    # ---------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        r = self.rect()
        p.fillRect(r, QColor(0, 0, 0))

        # -------------------------------------------------
        # Layout
        # -------------------------------------------------

        scale_h = 22

        top_ticks_h = 8
        bottom_ticks_h = 8
        scale_h = 30

        available = (r.height() - top_ticks_h - bottom_ticks_h - scale_h)
        meter_h = available // 2

        left_rect = QRect(10, top_ticks_h, r.width() - 20, meter_h)
        right_rect = QRect(10, top_ticks_h + meter_h, r.width() - 20, meter_h)

        self.draw_channel(p, left_rect, "L", self.left_level, self.left_peak)
        self.draw_channel(p, right_rect, "R", self.right_level, self.right_peak)

        # -------------------------------------------------
        # Scale
        # -------------------------------------------------

        scale_values = [-50, -40, -30, -20, -10, 0, 3, 6]
        self.meter_left = 32
        
        self.meter_right_margin = 20
        meter_width = (self.width() - self.meter_left - self.meter_right_margin)

        p.setPen(QColor(180, 180, 180))

        font = QFont()
        font.setPointSize(8)
        p.setFont(font)

        # Move the scale up by using a fixed offset from the bottom
        y_offset_from_bottom = 20
        y = r.height() - y_offset_from_bottom
        
        for db in scale_values:
            norm = ((db - self.db_min) / (self.db_max - self.db_min))
            x = self.meter_left + norm * meter_width
            
            # Draw tick mark (pointing down from the scale position)
            p.drawLine(int(x), y - 8, int(x), y - 2)  # Shorter tick mark
            
            # Draw text directly below the tick mark with no gap
            text = f"+{db}" if db > 0 else str(db)
            # Position text right below the tick mark
            p.drawText(int(x) - 12, y, 30, 12, Qt.AlignCenter, text)

        # -------------------------------------------------
        # Top tick marks
        # -------------------------------------------------

        p.setPen(QColor(90, 90, 90))
        tick_count = 80
        for i in range(tick_count):
            x = self.meter_left + i * meter_width / tick_count
            p.drawLine(int(x), 2, int(x), 6)
            p.drawLine(int(x), r.height() - scale_h - 4, int(x), r.height() - scale_h)

# ----------------------------------------------------------------------
# Helper: Detect available Whisper models
# ----------------------------------------------------------------------

def get_whisper_model_path(model_name):
    path = Path.home() / "whisper.cpp" / "models" / f"ggml-{model_name}.bin"
    return str(path) if path.exists() else None

def get_whisper_models_info():
    """
    Returns a list of dicts for each available model.
    Each dict: name, downloaded (bool), disk_size (str), mem_usage (str).
    """
    # Hardcoded model sizes (disk & memory)
    MODEL_DATA = {
        "tiny":           {"disk": "75 MiB",  "mem": "~273 MB", "lan": "any",     "speed": "⚡⚡⚡⚡⚡", "accuracy": "⭐⭐",      "usage": "Quick tests"},
        "tiny.en":        {"disk": "75 MiB",  "mem": "~273 MB", "lan": "english", "speed": "⚡⚡⚡⚡⚡", "accuracy": "⭐⭐",      "usage": "Quick tests"},
        "base":           {"disk": "142 MiB", "mem": "~388 MB", "lan": "any",     "speed": "⚡⚡⚡⚡",  "accuracy": "⭐⭐⭐",     "usage": "Simple transcriptions"},
        "base.en":        {"disk": "142 MiB", "mem": "~388 MB", "lan": "english", "speed": "⚡⚡⚡⚡",  "accuracy": "⭐⭐⭐",     "usage": "Simple transcriptions"},
        "small":          {"disk": "466 MiB", "mem": "~852 MB", "lan": "any",     "speed": "⚡⚡⚡",   "accuracy": "⭐⭐⭐⭐",    "usage": "Balanced"},
        "small.en":       {"disk": "466 MiB", "mem": "~852 MB", "lan": "english", "speed": "⚡⚡⚡",   "accuracy": "⭐⭐⭐⭐",    "usage": "Balanced"},
        "medium":         {"disk": "1.5 GiB", "mem": "~2.1 GB", "lan": "any",     "speed": "⚡⚡",    "accuracy": "⭐⭐⭐⭐⭐",  "usage": "Recommended"},
        "medium.en":      {"disk": "1.5 GiB", "mem": "~2.1 GB", "lan": "english", "speed": "⚡⚡",    "accuracy": "⭐⭐⭐⭐⭐",  "usage": "Recommended"},
        "large-v3":       {"disk": "2.9 GiB", "mem": "~3.9 GB", "lan": "any",     "speed": "⚡",     "accuracy": "⭐⭐⭐⭐⭐",  "usage": "Maximum accuracy"},
        "large-v3-turbo": {"disk": "1.5 GiB", "mem": "~2.1 GB", "lan": "any",     "speed": "⚡",     "accuracy": "⭐⭐⭐⭐⭐",  "usage": "Maximum accuracy"},
    }

    models_dir = Path.home() / "whisper.cpp" / "models"
    existing = set()
    if models_dir.exists():
        for f in models_dir.glob("ggml-*.bin"):
            name = f.name.replace("ggml-", "").replace(".bin", "")
            existing.add(name)

    # Build a list from the standard set first
    result = []
    for name in MODEL_DATA:
        result.append({
            "name": name,
            "downloaded": name in existing,
            "disk_size": MODEL_DATA[name]["disk"],
            "mem_usage": MODEL_DATA[name]["mem"],
            'language': MODEL_DATA[name]["lan"],
            'speed': MODEL_DATA[name]["speed"],
            'accuracy': MODEL_DATA[name]["accuracy"],
            'usage': MODEL_DATA[name]["usage"],
        })

    # Add any extra models found in the folder (not in the standard list)
    extra = existing - set(MODEL_DATA.keys())
    for name in sorted(extra):
        result.append({
            "name": name,
            "downloaded": True,
            "disk_size": "?",
            "mem_usage": "?",
            "language": "?",
            'speed': "?",
            'accuracy': "?",
            'usage': "?",
        })
    return result

def download_whisper_model(model_name):
    """
    Downloads the ggml model using the whisper.cpp download script.
    Returns True on success, False on failure.
    """
    whisper_cpp_dir = Path.home() / "whisper.cpp"
    script = whisper_cpp_dir / "models" / "download-ggml-model.sh"
    if not script.exists():
        return False
    try:
        subprocess.run(
            ["bash", str(script), model_name],
            cwd=whisper_cpp_dir / "models",
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

# ----------------------------------------------------------------------
# Helper: Detect LLM backends
# ----------------------------------------------------------------------

def detect_backends():
    backends = {}
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            backends["Ollama"] = {"url": "http://localhost:11434", "api_type": "ollama", "models": models}
    except:
        pass

    for name, url in [
        ("vLLM", "http://localhost:8000"),
        ("LM Studio", "http://localhost:1234"),
        ("llama.cpp", "http://localhost:8080")
    ]:
        try:
            r = requests.get(f"{url}/v1/models", timeout=1.5)
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])]
                backends[name] = {"url": url, "api_type": "openai", "models": models}
        except:
            pass
    return backends

# ----------------------------------------------------------------------
# Job, ProcessingWorker, and QueueWorker for asynchronous job processing
# ----------------------------------------------------------------------

class Job:
    """A single transcription job."""
    _counter = 0

    def __init__(self, audio_file_path, whisper_model, backend_info, llm_model, use_whisper_cli, output_dir=None):
        Job._counter += 1
        self.id = Job._counter
        self.audio_file_path = audio_file_path
        self.whisper_model = whisper_model
        self.backend_info = backend_info
        self.llm_model = llm_model
        self.use_whisper_cli = use_whisper_cli
        self.output_dir = output_dir   # store the folder

class ProcessingWorker(QObject):
    """Performs transcription and summarization for a single job (no recording)."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    summary_chunk = pyqtSignal(str)
    finished = pyqtSignal(str)   # md_path
    error = pyqtSignal(str)

    def process(self, job, queue_worker):
        """Process a job: transcribe + summarize."""
        # Use provided output_dir, or create a new one
        if job.output_dir is not None:
            self.output_dir = Path(job.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = Path.cwd() / "transcripts"
            base_dir.mkdir(exist_ok=True)
            folder_name = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
            self.output_dir = base_dir / folder_name
            self.output_dir.mkdir(exist_ok=True)
        self.log.emit(f"📁 Output folder: {self.output_dir}")
        
        # Check for cancellation at various points
        if queue_worker.stop_current:
            self.log.emit("Job cancelled.")
            self.error.emit("Cancelled by user.")
            return
        try:
            base_dir = Path.cwd() / "transcripts"
            base_dir.mkdir(exist_ok=True)
            folder_name = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
            self.output_dir = base_dir / folder_name
            self.output_dir.mkdir(exist_ok=True)
            self.log.emit(f"📁 Output folder: {self.output_dir}")

            audio_file = job.audio_file_path
            if not os.path.exists(audio_file):
                self.error.emit(f"Audio file not found: {audio_file}")
                return

            self.log.emit(f"📂 Processing audio file: {audio_file}")
            self.progress.emit(10)

            self.log.emit("📝 Transcribing with Whisper...")
            transcript = self._transcribe(audio_file, job.whisper_model, job.use_whisper_cli, queue_worker)
            if transcript is None:
                self.error.emit("Transcription failed.")
                return
            self.log.emit("✅ Transcription complete.")
            self.progress.emit(60)

            self.log.emit("🤖 Summarizing with LLM... (streaming response below)")
            self.progress.emit(70)
            # FIX: pass whisper_model as the 5th argument
            summary, md_path = self._summarize(transcript, audio_file, job.backend_info, job.llm_model, job.whisper_model)
            if summary is None:
                self.error.emit("Summarization failed.")
                return
            self.log.emit("✅ Summary generated.")
            self.progress.emit(100)
            self.finished.emit(md_path)
        except Exception as e:
            self.error.emit(str(e))

    # ------------------------------------------------------------------
    # Transcribe helpers (copied from Worker with minor adjustments)
    # ------------------------------------------------------------------
    def _transcribe(self, audio_file, whisper_model, use_cli, queue_worker=None):
        if use_cli:
            return self._transcribe_cli(audio_file, whisper_model, queue_worker)
        else:
            return self._transcribe_faster(audio_file, whisper_model, queue_worker)

    def _transcribe_faster(self, audio_file, whisper_model, queue_worker=None):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.log.emit("faster-whisper not installed. Falling back to whisper-cli.")
            return self._transcribe_cli(audio_file, whisper_model, queue_worker)

        try:
            self.log.emit(f"⏳ Loading Whisper model '{whisper_model}' on CPU with int8 quantization...")
            self.log.emit("   (This may take several minutes for large models like large-v3)")
            model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
            self.log.emit("✅ Model loaded successfully.")
        except Exception as e:
            self.log.emit(f"❌ Failed to load Whisper model: {e}")
            return None

        try:
            self.log.emit(f"🎤 Starting transcription of '{audio_file}'...")
            segments, info = model.transcribe(audio_file, beam_size=5)
            self.log.emit(f"📊 Language: {info.language}, probability: {info.language_probability:.2f}")

            transcript_parts = []
            segment_count = 0
            for seg in segments:
                # Check cancellation
                if queue_worker is not None and getattr(queue_worker, 'stop_current', False):
                    self.log.emit("🛑 Cancelled during transcription.")
                    return None

                segment_count += 1
                text = seg.text.strip()
                if text:
                    # Emit each segment to the log for live feedback
                    self.log.emit(f"[{segment_count}] {text}")
                transcript_parts.append(seg.text)

            transcript = " ".join(transcript_parts)
            self.log.emit(f"✅ Transcription complete. {segment_count} segments processed.")

        except Exception as e:
            self.log.emit(f"❌ Transcription error: {e}")
            return None

        transcript_file = self.output_dir / "transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        return transcript

    def _transcribe_cli(self, audio_file, whisper_model, queue_worker=None):
        model_path = get_whisper_model_path(whisper_model)
        if not model_path:
            self.log.emit(f"Model '{whisper_model}' not found. Attempting to download...")
            if not download_whisper_model(whisper_model):
                self.log.emit(f"Failed to download model '{whisper_model}'. Please download manually.")
                return None
            model_path = get_whisper_model_path(whisper_model)
            if not model_path:
                self.log.emit(f"Model still not found after download.")
                return None
            self.log.emit(f"Download complete.")

        # Check cancellation after download (if queue_worker provided)
        if queue_worker is not None and getattr(queue_worker, 'stop_current', False):
            self.log.emit("Cancelled before transcription.")
            return None

        whisper_cli = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
        if not whisper_cli.exists():
            self.log.emit("whisper-cli not found. Please build whisper.cpp or install faster-whisper.")
            return None

        output_base = str(self.output_dir / "meeting")
        cmd = [str(whisper_cli), "-m", model_path, "-f", audio_file, "-otxt", "-of", output_base]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            transcript_file = self.output_dir / "meeting.txt"
            with open(transcript_file, 'r') as f:
                return f.read()
        except subprocess.TimeoutExpired:
            self.log.emit("❌ whisper-cli timed out after 5 minutes.")
            return None
        except subprocess.CalledProcessError as e:
            self.log.emit(f"❌ whisper-cli error: {e.stderr}")
            return None

    # ------------------------------------------------------------------
    # Summarize helpers – FIXED: add whisper_model parameter
    # ------------------------------------------------------------------
    def _summarize(self, transcript, audio_file, backend_info, llm_model, whisper_model):
        prompt = (
            "Summarize the following meeting transcript. "
            "Provide a concise summary with key points, decisions, and action items.\n\n"
            f"Transcript:\n{transcript}"
        )
        backend_url = backend_info["url"]
        api_type = backend_info["api_type"]
        model = llm_model

        if api_type == "ollama":
            payload = {"model": model, "prompt": prompt, "stream": True}
            try:
                with requests.post(f"{backend_url}/api/generate", json=payload, stream=True) as r:
                    if r.status_code != 200:
                        self.log.emit(f"Ollama error: {r.status_code} {r.text}")
                        return None, None
                    summary = ""
                    for line in r.iter_lines():
                        if line:
                            data = json.loads(line.decode())
                            if "response" in data:
                                chunk = data["response"]
                                summary += chunk
                                self.summary_chunk.emit(chunk)
                    # Pass whisper_model to _save_markdown
                    md_path = self._save_markdown(audio_file, transcript, summary, backend_info, model, whisper_model)
                    return summary, md_path
            except Exception as e:
                self.log.emit(f"Ollama error: {e}")
                return None, None

        elif api_type == "openai":
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
            headers = {"Content-Type": "application/json"}
            try:
                with requests.post(f"{backend_url}/v1/chat/completions", json=payload, headers=headers, stream=True) as r:
                    if r.status_code != 200:
                        self.log.emit(f"API error: {r.status_code} {r.text}")
                        return None, None
                    summary = ""
                    for line in r.iter_lines():
                        if line:
                            line_str = line.decode()
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                            else:
                                data_str = line_str
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    summary += delta
                                    self.summary_chunk.emit(delta)
                            except json.JSONDecodeError:
                                pass
                    # Pass whisper_model to _save_markdown
                    md_path = self._save_markdown(audio_file, transcript, summary, backend_info, model, whisper_model)
                    return summary, md_path
            except Exception as e:
                self.log.emit(f"OpenAI error: {e}")
                return None, None
        else:
            self.log.emit(f"Unknown API type: {api_type}")
            return None, None

    def _save_markdown(self, audio_file, transcript, summary, backend_info, model, whisper_model):
        md_file = self.output_dir / "summary.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# Meeting Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Audio File:** {audio_file}\n\n")
            f.write(f"**Whisper Model:** {whisper_model}\n\n")
            f.write(f"**LLM Backend:** {backend_info.get('name', 'Unknown')}\n")
            f.write(f"**LLM Model:** {model}\n\n")
            f.write("---\n\n## Summary\n\n")
            f.write(summary)
            f.write("\n\n---\n\n## Full Transcript\n\n")
            f.write("```\n")
            f.write(transcript)
            f.write("\n```\n\n")
            f.write("---\n*Generated by Meeting Transcriber*")
        return str(md_file)

class QueueWorker(QObject):
    """Manages a queue of jobs and processes them sequentially in its own thread."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    summary_chunk = pyqtSignal(str)
    job_finished = pyqtSignal(str)   # md_path
    job_error = pyqtSignal(str)
    job_started = pyqtSignal(int)    # job id

    def __init__(self):
        super().__init__()
        self.queue = deque()
        self.processing = False
        self.stop_requested = False
        self.stop_current = False
        self.current_job = None

    def stop_current_job(self):
        self.stop_current = True
        if self.current_job is not None:
            # We'll let the processing worker check this flag
            pass

    def add_job(self, job):
        self.queue.append(job)

    def stop(self):
        self.stop_requested = True

    def run(self):
        while not self.stop_requested:
            if self.queue and not self.processing:
                job = self.queue.popleft()
                self.current_job = job
                self.processing = True
                self.stop_current = False   # reset for new job
                self.job_started.emit(job.id)

                # Process this job
                processor = ProcessingWorker()
                processor.stop_flag = self 
                # Connect processor signals to our own signals
                processor.log.connect(self.log)
                processor.progress.connect(self.progress)
                processor.summary_chunk.connect(self.summary_chunk)
                processor.finished.connect(self.job_finished)
                processor.error.connect(self.job_error)

                try:
                    processor.process(job, self)
                except Exception as e:
                    self.job_error.emit(str(e))
                finally:
                    # Clean up connections to avoid duplicates
                    processor.log.disconnect()
                    processor.progress.disconnect()
                    processor.summary_chunk.disconnect()
                    processor.finished.disconnect()
                    processor.error.disconnect()
                    processor.deleteLater()
                    # disconnect signals...
                    self.processing = False
                    self.current_job = None

                self.processing = False
                self.current_job = None
            else:
                QThread.msleep(100)   # avoid busy-waiting

# ----------------------------------------------------------------------
# Worker to download Whisper models on locally present
# ----------------------------------------------------------------------

class DownloadWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)  # success, message
    log = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    @pyqtSlot()
    def run(self):
        self.log.emit(f"Downloading model '{self.model_name}'...")
        success = download_whisper_model(self.model_name)
        if success:
            self.log.emit("Download completed.")
        else:
            self.log.emit("Download failed.")
        self.finished.emit(success, self.model_name)

# ----------------------------------------------------------------------
# Main GUI Window
# ----------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Transcriber")
        self.setMinimumSize(700, 700)

        self.button_height = 30  # pixels
        self.button_radius = 8   # pixels

        # Recording related attributes
        self.recording_buffer = deque(maxlen=16000)   # for live visualisation
        self.recording_chunks = []
        self.is_recording = False
        self.stream = None

        self.loaded_samples = None
        self.loaded_sr = None
        self.playhead = 0          # current sample index
        self.playback_timer = None
        self.update_timer = None

        # Set application icon (for the window and taskbar)
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Build the UI first – this creates log_text, progress_bar, etc.
        self.setup_ui()
        self.refresh_devices()
        self.refresh_whisper_models()
        self.refresh_backends()

        # Create the queue worker and thread, and connect signals
        self.queue_worker = QueueWorker()
        self.queue_thread = QThread()
        self.queue_worker.moveToThread(self.queue_thread)
        self.queue_thread.started.connect(self.queue_worker.run)
        self.queue_thread.start()

        # Connect queue worker signals to GUI slots (widgets now exist)
        self.queue_worker.log.connect(self.log_text.append)
        self.queue_worker.progress.connect(self.progress_bar.setValue)
        self.queue_worker.summary_chunk.connect(self.append_summary)
        self.queue_worker.job_finished.connect(self.on_job_finished)
        self.queue_worker.job_error.connect(self.on_error)
        # Optionally connect job_started to show which job is running
        self.queue_worker.job_started.connect(self.on_job_started)

        # Now we can connect signals after UI creation
        self.whisper_combo.currentIndexChanged.connect(self.on_whisper_model_changed)

        # Job counter for locking configuration
        self.job_count = 0
        # Update config lock state initially
        self.update_config_lock()

    @pyqtSlot(np.ndarray, int)
    def on_audio_loaded(self, samples, sr):
        self.loaded_samples = samples
        self.loaded_sr = sr
        self.playhead = 0

        # If an update timer already runs (from recording), stop it
        if self.update_timer:
            self.update_timer.stop()
            self.update_timer = None

        # Start a new timer that drives the visualisation
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_playback_visualisation)
        self.update_timer.start(33)  # ~30 fps

    def update_playback_visualisation(self):
        if self.loaded_samples is None or self.loaded_sr is None:
            return

        # Advance playhead by (sample_rate / fps) samples per frame
        dt = 1.0 / 30.0          # frame duration
        samples_per_frame = int(self.loaded_sr * dt)
        self.playhead += samples_per_frame

        total_samples = len(self.loaded_samples)
        if self.playhead >= total_samples:
            # End of file – stop timer and clear visualisation
            self.update_timer.stop()
            self.update_timer = None
            self.waveform.update_buffer([])
            self.vumeter.update_level(0)
            return

        # Get a window of the last, say, 16000 samples (1 sec at 16 kHz)
        # We'll take a slice ending at playhead (or centred, but this gives scrolling effect)
        window_size = 16000
        start = max(0, self.playhead - window_size)
        end = min(total_samples, self.playhead)
        window = self.loaded_samples[start:end]

        if len(window) > 0:
            # Update waveform display (it expects a deque or list)
            self.waveform.update_buffer(window.tolist())

            # Compute RMS for VU meter (use the current frame, not the whole window)
            frame_start = max(0, self.playhead - samples_per_frame)
            frame_end = self.playhead
            frame = self.loaded_samples[frame_start:frame_end]
            if len(frame) > 0:
                rms = np.sqrt(np.mean(frame**2))
                self.vumeter.update_level(rms)
            else:
                self.vumeter.update_level(0)
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Default styles with background color
        action_btn_style = (f"height: {self.button_height}px; border-radius: {self.button_radius}px; padding: 8px; border: 1px solid #cccccc;")
        refresh_btn_style = (f"border-radius: {self.button_radius}px; padding: 8px; border: 1px solid #cccccc;")

        # ----- Audio device -----
        self.dev_group = QGroupBox("Audio Input")
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Select your microphone or input device")
        dev_layout.addWidget(self.device_combo)
        refresh_dev_btn = QPushButton("Refresh")
        refresh_dev_btn.clicked.connect(self.refresh_devices)
        #refresh_dev_btn.setStyleSheet(refresh_btn_style)
        dev_layout.addWidget(refresh_dev_btn)
        self.dev_group.setLayout(dev_layout)
        layout.addWidget(self.dev_group)

        # ----- Whisper model -----
        self.whisper_group = QGroupBox("Whisper Model")
        whisper_layout = QHBoxLayout()
        whisper_layout.addWidget(QLabel("Model:"))
        self.whisper_combo = QComboBox()
        self.whisper_combo.setToolTip("Select a Whisper model size")
        whisper_layout.addWidget(self.whisper_combo)
        self.use_cli_check = QCheckBox("Use whisper-cli (requires built binary)")
        self.use_cli_check.setToolTip("If unchecked, uses faster-whisper (Python)")
        whisper_layout.addWidget(self.use_cli_check)
        refresh_whisper_btn = QPushButton("Refresh")
        refresh_whisper_btn.clicked.connect(self.refresh_whisper_models)
        #refresh_whisper_btn.setStyleSheet(refresh_btn_style)
        whisper_layout.addWidget(refresh_whisper_btn)
        self.whisper_group.setLayout(whisper_layout)
        layout.addWidget(self.whisper_group)

        # ----- LLM Backend and Model -----
        self.llm_group = QGroupBox("LLM Backend")
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
        refresh_backend_btn = QPushButton("Refresh")
        refresh_backend_btn.clicked.connect(self.refresh_backends)
        #refresh_backend_btn.setStyleSheet(refresh_btn_style)
        llm_layout.addWidget(refresh_backend_btn)
        self.llm_group.setLayout(llm_layout)
        layout.addWidget(self.llm_group)

        # ----- Audio Visualization (Waveform + VU) -----
        vis_group = QGroupBox("Audio Monitor")
        vis_main_layout = QVBoxLayout()          # overall vertical

        # Style selector (top row)
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("VU Style:"))
        self.vu_style_combo = QComboBox()
        self.vu_style_combo.addItems(
            [
                "Basic VU-meter", 
                "Retro LED (Vertical)",
                "Retro LED (Horizontal)",
                "Modern VU-meter",
                "Analog VU-meter", 
                "Classic VU-meter (Sifam)", 
                "Glass VU-meter",
                "Liquid Glass",
                "Neon Retro",
                "Tube Amplifier",
                "BBC PPM",
                "Nordic VU",
                "LED Matrix Meter",
                "Broadcast Stereo VU-meter" 
            ]
        )
        self.vu_style_combo.currentIndexChanged.connect(self.switch_vu_style)
        style_layout.addWidget(self.vu_style_combo)
        style_layout.addStretch()
        vis_main_layout.addLayout(style_layout)

        # Main horizontal area: waveform (left) + VU (right)
        h_layout = QHBoxLayout()
        self.waveform = WaveformDisplay()
        h_layout.addWidget(self.waveform, stretch=4)   # waveform gets more space

        # VU container (holds the actual VU meter)
        self.vu_container = QWidget()
        self.vu_container_layout = QVBoxLayout(self.vu_container)
        self.vu_container_layout.setContentsMargins(0, 0, 0, 0)
        # Create the default VU meter (Basic)
        self.vumeter = BasicVUMeter()
        self.vumeter.setAutoFillBackground(True)
        self.vu_style_combo.setCurrentIndex(4)          # set default VU-meter at launch
        pal = self.vumeter.palette()
        pal.setColor(self.vumeter.backgroundRole(), QColor(30, 30, 30))
        self.vumeter.setPalette(pal)
        self.vumeter.setAttribute(Qt.WA_TranslucentBackground, False)
        self.vumeter.setStyleSheet("")
        self.vumeter.setMinimumWidth(60)
        self.vumeter.setMaximumWidth(16777215)
        self.vumeter.setMinimumHeight(80)
        self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vu_container_layout.addWidget(self.vumeter)

        # Add the container to the horizontal layout with a fixed width
        h_layout.addWidget(self.vu_container, stretch=1)   # VU gets less stretch

        vis_main_layout.addLayout(h_layout)
        vis_group.setLayout(vis_main_layout)
        layout.addWidget(vis_group)

        # ----- Record/Stop button + progress -----
        control_layout = QHBoxLayout()
        self.record_btn = QPushButton("🎤 Record")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet(f"font-size: 14px; {action_btn_style}")
        control_layout.addWidget(self.record_btn)

        self.load_btn = QPushButton("📂 Load Audio")
        self.load_btn.clicked.connect(self.load_audio_file)
        self.load_btn.setStyleSheet(action_btn_style)
        control_layout.addWidget(self.load_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        # ------ Cancel button --------
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self.cancel_current_job)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(action_btn_style)
        control_layout.addWidget(self.cancel_btn)

        # ----- Clear Log button ------
        self.clear_log_btn = QPushButton("🗑️ Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.clear_log_btn.setStyleSheet(action_btn_style)
        control_layout.addWidget(self.clear_log_btn)

        # Load layout (at last)
        layout.addLayout(control_layout)

        # ----- Log / Output area -----
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("monospace")
        layout.addWidget(QLabel("Log / Summary Output:"))
        layout.addWidget(self.log_text)

        # ----- Save / Open buttons -----
        save_layout = QHBoxLayout()
        self.save_md_btn = QPushButton("💾 Save Markdown As...")
        self.save_md_btn.clicked.connect(self.save_markdown)
        self.save_md_btn.setStyleSheet(action_btn_style)
        self.open_folder_btn = QPushButton("📂 Open Output Folder")
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.open_folder_btn.setStyleSheet(action_btn_style)
        save_layout.addWidget(self.save_md_btn)
        save_layout.addWidget(self.open_folder_btn)
        layout.addLayout(save_layout)

    def closeEvent(self, event):
        if self.queue_worker:
            self.queue_worker.stop()
            self.queue_thread.quit()
            self.queue_thread.wait()
        event.accept()
    
    def switch_vu_style(self, index):
        """Replace the current VU meter with the selected style."""
        # Remove the old widget
        old = self.vumeter
        self.vu_container_layout.removeWidget(old)
        old.setParent(None)
        old.deleteLater()

        # Create the new one
        if index == 0:   # Basic VU-meter
            self.vumeter = BasicVUMeter(alpha = 0.25)
            # Make it opaque and expandable
            self.vumeter.setAutoFillBackground(True)
            pal = self.vumeter.palette()
            pal.setColor(self.vumeter.backgroundRole(), QColor(30, 30, 30))
            self.vumeter.setPalette(pal)
            self.vumeter.setAttribute(Qt.WA_TranslucentBackground, False)
            self.vumeter.setStyleSheet("")
            # Remove any width limits and set expandable size policy
            self.vumeter.setMinimumWidth(60)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 1:  # Vertical LED
            self.vumeter = RetroLEDVerticalVUMeter(alpha=0.10)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(80)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 2:  # Horizontal LED# Retro LED (horizontal)
            self.vumeter = RetroLEDHorizontalVUMeter(alpha=0.10)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(80)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 3:   # Modern VU-meter
            self.vumeter = ModernVUMeter(alpha = 0.35)
            self.vumeter.setMaximumWidth(16777215)   # remove max width
            self.vumeter.setMinimumWidth(80)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 4:   # Analog VU-meter
            self.vumeter = AnalogStyleVUMeter(alpha = 0.10)
            self.vumeter.setMaximumWidth(16777215)   # remove max width
            self.vumeter.setMinimumWidth(80)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 5:   # Classic Horizontal (Sifam)
            self.vumeter = ClassicHorizontalVUMeter(alpha = 0.15)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 6:   # Glass VU-meter
            self.vumeter = GlassVUMeter(alpha=0.10)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(80)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 7:   # Liquid Glass
            self.vumeter = LiquidGlassVUMeter(alpha=0.10)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(60)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 8:   # Neon Retro
            self.vumeter = NeonRetroVUMeter(alpha=0.10)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(80)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 9:   # Tube Amplifier
            self.vumeter = TubeAmplifierVUMeter(alpha=0.10)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(80)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 10:  # BBC PPM
            self.vumeter = ClassicBBCPPM(alpha=0.15)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(60)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 11:  # Nordic VU
            self.vumeter = NordicVUMeter(alpha=0.15)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(60)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 12:  # Rotary Dial VU-meter
            self.vumeter = LEDMatrixBarMeter(alpha=0.10, cols=16, rows=8)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(120)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        elif index == 13:  # Custom Vertical LED (13-step)
            self.vumeter = BroadcastStereoVUMeter(alpha=0.10)
            self.vumeter.setMaximumWidth(16777215)
            self.vumeter.setMinimumWidth(80)
            self.vumeter.setMinimumHeight(80)
            self.vumeter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Insert it into the container
        self.vu_container_layout.addWidget(self.vumeter)
        # Force layout update
        self.vu_container_layout.activate()
        self.vu_container.updateGeometry()
        # Ensure the widget redraws
        self.vumeter.update()
        self.vumeter.repaint()

        # Immediately refresh with current level (if any)
        if hasattr(self, 'loaded_samples') and self.loaded_samples is not None:
            # The playback timer will update the meter; no action needed
            pass
        elif self.is_recording:
            if self.recording_buffer:
                arr = np.array(list(self.recording_buffer), dtype=np.float32)
                rms = np.sqrt(np.mean(arr**2))
                self.vumeter.update_level(rms)
        else:
            self.vumeter.update_level(0)
   
    # Refresh methods
    def refresh_devices(self):
        self.device_combo.clear()
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    self.device_combo.addItem(f"{dev['name']} (index {i})", i)
        except Exception as e:
            self.log_text.append(f"Error listing audio devices: {e}")
        try:
            default = sd.default.device[0]
            for idx in range(self.device_combo.count()):
                if self.device_combo.itemData(idx) == default:
                    self.device_combo.setCurrentIndex(idx)
                    break
        except:
            pass

    def refresh_whisper_models(self):
        self.whisper_combo.clear()
        models_info = get_whisper_models_info()

        for info in models_info:
            status = "✅ Downloaded" if info["downloaded"] else "⬇️ Not downloaded"
            display = f"{info['name']} ({info['disk_size']} disk, {info['mem_usage']} mem) {status}"
            self.whisper_combo.addItem(display)
            # Store full info as item data for later use
            idx = self.whisper_combo.count() - 1
            self.whisper_combo.setItemData(idx, info, Qt.UserRole)
            # Tooltip with more details
            tooltip = (f"Model: {info['name']}\n"
                      f"Disk: {info['disk_size']}\n"
                      f"Memory: {info['mem_usage']}\n"
                      f"Language: {info['language']}\n"
                      f"Speed: {info['speed']}\n"
                      f"Accuracy: {info['accuracy']}\n"
                      f"Usage: {info['usage']}\n"
                      f"Status: {'Downloaded' if info['downloaded'] else 'Not downloaded'}")
            self.whisper_combo.setItemData(idx, tooltip, Qt.ToolTipRole)

        # --- Set default selection: "medium" if it exists, otherwise first downloaded, otherwise first item ---
        default_index = 0
        for i in range(self.whisper_combo.count()):
            info = self.whisper_combo.itemData(i, Qt.UserRole)
            if info and info["name"] == "medium":
                default_index = i
                break
        else:
            # If medium not found, pick the first downloaded model
            for i in range(self.whisper_combo.count()):
                info = self.whisper_combo.itemData(i, Qt.UserRole)
                if info and info["downloaded"]:
                    default_index = i
                    break
        self.whisper_combo.setCurrentIndex(default_index)

    def refresh_backends(self):
        self.backend_combo.clear()
        self.backends = detect_backends()
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
        models = self.backends[backend_name]["models"]
        if models:
            self.model_combo.addItems(models)
            # --- Set default selection ---
            default_index = 0
            for i, model in enumerate(models):
                if model == DEFAULT_LLM_MODEL:
                    default_index = i
                    break
            self.model_combo.setCurrentIndex(default_index)
        else:
            self.model_combo.addItem("(no models)")

    def on_whisper_model_changed(self, index):
        if index < 0:
            return
        if hasattr(self, '_downloading') and self._downloading:
            return

        info = self.whisper_combo.itemData(index, Qt.UserRole)

        # Fallback: if info is not a dict, parse model name from display text
        if not isinstance(info, dict):
            text = self.whisper_combo.currentText()
            model_name = text.split()[0] if text else "unknown"
            info = {"name": model_name, "downloaded": False, "disk_size": "?", "mem_usage": "?"}
            self.log_text.append(f"⚠️ Could not retrieve model info; using fallback for '{model_name}'.")

        if info.get("downloaded", False):
            return  # already available

        # Ask user if they want to download
        reply = QMessageBox.question(
            self,
            "Model not downloaded",
            f"The model '{info['name']}' is not present on disk.\n"
            f"Disk size: {info['disk_size']}, Memory: {info['mem_usage']}\n\n"
            "Do you want to download it now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            self.download_model(info["name"])
        else:
            # User chose not to download – revert to a downloaded model.
            # Priority: "medium" if it exists and is downloaded, else the first downloaded model.
            target_index = None

            # 1) Look for "medium"
            for i in range(self.whisper_combo.count()):
                info2 = self.whisper_combo.itemData(i, Qt.UserRole)
                if isinstance(info2, dict) and info2.get("name") == "medium" and info2.get("downloaded", False):
                    target_index = i
                    break

            # 2) If medium not found, fallback to the first downloaded model
            if target_index is None:
                for i in range(self.whisper_combo.count()):
                    info2 = self.whisper_combo.itemData(i, Qt.UserRole)
                    if isinstance(info2, dict) and info2.get("downloaded", False):
                        target_index = i
                        break

            # Set the combo to the found index (or keep the current if none – but that shouldn't happen)
            if target_index is not None:
                self.whisper_combo.setCurrentIndex(target_index)
    
    # ---------- Recording control ----------
    def toggle_recording(self):
        if self.record_btn.text() == "🎤 Record":
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.load_btn.setEnabled(False)
        self.record_btn.setText("⏹ Stop")
        self.record_btn.setStyleSheet(
            f"background-color: #ff6b6b; font-weight: bold; font-size: 14px; "
            f"height: {self.button_height}px; border-radius: {self.button_radius}px; padding: 8px; "
            "border: 1px solid #cc0000;"
        )
        self.progress_bar.setValue(0)
        self.log_text.append("🎤 Recording... (press Stop to finish)")

        # Reset playback_timer
        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()
            self.playback_timer = None
        self.loaded_samples = None

        # Initialise recording buffers
        self.recording_buffer.clear()
        self.recording_chunks = []
        self.is_recording = True
        self.dev_group.setEnabled(False)        # <-- disable only audio input

        device_index = self.device_combo.currentData()
        fs = 16000

        try:
            self.stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=fs,
                callback=self._audio_callback,
                blocksize=1024,
                latency='low',
                dtype='float32'
            )
            self.stream.start()

            # Start visualisation timer
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.update_visualization)
            self.update_timer.start(33)

        except Exception as e:
            self.log_text.append(f"❌ Recording error: {e}")
            self.reset_ui()

    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aac);;All Files (*)"
        )
        if not file_path:
            return

        # Load audio for visualisation (in main thread, quick)
        try:
            data, sr = sf.read(file_path, dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            self.loaded_samples = data
            self.loaded_sr = sr
            self.playhead = 0
            # Start a timer to scroll through the waveform
            if self.playback_timer:
                self.playback_timer.stop()
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self.update_playback_visualisation)
            self.playback_timer.start(33)
        except Exception as e:
            self.log_text.append(f"Could not load audio for visualisation: {e}")
            self.loaded_samples = None

        # Add a job for processing
        self._add_job_from_audio(file_path)
    
    def stop_recording(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        self.dev_group.setEnabled(True)         # <-- enable audio input

        if self.update_timer:
            self.update_timer.stop()
            self.update_timer = None

        if not self.recording_chunks:
            self.log_text.append("⏹ No audio recorded.")
            self.load_btn.setEnabled(True)
            self.update_config_lock()
            self.reset_ui()
            return
        
        # Reset playback_timer
        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()
            self.playback_timer = None
        self.loaded_samples = None

        full = np.concatenate(self.recording_chunks, axis=0)
        fs = 16000
        # Create a timestamped folder and save the recording
        base_dir = Path.cwd() / "transcripts"
        base_dir.mkdir(exist_ok=True)
        folder_name = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        output_dir = base_dir / folder_name
        output_dir.mkdir(exist_ok=True)
        audio_file = output_dir / "meeting.wav"
        sf.write(str(audio_file), full, fs)
        self.log_text.append(f"✅ Recorded to {audio_file}")

        # Now create a job and add to queue
        self._add_job_from_audio(str(audio_file), output_dir=output_dir)

        # Clear visualisation
        self.recording_buffer.clear()
        self.waveform.update_buffer([])
        self.vumeter.update_level(0)

        self.load_btn.setEnabled(True)
        self.update_config_lock()
        self.reset_ui()   # re-enables buttons, resets progress

    def update_visualization(self):
        """Read from the worker's audio buffer and update the widgets."""
        if self.is_recording and self.recording_buffer:
            # Get a snapshot (copy) to avoid threading issues
            try:
                # Convert to list (copy) – deque is thread-safe for iteration
                samples = list(self.recording_buffer)
                if samples:
                    # Update waveform
                    self.waveform.update_buffer(samples)
                    # Compute RMS for VU
                    arr = np.array(samples, dtype=np.float32)
                    rms = np.sqrt(np.mean(arr**2))
                    self.vumeter.update_level(rms)
                else:
                    self.waveform.update_buffer([])
                    self.vumeter.update_level(0)
            except Exception:
                pass

    def append_summary(self, chunk):
        self.log_text.insertPlainText(chunk)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    def on_error(self, msg):
        self.log_text.append(f"❌ Error: {msg}")
        self.job_count -= 1
        self.update_config_lock()
        self.reset_ui()

    def reset_ui(self):
        self.record_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.record_btn.setText("🎤 Record")
        action_btn_style = (
            f"height: {self.button_height}px; border-radius: {self.button_radius}px; "
            "padding: 8px; border: 1px solid #cccccc;"
        )
        self.record_btn.setStyleSheet(f"font-size: 14px; {action_btn_style}")
        self.load_btn.setStyleSheet(action_btn_style)

        self.progress_bar.setValue(0)

        self.clear_loaded_audio_visualization()
        self.cancel_btn.setEnabled(False)

        # Clear visualisation if not recording
        if not self.is_recording:
            self.waveform.update_buffer([])
            self.vumeter.update_level(0)
            if self.update_timer:
                self.update_timer.stop()
                self.update_timer = None

    def _audio_callback(self, indata, frames, time, status):
        if status:
            self.log_text.append(f"⚠️ Audio callback status: {status}")
        self.recording_chunks.append(indata.copy())
        self.recording_buffer.extend(indata.flatten())

    # ---------- Save / Open ----------
    def save_markdown(self):
        if not self.last_md_path or not os.path.exists(self.last_md_path):
            QMessageBox.information(self, "No file", "No Markdown file has been generated yet.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Markdown As", self.last_md_path, "Markdown files (*.md)"
        )
        if save_path:
            import shutil
            shutil.copy2(self.last_md_path, save_path)
            self.log_text.append(f"📁 Saved copy to: {save_path}")

    def open_folder(self):
        if self.last_md_path and os.path.exists(self.last_md_path):
            folder = os.path.dirname(self.last_md_path)
        else:
            transcripts_dir = Path.cwd() / "transcripts"
            folder = str(transcripts_dir) if transcripts_dir.exists() else os.getcwd()
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # ----------- Clear logs ----------
    def clear_log(self):
            self.log_text.clear()

    # --------- Jobs helpers ----------
    def _add_job_from_audio(self, audio_file_path, output_dir=None):
        """Create a job from the given audio file and add to queue."""
        current_idx = self.whisper_combo.currentIndex()
        item_data = self.whisper_combo.itemData(current_idx, Qt.UserRole)
        whisper_model = item_data["name"] if item_data and "name" in item_data else self.whisper_combo.currentText().split()[0]

        backend_name = self.backend_combo.currentText()
        if backend_name not in self.backends:
            QMessageBox.warning(self, "Backend Error", "No valid LLM backend selected.")
            return
        backend_info = self.backends[backend_name].copy()
        backend_info["name"] = backend_name
        llm_model = self.model_combo.currentText()
        if llm_model == "(no models)" or not llm_model:
            QMessageBox.warning(self, "Model Error", "No LLM model selected.")
            return

        use_cli = self.use_cli_check.isChecked()

        job = Job(audio_file_path, whisper_model, backend_info, llm_model, use_cli, output_dir=output_dir)
        self.queue_worker.add_job(job)
        self.job_count += 1
        self.update_config_lock()
        self.log_text.append(f"📥 Job #{job.id} added to queue.")
        self.cancel_btn.setEnabled(True)   # allow cancellation while waiting in queue

    def on_job_finished(self, md_path):
        self.job_count -= 1
        self.update_config_lock()

        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()
        self.waveform.update_buffer([])
        self.vumeter.update_level(0)
        self.loaded_samples = None   # optional

        self.clear_loaded_audio_visualization()

        self.last_md_path = md_path
        self.log_text.append(f"\n✅ Markdown saved to: {md_path}")
        self.log_text.append(f"📁 All files in: {os.path.dirname(md_path)}")
        # Optionally, set progress to 100
        self.progress_bar.setValue(100)

        self.cancel_btn.setEnabled(False)

    def on_job_started(self, job_id):
        self.log_text.append(f"🔄 Processing job #{job_id}...")
        self.cancel_btn.setEnabled(True)

    def cancel_current_job(self):
        if self.queue_worker:
            self.queue_worker.stop_current_job()
            self.log_text.append("⏹ Cancelling current job...")
            self.cancel_btn.setEnabled(False)

    # ---- Download Whisper Models ----
    def download_model(self, model_name):
      # Prevent multiple downloads
      if hasattr(self, '_downloading') and self._downloading:
          return
      self._downloading = True
      self.update_config_lock()   # lock config during download

      # Disable UI elements
      self.record_btn.setEnabled(False)
      self.load_btn.setEnabled(False)
      self.progress_bar.setRange(0, 0)  # indeterminate
      self.progress_bar.setValue(0)

      # Create thread and worker
      self.download_thread = QThread()
      self.download_worker = DownloadWorker(model_name)
      self.download_worker.moveToThread(self.download_thread)

      self.download_thread.started.connect(self.download_worker.run)
      self.download_worker.log.connect(self.log_text.append)
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

      # Refresh model list to update status
      self.refresh_whisper_models()

      if success:
          self.log_text.append(f"✅ Model '{model_name}' is now available.")
          # Optionally re-select the downloaded model
          for i in range(self.whisper_combo.count()):
              info = self.whisper_combo.itemData(i, Qt.UserRole)
              if info and info["name"] == model_name:
                  self.whisper_combo.setCurrentIndex(i)
                  break
      else:
          self.log_text.append(f"❌ Download of '{model_name}' failed. Please download manually.")
          # Revert to a downloaded model
          for i in range(self.whisper_combo.count()):
              info = self.whisper_combo.itemData(i, Qt.UserRole)
              if info and info["downloaded"]:
                  self.whisper_combo.setCurrentIndex(i)
                  break

    # ---------- Cancel Click ---------
    def clear_loaded_audio_visualization(self):
        """Stop the playback timer and clear the waveform/VU meter."""
        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()
            self.playback_timer = None
        self.waveform.update_buffer([])
        self.vumeter.update_level(0)
        self.loaded_samples = None
        self.loaded_sr = None
        self.playhead = 0

    # ---------- Lock configuration ----------
    def update_config_lock(self):
        """Enable or disable configuration controls based on activity."""
        locked = self.is_recording or self.job_count > 0 or getattr(self, '_downloading', False)
        enabled = not locked

        self.dev_group.setEnabled(enabled)
        self.whisper_group.setEnabled(enabled)
        self.llm_group.setEnabled(enabled)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.svg"))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())