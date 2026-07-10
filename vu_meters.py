#!/usr/bin/env python3
"""
vu_meters.py
------------
All audio visualisation widgets: the scrolling waveform display and every
VU-meter style (Basic, Retro LED, Modern, Analog, Glass, Neon, Tube,
BBC PPM, Nordic, LED Matrix, Broadcast Stereo...).

This file is a straight relocation of the widget classes that used to live
inline in transcribe.py -- no behavioural changes. Each class still exposes
the same two-method contract the rest of the app relies on:

    widget.update_level(rms: float)   # rms in 0..1, called every ~33ms
    widget.update_buffer(samples)     # WaveformDisplay only

Because every one of these classes independently duplicated the same
"set min/max width, size policy, translucent background" setup code in
MainWindow.switch_vu_style(), this file also adds a small VU_METER_STYLES
registry + create_vu_meter() factory at the bottom. main.py uses that
factory instead of a 90-line if/elif chain -- same visual result, far
less duplication, and adding a 16th meter style is now a 2-line change
instead of editing MainWindow.
"""

from collections import deque

import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QLinearGradient, QConicalGradient, QBrush,
    QPainterPath, QFont, QRadialGradient
)

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
# Style registry + factory
# ----------------------------------------------------------------------
# Each entry: (display name, widget class, default alpha, min_width)
# alpha = smoothing factor passed to the constructor (matches the values
# that used to be hardcoded per-branch in MainWindow.switch_vu_style()).
VU_METER_STYLES = [
    ("Basic VU-meter",              BasicVUMeter,             0.25, 60),
    ("Retro LED (Vertical)",        RetroLEDVerticalVUMeter,  0.10, 80),
    ("Retro LED (Horizontal)",      RetroLEDHorizontalVUMeter, 0.10, 80),
    ("Modern VU-meter",             ModernVUMeter,            0.35, 80),
    ("Analog VU-meter",             AnalogStyleVUMeter,       0.10, 80),
    ("Classic VU-meter (Sifam)",    ClassicHorizontalVUMeter, 0.15, None),
    ("Glass VU-meter",              GlassVUMeter,             0.10, 80),
    ("Liquid Glass",                LiquidGlassVUMeter,       0.10, 60),
    ("Neon Retro",                  NeonRetroVUMeter,         0.10, 80),
    ("Tube Amplifier",              TubeAmplifierVUMeter,     0.10, 80),
    ("BBC PPM",                     ClassicBBCPPM,            0.15, 60),
    ("Nordic VU",                   NordicVUMeter,            0.15, 60),
    ("LED Matrix Meter",            LEDMatrixBarMeter,        0.10, 120),
    ("Broadcast Stereo VU-meter",   BroadcastStereoVUMeter,   0.10, 80),
]


def vu_meter_style_names():
    """Display names for the VU-style QComboBox, in registry order."""
    return [name for name, *_ in VU_METER_STYLES]


def create_vu_meter(index, parent=None):
    """
    Build a fresh VU-meter widget for VU_METER_STYLES[index] with the
    same sizing/background setup every style needs. Returns a ready-to-use
    QWidget you can drop straight into a layout.
    """
    if not (0 <= index < len(VU_METER_STYLES)):
        index = 0
    _, widget_cls, alpha, min_width = VU_METER_STYLES[index]

    if widget_cls is BasicVUMeter:
        # BasicVUMeter is opaque (it doesn't paint a translucent bg itself),
        # so it needs the palette override the original code gave it.
        w = widget_cls(parent, alpha=alpha)
        w.setAutoFillBackground(True)
        pal = w.palette()
        pal.setColor(w.backgroundRole(), QColor(30, 30, 30))
        w.setPalette(pal)
        w.setAttribute(Qt.WA_TranslucentBackground, False)
        w.setStyleSheet("")
    else:
        w = widget_cls(parent, alpha=alpha)

    if min_width is not None:
        w.setMinimumWidth(min_width)
    w.setMaximumWidth(16777215)
    w.setMinimumHeight(80)
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return w
