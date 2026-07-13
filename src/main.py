#!/usr/bin/env python3
"""
Meeting Transcriber - PyQt5 GUI
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
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QProgressBar,
    QGroupBox, QFileDialog, QMessageBox, QCheckBox, QSizePolicy, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QIcon, QFont

import audio_engine
import vu_meters
import whisper_engine
import llm_backend
import pipeline
import sysmon


# ----------------------------------------------------------------------
# One row in the "Audio Sources" panel: name, gain, mute, mini VU meter,
# remove button. Purely a UI widget -- AudioMixerEngine doesn't know this
# class exists; MainWindow is the only thing that wires the two together.
# ----------------------------------------------------------------------
class SourceRow(QWidget):
    remove_clicked = pyqtSignal(str)   # emits the source name to remove
    gain_changed = pyqtSignal(str, float)     # name, gain (0.0 .. 2.0)
    mute_changed = pyqtSignal(str, bool)      # name, muted

    def __init__(self, name, display_label, parent=None):
        super().__init__(parent)
        self.source_name = name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        label = QLabel(display_label)
        label.setMinimumWidth(220)
        layout.addWidget(label, stretch=2)

        layout.addWidget(QLabel("Gain:"))
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 200)  # percent
        self.gain_slider.setValue(100)
        self.gain_slider.setMaximumWidth(110)
        layout.addWidget(self.gain_slider)
        self.gain_value_label = QLabel("100%")
        self.gain_value_label.setMinimumWidth(38)
        layout.addWidget(self.gain_value_label)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)

        self.mute_check = QCheckBox("Mute")
        self.mute_check.toggled.connect(lambda checked: self.mute_changed.emit(self.source_name, checked))
        layout.addWidget(self.mute_check)

        # Small, fixed-size VU meter just for this one source.
        self.vu = vu_meters.MiniLEDHorizontalVUMeter(alpha=0.10)
        self.vu.setMinimumHeight(20)
        self.vu.setMaximumHeight(24)
        self.vu.setMinimumWidth(90)
        self.vu.setMaximumWidth(140)
        layout.addWidget(self.vu)

        remove_btn = QPushButton("✕")
        remove_btn.setMaximumWidth(28)
        remove_btn.setToolTip("Remove this source")
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self.source_name))
        layout.addWidget(remove_btn)

    def _on_gain_changed(self, percent):
        self.gain_value_label.setText(f"{percent}%")
        self.gain_changed.emit(self.source_name, percent / 100.0)


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Transcriber")
        self.setMinimumSize(960, 925)

        self.button_height = 30
        self.button_radius = 8

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

        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setup_ui()
        self.refresh_source_picker()
        self.refresh_whisper_models()
        self.refresh_backends()

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

        action_btn_style = (f"height: {self.button_height}px; border-radius: {self.button_radius}px; padding: 8px; border: 1px solid #cccccc;")
        refresh_btn_style = (f"border-radius: {self.button_radius}px; padding: 8px; border: 1px solid #cccccc;")

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
        add_source_btn = QPushButton("➕ Add")
        add_source_btn.clicked.connect(self.add_selected_source)
        picker_row.addWidget(add_source_btn)
        refresh_dev_btn = QPushButton("Refresh")
        refresh_dev_btn.clicked.connect(self.refresh_source_picker)
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
        llm_layout.addWidget(refresh_backend_btn)
        self.llm_group.setLayout(llm_layout)
        layout.addWidget(self.llm_group)

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

        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self.cancel_current_job)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(action_btn_style)
        control_layout.addWidget(self.cancel_btn)

        self.clear_log_btn = QPushButton("🗑️ Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.clear_log_btn.setStyleSheet(action_btn_style)
        control_layout.addWidget(self.clear_log_btn)

        layout.addLayout(control_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("monospace"))
        layout.addWidget(QLabel("Log / Summary Output:"))
        layout.addWidget(self.log_text)

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

        self.last_md_path = None

    def closeEvent(self, event):
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
            icon = "💻" if d["is_loopback"] else "🎤"
            already_added = d["name"] in self.mixer.sources
            label = f"{icon} {d['name']}" + ("  (already added)" if already_added else "")
            self.source_picker_combo.addItem(label, d)
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

        icon = "💻" if dev["is_loopback"] else "🎤"
        row = SourceRow(name, f"{icon} {name}")
        row.remove_clicked.connect(self.remove_source)
        row.gain_changed.connect(lambda n, g: self.mixer.set_gain(n, g))
        row.mute_changed.connect(lambda n, m: self.mixer.set_muted(n, m))
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
        if self.record_btn.text() == "🎤 Record":
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.mixer.sources:
            QMessageBox.warning(self, "No sources", "Add at least one audio source before recording.")
            return

        self.load_btn.setEnabled(False)
        self.record_btn.setText("⏹ Stop")
        self.record_btn.setStyleSheet(
            f"background-color: #ff6b6b; font-weight: bold; font-size: 14px; "
            f"height: {self.button_height}px; border-radius: {self.button_radius}px; padding: 8px; "
            "border: 1px solid #cc0000;"
        )
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
                self.whisper_combo.setItemData(idx, info, Qt.UserRole)
                tooltip = (f"Model: {info['name']}\n"
                           f"Disk: {info['disk_size']}\n"
                           f"Memory: {info['mem_usage']}\n"
                           f"Language: {info['language']}\n"
                           f"Speed: {info['speed']}\n"
                           f"Accuracy: {info['accuracy']}\n"
                           f"Usage: {info['usage']}\n"
                           f"Status: {'Downloaded' if info['downloaded'] else 'Not downloaded'}")
                self.whisper_combo.setItemData(idx, tooltip, Qt.ToolTipRole)

            default_name = whisper_engine.pick_default_model(models_info)
            default_index = 0
            for i in range(self.whisper_combo.count()):
                info = self.whisper_combo.itemData(i, Qt.UserRole)
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

        info = self.whisper_combo.itemData(index, Qt.UserRole)
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
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            self.download_model(info["name"])
        else:
            all_infos = [self.whisper_combo.itemData(i, Qt.UserRole) for i in range(self.whisper_combo.count())]
            all_infos = [i for i in all_infos if isinstance(i, dict)]
            revert_name = whisper_engine.pick_default_model(all_infos) if all_infos else None

            target_index = None
            for i in range(self.whisper_combo.count()):
                info2 = self.whisper_combo.itemData(i, Qt.UserRole)
                if isinstance(info2, dict) and info2.get("name") == revert_name and info2.get("downloaded", False):
                    target_index = i
                    break
            if target_index is not None:
                self.whisper_combo.setCurrentIndex(target_index)

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
                    info = self.whisper_combo.itemData(i, Qt.UserRole)
                    if info and info["name"] == model_name:
                        self.whisper_combo.setCurrentIndex(i)
                        break
            else:
                self.append_log(f"❌ Download of '{model_name}' failed. Please download manually.")
                for i in range(self.whisper_combo.count()):
                    info = self.whisper_combo.itemData(i, Qt.UserRole)
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
        item_data = self.whisper_combo.itemData(current_idx, Qt.UserRole)
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

        job = pipeline.Job(audio_file_path, whisper_model, backend_info, llm_model, use_cli, output_dir=output_dir)
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

    def on_job_started(self, job_id):
        self.append_log(f"🔄 Processing job #{job_id}...")
        self.cancel_btn.setEnabled(True)

    def cancel_current_job(self):
        if self.queue_worker:
            self.queue_worker.stop_current_job()
            self.append_log("⏹ Cancelling current job...")
            self.cancel_btn.setEnabled(False)

    def on_error(self, msg):
        self.append_log(f"❌ Error: {msg}")
        self.job_count -= 1
        self.update_config_lock()
        self.reset_ui()

    def append_log(self, text):
        """Add a diagnostic log line to the console. Markdown-special characters
        are escaped so paths/messages (e.g. 'a_b*c') don't get misread as
        emphasis -- unlike append_summary, this text isn't meant to be Markdown."""
        escaped = re.sub(r'([\\`*_\[\]#])', r'\\\1', text)
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

    def _flush_console(self):
        scrollbar = self.log_text.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        self.log_text.setMarkdown(self._console_md)
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

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
        # No manual blanking or timer bookkeeping needed here -- self.monitor_timer
        # runs continuously and will resume driving the waveform/VU meters from
        # live source monitoring now that loaded_samples is back to None.

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
            self.append_log(f"📁 Saved copy to: {save_path}")

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

    def clear_log(self):
        self._console_render_timer.stop()
        self._console_md = ""
        self._console_last_was_summary = False
        self.log_text.clear()

    # ---------- Lock configuration ----------
    def update_config_lock(self):
        locked = self.is_recording or self.job_count > 0 or getattr(self, '_downloading', False)
        enabled = not locked
        self.dev_group.setEnabled(enabled)
        self.whisper_group.setEnabled(enabled)
        self.llm_group.setEnabled(enabled)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Transcriber")
    app.setWindowIcon(QIcon("icon.svg"))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
