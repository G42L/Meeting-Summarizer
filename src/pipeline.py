#!/usr/bin/env python3
"""
pipeline.py
------------
The job queue that turns a saved audio file into a Markdown summary:
Job describes one unit of work, ProcessingWorker does the actual
transcribe -> summarize -> save steps for a single job, and QueueWorker
runs on its own QThread processing jobs one at a time so the GUI never
blocks (and so you can queue up several recordings back to back).
"""

import os
from datetime import datetime
from pathlib import Path
from collections import deque

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from . import whisper_engine
from . import llm_backend


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
        self.output_dir = output_dir


class ProcessingWorker(QObject):
    """Performs transcription and summarization for a single job (no recording)."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    summary_chunk = pyqtSignal(str)
    finished = pyqtSignal(str)   # md_path
    error = pyqtSignal(str)

    def process(self, job, queue_worker):
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

        if queue_worker.stop_current:
            self.log.emit("Job cancelled.")
            self.error.emit("Cancelled by user.")
            return
        try:
            audio_file = job.audio_file_path
            if not os.path.exists(audio_file):
                self.error.emit(f"Audio file not found: {audio_file}")
                return

            self.log.emit(f"📂 Processing audio file: {audio_file}")
            self.progress.emit(10)

            self.log.emit("📝 Transcribing with Whisper...")
            transcript = whisper_engine.transcribe(
                audio_file, job.whisper_model, job.use_whisper_cli,
                self.output_dir, self.log.emit, queue_worker,
            )
            if transcript is None:
                self.error.emit("Transcription failed.")
                return
            self.log.emit("✅ Transcription complete.")
            self.progress.emit(60)

            self.log.emit("🤖 Summarizing with LLM... (streaming response below)")
            self.progress.emit(70)
            summary = llm_backend.summarize(
                transcript, job.backend_info, job.llm_model,
                on_chunk=self.summary_chunk.emit, log=self.log.emit,
            )
            if summary is None:
                self.error.emit("Summarization failed.")
                return
            md_path = llm_backend.save_markdown(
                self.output_dir, audio_file, transcript, summary,
                job.backend_info, job.llm_model, job.whisper_model,
            )
            self.log.emit("✅ Summary generated.")
            self.progress.emit(100)
            self.finished.emit(md_path)
        except Exception as e:
            self.error.emit(str(e))


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
                self.stop_current = False
                self.job_started.emit(job.id)

                processor = ProcessingWorker()
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
                    processor.log.disconnect()
                    processor.progress.disconnect()
                    processor.summary_chunk.disconnect()
                    processor.finished.disconnect()
                    processor.error.disconnect()
                    processor.deleteLater()
                    self.processing = False
                    self.current_job = None
            else:
                QThread.msleep(100)
