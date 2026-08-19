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

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from . import whisper_engine
from . import llm_backend
from . import diarization
from . import tts_engine


def new_output_dir():
    """
    Create and return a fresh timestamped folder under ./transcripts/, e.g.
    ./transcripts/2026-07-12 14.30.00/. Shared by both the record-a-new-
    meeting path (MainWindow.stop_recording) and the load-existing-file
    path (ProcessingWorker.process, when the caller didn't pre-assign an
    output_dir) so the two don't drift out of sync with each other.
    """
    base_dir = Path.cwd() / "transcripts"
    base_dir.mkdir(exist_ok=True)
    folder_name = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    output_dir = base_dir / folder_name
    output_dir.mkdir(exist_ok=True)
    return output_dir


class Job:
    """A single transcription job."""
    _counter = 0

    def __init__(self, audio_file_path, whisper_model, backend_info, llm_model, use_whisper_cli, output_dir=None,
                 prompt_template=None, review_transcript=False, enable_diarization=False, hf_token=None,
                 enable_tts=False):
        Job._counter += 1
        self.id = Job._counter
        self.audio_file_path = audio_file_path
        self.whisper_model = whisper_model
        self.backend_info = backend_info
        self.llm_model = llm_model
        self.use_whisper_cli = use_whisper_cli
        self.output_dir = output_dir
        self.prompt_template = prompt_template
        self.review_transcript = review_transcript
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token
        self.enable_tts = enable_tts


class ProcessingWorker(QObject):
    """Performs transcription and summarization for a single job (no recording)."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    summary_chunk = pyqtSignal(str)
    finished = pyqtSignal(str)   # md_path
    error = pyqtSignal(str)
    transcript_ready = pyqtSignal(str)   # emitted when job.review_transcript is True

    def __init__(self):
        super().__init__()
        # Set right before transcript_ready is emitted, polled by process()
        # below until the GUI thread (which owns whatever dialog shows the
        # transcript) sets one of them -- same plain-attribute-polling idiom
        # already used for queue_worker.stop_current, just for a different
        # pause point. Never touched from here except to reset/read them.
        self.review_result = None
        self.review_cancelled = False

    def process(self, job, queue_worker):
        if job.output_dir is not None:
            self.output_dir = Path(job.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = new_output_dir()
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
            segments = None
            if job.enable_diarization and job.use_whisper_cli:
                self.log.emit("⚠️ Diarization requires the faster-whisper backend (not whisper-cli) -- skipping.")

            if job.enable_diarization and not job.use_whisper_cli:
                transcript, segments = whisper_engine.transcribe_faster_with_segments(
                    audio_file, job.whisper_model, self.output_dir, self.log.emit, queue_worker,
                )
            else:
                transcript = whisper_engine.transcribe(
                    audio_file, job.whisper_model, job.use_whisper_cli,
                    self.output_dir, self.log.emit, queue_worker,
                )

            if transcript is None:
                self.error.emit("Transcription failed.")
                return
            self.log.emit("✅ Transcription complete.")
            self.progress.emit(60)

            if segments is not None:
                if diarization.is_available():
                    self.log.emit("🗣️ Running speaker diarization...")
                    turns = diarization.diarize(audio_file, job.hf_token, self.log.emit)
                    if turns is not None:
                        transcript = diarization.label_transcript(segments, turns)
                        self.log.emit("✅ Speakers labeled.")
                    else:
                        self.log.emit("⚠️ Diarization failed -- continuing with the plain transcript.")
                else:
                    self.log.emit(
                        "⚠️ Diarization requested but pyannote.audio isn't installed -- "
                        "continuing with the plain transcript."
                    )

            if job.review_transcript:
                self.log.emit("📝 Awaiting transcript review...")
                self.review_result = None
                self.review_cancelled = False
                self.transcript_ready.emit(transcript)
                while self.review_result is None and not self.review_cancelled:
                    if queue_worker.stop_current:
                        self.log.emit("Job cancelled during review.")
                        self.error.emit("Cancelled by user.")
                        return
                    QThread.msleep(100)
                if self.review_cancelled:
                    self.log.emit("🛑 Transcript review cancelled.")
                    self.error.emit("Cancelled during transcript review.")
                    return
                transcript = self.review_result
                self.log.emit("✅ Transcript reviewed.")

            self.log.emit("🤖 Summarizing with LLM... (streaming response below)")
            self.progress.emit(70)
            summary = llm_backend.summarize(
                transcript, job.backend_info, job.llm_model,
                on_chunk=self.summary_chunk.emit, log=self.log.emit,
                queue_worker=queue_worker, prompt_template=job.prompt_template,
            )
            if summary is None:
                self.error.emit("Summarization failed.")
                return
            md_path = llm_backend.save_markdown(
                self.output_dir, audio_file, transcript, summary,
                job.backend_info, job.llm_model, job.whisper_model,
            )
            self.log.emit("✅ Summary generated.")
            self.progress.emit(90)

            if job.enable_tts:
                if tts_engine.is_available():
                    self.log.emit("🔊 Reading summary aloud...")
                    wav_path = self.output_dir / "summary.wav"
                    result = tts_engine.synthesize(summary, tts_engine.DEFAULT_VOICE, wav_path, self.log.emit)
                    if result is None:
                        self.log.emit("⚠️ Text-to-speech failed -- continuing without audio.")
                else:
                    self.log.emit(
                        "⚠️ Text-to-speech requested but piper-tts isn't installed -- "
                        "continuing without audio."
                    )

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
    transcript_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.queue = deque()
        self.processing = False
        self.stop_requested = False
        self.stop_current = False
        self.current_job = None
        self.current_processor = None   # the live ProcessingWorker, while one is running

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
                processor.transcript_ready.connect(self.transcript_ready)
                self.current_processor = processor

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
                    processor.transcript_ready.disconnect()
                    processor.deleteLater()
                    self.processing = False
                    self.current_job = None
                    self.current_processor = None
            else:
                QThread.msleep(100)
