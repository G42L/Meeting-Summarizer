#!/usr/bin/env python3
"""
whisper_engine.py
------------------
Everything about the Whisper side of the pipeline: which models exist,
which are downloaded, how to download one, and how to transcribe a file
with either faster-whisper (Python) or whisper-cli (compiled binary).

Note: this file is named whisper_engine.py rather than whisper.py on
purpose -- a local module literally called whisper.py in the same folder
as main.py would shadow the real `whisper`/`faster_whisper` packages if
anything ever does `import whisper`, which is the kind of bug that's
invisible until it silently isn't.
"""

import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# ----------------------------------------------------------------------
# Model catalogue
# ----------------------------------------------------------------------

# Hardcoded model sizes (disk & memory) -- these don't change often enough
# to be worth fetching remotely.
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


def get_whisper_model_path(model_name):
    path = Path.home() / "whisper.cpp" / "models" / f"ggml-{model_name}.bin"
    return str(path) if path.exists() else None


# Best-to-worst tiers for auto-picking a sensible default. Deliberately
# just the base names (no ".en") -- pick_default_model() checks each tier's
# non-English variant first and only falls back to ".en" if that's what's
# actually downloaded, since a ".en" model can't be used on non-English
# audio and the app has no way to know the recording's language upfront.
MODEL_PRIORITY = [
    "large-v3", "large-v3-turbo", "large-v2", "large-v1",
    "medium", "small", "base", "tiny",
]


def pick_default_model(models_info):
    """
    Given the list from get_whisper_models_info(), return the name of the
    best downloaded model per MODEL_PRIORITY (non-.en preferred over .en
    within the same tier). Falls back to any other downloaded model, then
    to the first model in the list, so this always returns *something* as
    long as models_info is non-empty.
    """
    if not models_info:
        return None
    downloaded = {m["name"] for m in models_info if m["downloaded"]}

    for base in MODEL_PRIORITY:
        if base in downloaded:
            return base
        en_variant = f"{base}.en"
        if en_variant in downloaded:
            return en_variant

    # Nothing from the priority list is downloaded (e.g. only tiny.en, or
    # an oddly-named extra model on disk) -- take whatever *is* downloaded.
    for m in models_info:
        if m["downloaded"]:
            return m["name"]

    # Nothing downloaded at all -- fall back to the first entry so the
    # combo box still has a valid selection (the app will prompt to
    # download it, same as selecting any other not-yet-downloaded model).
    return models_info[0]["name"]


def get_whisper_models_info():
    """
    Returns a list of dicts for each available model.
    Each dict: name, downloaded (bool), disk_size (str), mem_usage (str), ...
    """
    models_dir = Path.home() / "whisper.cpp" / "models"
    existing = set()
    if models_dir.exists():
        for f in models_dir.glob("ggml-*.bin"):
            name = f.name.replace("ggml-", "").replace(".bin", "")
            existing.add(name)

    result = []
    for name in MODEL_DATA:
        result.append({
            "name": name,
            "downloaded": name in existing,
            "disk_size": MODEL_DATA[name]["disk"],
            "mem_usage": MODEL_DATA[name]["mem"],
            "language": MODEL_DATA[name]["lan"],
            "speed": MODEL_DATA[name]["speed"],
            "accuracy": MODEL_DATA[name]["accuracy"],
            "usage": MODEL_DATA[name]["usage"],
        })

    # Any extra models found on disk that aren't in the standard catalogue
    extra = existing - set(MODEL_DATA.keys())
    for name in sorted(extra):
        result.append({
            "name": name,
            "downloaded": True,
            "disk_size": "?",
            "mem_usage": "?",
            "language": "?",
            "speed": "?",
            "accuracy": "?",
            "usage": "?",
        })
    return result


def download_whisper_model(model_name):
    """
    Downloads the ggml model using the whisper.cpp download script.
    Returns (success, error_message) -- error_message is None on success,
    and a human-readable reason (network error, disk full, bad model name,
    ...) on failure, instead of just a bare bool that throws the reason away.
    """
    whisper_cpp_dir = Path.home() / "whisper.cpp"
    script = whisper_cpp_dir / "models" / "download-ggml-model.sh"
    if not script.exists():
        return False, f"download-ggml-model.sh not found at {script}"
    try:
        subprocess.run(
            ["bash", str(script), model_name],
            cwd=whisper_cpp_dir / "models",
            check=True,
            capture_output=True,
            text=True,
        )
        return True, None
    except subprocess.CalledProcessError as e:
        # download-ggml-model.sh writes its actual error (bad model name,
        # curl/network failure, disk full, ...) to stdout/stderr; stderr is
        # usually where curl reports failures, but the script itself often
        # prints to stdout, so surface whichever one has content.
        detail = (e.stderr or e.stdout or "").strip()
        return False, detail or f"exit code {e.returncode}"
    except OSError as e:
        return False, str(e)


# ----------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------

def _load_faster_whisper_model(whisper_model, log):
    """
    Shared by transcribe_faster/transcribe_faster_with_segments. Returns the
    loaded WhisperModel, or None (with a log message already emitted) if
    faster-whisper isn't installed or the model couldn't be loaded.
    """
    from faster_whisper import WhisperModel

    try:
        log(f"⏳ Loading Whisper model '{whisper_model}' from local cache (no network)...")
        model = WhisperModel(whisper_model, device="cpu", compute_type="int8", local_files_only=True)
        log("✅ Model loaded from local cache.")
        return model
    except Exception:
        # Not cached yet -- this is the one-time case where a network call to
        # the Hugging Face Hub is unavoidable. After this, local_files_only=True
        # above will find it and every future run of this model is fully offline.
        log(f"Model '{whisper_model}' isn't cached locally yet -- downloading it once from Hugging Face...")
        try:
            log(f"⏳ Loading Whisper model '{whisper_model}' on CPU with int8 quantization...")
            log("   (This may take several minutes for large models like large-v3)")
            model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
            log("✅ Model downloaded and cached. Future runs of this model will be fully offline.")
            return model
        except Exception as e:
            log(f"❌ Failed to load Whisper model: {e}")
            return None


def transcribe_faster(audio_file, whisper_model, output_dir, log, queue_worker=None):
    """
    Transcribe with faster-whisper (pure Python, CPU int8). Emits progress
    via `log(str)`. Returns the transcript string, or None on failure.
    `queue_worker`, if given, is polled for `.stop_current` so a job can be
    cancelled mid-transcription.
    """
    try:
        model = _load_faster_whisper_model(whisper_model, log)
    except ImportError:
        log("faster-whisper not installed. Falling back to whisper-cli.")
        return transcribe_cli(audio_file, whisper_model, output_dir, log, queue_worker)
    if model is None:
        return None

    try:
        log(f"🌊 Starting transcription of '{audio_file}'...")
        segments, info = model.transcribe(audio_file, beam_size=5)
        log(f"📊 Language: {info.language}, probability: {info.language_probability:.2f}")

        transcript_parts = []
        segment_count = 0
        for seg in segments:
            if queue_worker is not None and getattr(queue_worker, "stop_current", False):
                log("🛑 Cancelled during transcription.")
                return None
            segment_count += 1
            text = seg.text.strip()
            if text:
                log(f"[{segment_count}] {text}")
            transcript_parts.append(seg.text)

        transcript = " ".join(transcript_parts)
        log(f"✅ Transcription complete. {segment_count} segments processed.")
    except Exception as e:
        log(f"❌ Transcription error: {e}")
        return None

    transcript_file = Path(output_dir) / "transcript.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    return transcript


def transcribe_faster_with_segments(audio_file, whisper_model, output_dir, log, queue_worker=None):
    """
    Same as transcribe_faster, but also returns the per-segment start/end
    timestamps faster-whisper already computes (and transcribe_faster
    discards) -- needed to align speaker-diarization turns to transcript
    text. Returns (transcript, segments) where segments is a list of
    {"start": float, "end": float, "text": str}, or (None, None) on
    failure. whisper-cli has no equivalent (its plain -otxt output carries
    no timestamps), so diarization is faster-whisper-only.
    """
    try:
        model = _load_faster_whisper_model(whisper_model, log)
    except ImportError:
        log("faster-whisper not installed -- can't transcribe with segment timestamps for diarization.")
        return None, None
    if model is None:
        return None, None

    try:
        log(f"🌊 Starting transcription of '{audio_file}'...")
        raw_segments, info = model.transcribe(audio_file, beam_size=5)
        log(f"📊 Language: {info.language}, probability: {info.language_probability:.2f}")

        transcript_parts = []
        segments = []
        segment_count = 0
        for seg in raw_segments:
            if queue_worker is not None and getattr(queue_worker, "stop_current", False):
                log("🛑 Cancelled during transcription.")
                return None, None
            segment_count += 1
            text = seg.text.strip()
            if text:
                log(f"[{segment_count}] {text}")
            transcript_parts.append(seg.text)
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

        transcript = " ".join(transcript_parts)
        log(f"✅ Transcription complete. {segment_count} segments processed.")
    except Exception as e:
        log(f"❌ Transcription error: {e}")
        return None, None

    transcript_file = Path(output_dir) / "transcript.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    return transcript, segments


def transcribe_cli(audio_file, whisper_model, output_dir, log, queue_worker=None):
    """Transcribe with the compiled whisper-cli binary. Returns transcript or None."""
    model_path = get_whisper_model_path(whisper_model)
    if not model_path:
        log(f"Model '{whisper_model}' not found. Attempting to download...")
        success, error = download_whisper_model(whisper_model)
        if not success:
            log(f"❌ Failed to download model '{whisper_model}': {error}")
            return None
        model_path = get_whisper_model_path(whisper_model)
        if not model_path:
            log("Model still not found after download.")
            return None
        log("Download complete.")

    if queue_worker is not None and getattr(queue_worker, "stop_current", False):
        log("Cancelled before transcription.")
        return None

    whisper_cli = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    if not whisper_cli.exists():
        log("whisper-cli not found. Please build whisper.cpp or install faster-whisper.")
        return None

    output_base = str(Path(output_dir) / "meeting")
    cmd = [str(whisper_cli), "-m", model_path, "-f", audio_file, "-otxt", "-of", output_base]

    # No fixed wall-clock timeout here -- a real meeting recording can take
    # whisper-cli well over 5 minutes to transcribe on CPU, especially with
    # larger models, and a hard cap was killing perfectly healthy long
    # transcriptions. Instead we poll so Cancel can still interrupt it.
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as e:
        log(f"❌ Failed to start whisper-cli: {e}")
        return None

    while proc.poll() is None:
        if queue_worker is not None and getattr(queue_worker, "stop_current", False):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            log("🛑 Cancelled during transcription.")
            return None
        time.sleep(0.5)

    _, stderr = proc.communicate()
    if proc.returncode != 0:
        log(f"❌ whisper-cli error: {stderr}")
        return None

    transcript_file = Path(output_dir) / "meeting.txt"
    with open(transcript_file, "r") as f:
        return f.read()


def transcribe(audio_file, whisper_model, use_cli, output_dir, log, queue_worker=None):
    """Dispatch to the CLI or Python backend depending on `use_cli`."""
    if use_cli:
        return transcribe_cli(audio_file, whisper_model, output_dir, log, queue_worker)
    return transcribe_faster(audio_file, whisper_model, output_dir, log, queue_worker)


# ----------------------------------------------------------------------
# Background model download (keeps the GUI thread responsive)
# ----------------------------------------------------------------------

class DownloadWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)  # success, model_name
    log = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    @pyqtSlot()
    def run(self):
        self.log.emit(f"Downloading model '{self.model_name}'...")
        success, error = download_whisper_model(self.model_name)
        self.log.emit("Download completed." if success else f"❌ Download failed: {error}")
        self.finished.emit(success, self.model_name)


# ----------------------------------------------------------------------
# Live (rolling) transcription while recording
# ----------------------------------------------------------------------

class LiveTranscriptionWorker(QObject):
    """
    Runs in a background QThread for the duration of a recording,
    transcribing small rolling chunks of mixed audio (handed in via
    feed_audio(), fed from AudioMixerEngine.pull_live_transcription_audio())
    as they arrive, and emitting each chunk's text as a rough live preview.

    This is deliberately just a preview: chunk boundaries can fall
    mid-sentence (each chunk is transcribed independently, with no context
    from the one before it), and it uses beam_size=1 for speed over
    accuracy. The authoritative transcript still comes from transcribe()/
    transcribe_faster() run against the complete saved recording once Stop
    is pressed, same as before this existed -- this worker's output is
    never written to disk or used for anything but the on-screen preview.

    faster-whisper only: whisper-cli has no way to transcribe an in-memory
    numpy array without a round-trip through a temp file per chunk, which
    at a few-second chunk cadence would spend more time on process
    start-up than on the audio itself.
    """
    partial_text = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, whisper_model):
        super().__init__()
        self.whisper_model_name = whisper_model
        self._queue = deque()
        self._lock = threading.Lock()
        self._new_audio = threading.Event()
        self._stop_event = threading.Event()

    def feed_audio(self, audio_chunk):
        """Thread-safe -- called from the main thread. Queues a chunk of
        mono float32 audio at ENGINE_SAMPLE_RATE for the worker thread to
        transcribe next. No-op for an empty/None chunk (e.g. nothing new
        was recorded since the last poll, or the recording is paused)."""
        if audio_chunk is None or getattr(audio_chunk, "size", 0) == 0:
            return
        with self._lock:
            self._queue.append(audio_chunk)
        self._new_audio.set()

    def stop(self):
        """Thread-safe -- signals run()'s loop to exit after its current
        chunk (if any). Does not block; caller still needs QThread.wait()
        to know the thread has actually finished."""
        self._stop_event.set()
        self._new_audio.set()  # wake up a blocked wait() so the stop is noticed promptly

    def _drain(self):
        with self._lock:
            if not self._queue:
                return None
            chunks = list(self._queue)
            self._queue.clear()
        return np.concatenate(chunks)

    @pyqtSlot()
    def run(self):
        try:
            model = _load_faster_whisper_model(self.whisper_model_name, self.log.emit)
        except ImportError:
            self.log.emit("❌ Live transcript needs faster-whisper (not installed) -- disabling for this recording.")
            return
        if model is None:
            return

        while not self._stop_event.is_set():
            self._new_audio.wait(timeout=0.5)
            self._new_audio.clear()
            chunk = self._drain()
            if chunk is None or chunk.size == 0:
                continue
            try:
                segments, _info = model.transcribe(chunk, beam_size=1, condition_on_previous_text=False)
                text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            except Exception as e:
                self.log.emit(f"⚠️ Live transcript error: {e}")
                continue
            if text:
                self.partial_text.emit(text)
