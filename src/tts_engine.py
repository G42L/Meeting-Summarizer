#!/usr/bin/env python3
"""
tts_engine.py
-------------
Text-to-speech via Piper (optional, lightweight dependency -- pure Python +
onnxruntime, no torch). Prototype for reading a transcript/summary back as
audio.

Requires:
  - piper-tts installed (`pip install piper-tts`)
  - a voice model downloaded once (`python -m piper.download_voices <voice>`),
    cached locally afterward -- same shape as the faster-whisper model cache.

Degrades gracefully like diarization.py: if piper-tts isn't installed,
is_available() reports that cleanly instead of raising, and callers are
expected to skip TTS rather than fail the whole job.
"""

from pathlib import Path

# A few curated voices to start with -- Piper's full catalogue is much
# larger (https://github.com/rhasspy/piper/blob/master/VOICES.md); add more
# here as needed. Sizes are approximate (medium-quality voices, .onnx + .json).
VOICE_DATA = {
    "en_US-lessac-medium": {"lang": "en_US", "quality": "medium", "size": "~60 MiB"},
    "en_US-amy-medium":    {"lang": "en_US", "quality": "medium", "size": "~60 MiB"},
    "en_GB-alan-medium":   {"lang": "en_GB", "quality": "medium", "size": "~60 MiB"},
}

VOICES_DIR = Path.home() / ".local" / "share" / "piper" / "voices"

DEFAULT_VOICE = "en_US-lessac-medium"


def is_available():
    """True if piper-tts is importable."""
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


def get_voice_path(voice_name):
    path = VOICES_DIR / f"{voice_name}.onnx"
    return path if path.exists() else None


def get_tts_voices_info():
    """Returns a list of dicts for each known voice: name, downloaded, lang, quality, size."""
    result = []
    for name, data in VOICE_DATA.items():
        result.append({
            "name": name,
            "downloaded": get_voice_path(name) is not None,
            "language": data["lang"],
            "quality": data["quality"],
            "size": data["size"],
        })
    return result


def download_voice(voice_name, log):
    """
    Downloads a Piper voice via `python -m piper.download_voices`.
    Returns (success, error_message).
    """
    import subprocess
    import sys

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log(f"⏳ Downloading voice '{voice_name}'...")
        subprocess.run(
            [sys.executable, "-m", "piper.download_voices", voice_name,
             "--download-dir", str(VOICES_DIR)],
            check=True, capture_output=True, text=True,
        )
        log("✅ Voice downloaded.")
        return True, None
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "").strip()
        return False, detail or f"exit code {e.returncode}"
    except OSError as e:
        return False, str(e)


def synthesize(text, voice_name, output_path, log):
    """
    Synthesize `text` to a WAV file at `output_path` using the given voice.
    Downloads the voice first if it isn't cached locally yet. Returns the
    output path on success, or None on failure (each failure is logged via
    `log(str)` rather than raised).
    """
    try:
        from piper import PiperVoice
    except ImportError:
        log("❌ piper-tts not installed -- can't run text-to-speech.")
        return None

    voice_path = get_voice_path(voice_name)
    if voice_path is None:
        success, error = download_voice(voice_name, log)
        if not success:
            log(f"❌ Failed to download voice '{voice_name}': {error}")
            return None
        voice_path = get_voice_path(voice_name)
        if voice_path is None:
            log("Voice still not found after download.")
            return None

    try:
        log(f"⏳ Loading voice '{voice_name}'...")
        voice = PiperVoice.load(str(voice_path))
    except Exception as e:
        log(f"❌ Failed to load voice: {e}")
        return None

    try:
        import wave
        log("🔊 Synthesizing audio...")
        with wave.open(str(output_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        log(f"✅ Synthesis complete: {output_path}")
        return output_path
    except Exception as e:
        log(f"❌ Synthesis failed: {e}")
        return None


if __name__ == "__main__":
    # Quick manual smoke test: python -m src.tts_engine "some text"
    import sys

    sample_text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test of Piper text to speech."
    out = Path("tts_test_output.wav")
    result = synthesize(sample_text, "en_US-lessac-medium", out, log=print)
    if result:
        print(f"Wrote {result.resolve()}")
    else:
        print("Synthesis failed -- see log above.")
