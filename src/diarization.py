#!/usr/bin/env python3
"""
diarization.py
---------------
Speaker diarization via pyannote.audio (optional, heavyweight dependency --
pulls in torch/torchaudio). Labels transcript segments by which speaker was
talking, even when multiple people share one physical microphone (unlike a
cheap "which audio source was loudest" tag, this actually distinguishes
individual voices).

Requires:
  - pyannote.audio installed (`pip install pyannote.audio`)
  - a Hugging Face account, an access token, and having accepted the terms
    for the gated "pyannote/speaker-diarization-3.1" model on huggingface.co
  - network access the first time (to download the model); fully offline
    afterward, same shape as the faster-whisper model cache.

Degrades gracefully like the optional nvidia-ml-py GPU-stats dependency
(see sysmon.py / requirements.txt): if pyannote.audio isn't installed,
is_available() reports that cleanly instead of raising, and callers are
expected to skip diarization and fall back to the plain transcript.
"""


def is_available():
    """True if pyannote.audio is importable."""
    try:
        import pyannote.audio  # noqa: F401
        return True
    except ImportError:
        return False


def diarize(audio_file, hf_token, log):
    """
    Run speaker diarization on audio_file. Returns a list of
    {"start": float, "end": float, "speaker": str} turns, or None on
    failure (missing/invalid token, gated model not accepted, no network
    for the first-time model download, etc.) -- each failure logs a
    specific reason via `log(str)` rather than raising, so callers can
    fall back to the plain transcript instead of failing the whole job.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log("❌ pyannote.audio not installed -- can't run diarization.")
        return None

    if not hf_token:
        log(
            "❌ Diarization requires a Hugging Face access token (with the gated "
            "'pyannote/speaker-diarization-3.1' model's terms accepted at "
            "huggingface.co) -- set one in Whisper Model settings."
        )
        return None

    try:
        log("⏳ Loading speaker-diarization model (downloads once, then cached)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token,
        )
    except Exception as e:
        log(f"❌ Failed to load diarization model: {e}")
        return None

    try:
        log("🗣️ Running diarization (this can take a while on CPU)...")
        annotation = pipeline(audio_file)
    except Exception as e:
        log(f"❌ Diarization failed: {e}")
        return None

    turns = []
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append({"start": segment.start, "end": segment.end, "speaker": speaker})
    return turns


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def label_transcript(segments, turns):
    """
    Pure alignment logic, no I/O: tags each Whisper segment
    ({"start", "end", "text"}) with whichever diarization turn
    ({"start", "end", "speaker"}) it overlaps the most, then merges
    consecutive same-speaker segments into one paragraph. Returns the
    final "**Speaker X:** ..." Markdown string.

    A segment with no overlapping turn at all (e.g. diarization missed a
    short utterance) is attributed to the turn nearest in time, so no
    transcript text is ever silently dropped.
    """
    if not segments:
        return ""
    if not turns:
        # No diarization data at all -- can't label, just join the text.
        return " ".join(s["text"].strip() for s in segments if s["text"].strip())

    paragraphs = []
    current_speaker = None
    current_parts = []

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        best_turn = max(turns, key=lambda t: _overlap(seg["start"], seg["end"], t["start"], t["end"]))
        if _overlap(seg["start"], seg["end"], best_turn["start"], best_turn["end"]) == 0.0:
            # No actual overlap -- fall back to nearest by start-time distance.
            best_turn = min(turns, key=lambda t: abs(t["start"] - seg["start"]))
        speaker = best_turn["speaker"]

        if speaker != current_speaker:
            if current_parts:
                paragraphs.append((current_speaker, " ".join(current_parts)))
            current_speaker = speaker
            current_parts = [text]
        else:
            current_parts.append(text)

    if current_parts:
        paragraphs.append((current_speaker, " ".join(current_parts)))

    return "\n\n".join(f"**{speaker}:** {text}" for speaker, text in paragraphs)
