"""
Tests for diarization.py. The real pyannote.audio model/download/HF-token
flow isn't exercised here (no network, no gated-model access in CI) --
is_available()/diarize() are tested only for their graceful-degradation
paths, and label_transcript() (pure alignment logic, no I/O) is tested
thoroughly with synthetic timestamps.
"""
import sys

from src import diarization


# ---------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------

def test_is_available_false_when_pyannote_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    monkeypatch.setitem(sys.modules, "pyannote", None)
    assert diarization.is_available() is False


# ---------------------------------------------------------------------
# diarize -- graceful failure paths only (no real model/network here)
# ---------------------------------------------------------------------

def test_diarize_returns_none_when_pyannote_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    monkeypatch.setitem(sys.modules, "pyannote", None)
    logs = []
    result = diarization.diarize("audio.wav", "hf_token", logs.append)
    assert result is None
    assert any("not installed" in line for line in logs)


def test_diarize_returns_none_when_token_missing(monkeypatch):
    # Make the pyannote.audio import succeed so we reach the token check.
    import types
    fake_module = types.ModuleType("pyannote.audio")
    fake_module.Pipeline = object()
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    logs = []
    result = diarization.diarize("audio.wav", None, logs.append)
    assert result is None
    assert any("Hugging Face access token" in line for line in logs)


def test_diarize_uses_local_cache_first_without_requiring_a_token(monkeypatch):
    """If the model is already cached, diarize() must succeed with
    hf_token=None -- this is the whole point of trying local_files_only
    before ever asking for a token."""
    import types

    calls = []

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            return iter([])

    class FakePipeline:
        def __init__(self):
            pass

        @staticmethod
        def from_pretrained(name, local_files_only=False, use_auth_token=None):
            calls.append({"name": name, "local_files_only": local_files_only, "use_auth_token": use_auth_token})
            if local_files_only:
                return FakePipeline()
            raise AssertionError("should not need a token-authenticated download when already cached")

        def __call__(self, audio_file):
            return FakeAnnotation()

    fake_module = types.ModuleType("pyannote.audio")
    fake_module.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    logs = []
    result = diarization.diarize("audio.wav", None, logs.append)

    assert result == []
    assert calls[0]["local_files_only"] is True
    assert any("no token needed" in line for line in logs)


def test_diarize_falls_back_to_token_download_when_not_cached(monkeypatch):
    import types

    calls = []

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            return iter([])

    class FakePipeline:
        @staticmethod
        def from_pretrained(name, local_files_only=False, use_auth_token=None):
            calls.append({"local_files_only": local_files_only, "use_auth_token": use_auth_token})
            if local_files_only:
                raise OSError("not cached")
            return FakePipeline()

        def __call__(self, audio_file):
            return FakeAnnotation()

    fake_module = types.ModuleType("pyannote.audio")
    fake_module.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)

    logs = []
    result = diarization.diarize("audio.wav", "hf_faketoken", logs.append)

    assert result == []
    assert calls[0]["local_files_only"] is True
    assert calls[1]["use_auth_token"] == "hf_faketoken"


# ---------------------------------------------------------------------
# is_model_cached_locally
# ---------------------------------------------------------------------

def test_is_model_cached_locally_true_when_repo_id_present(monkeypatch):
    import types

    class FakeRepo:
        def __init__(self, repo_id):
            self.repo_id = repo_id

    class FakeCacheInfo:
        repos = [FakeRepo("some/other-model"), FakeRepo(diarization.MODEL_NAME)]

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.scan_cache_dir = lambda: FakeCacheInfo()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    assert diarization.is_model_cached_locally() is True


def test_is_model_cached_locally_false_when_repo_id_absent(monkeypatch):
    import types

    class FakeRepo:
        def __init__(self, repo_id):
            self.repo_id = repo_id

    class FakeCacheInfo:
        repos = [FakeRepo("some/other-model")]

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.scan_cache_dir = lambda: FakeCacheInfo()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    assert diarization.is_model_cached_locally() is False


def test_is_model_cached_locally_false_when_huggingface_hub_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert diarization.is_model_cached_locally() is False


def test_is_model_cached_locally_false_on_scan_error(monkeypatch):
    import types

    def _raise():
        raise OSError("cache dir unreadable")

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.scan_cache_dir = _raise
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    assert diarization.is_model_cached_locally() is False


# ---------------------------------------------------------------------
# label_transcript
# ---------------------------------------------------------------------

def test_label_transcript_empty_segments_returns_empty_string():
    assert diarization.label_transcript([], []) == ""


def test_label_transcript_no_turns_joins_plain_text():
    segments = [{"start": 0.0, "end": 1.0, "text": "hello"}, {"start": 1.0, "end": 2.0, "text": "world"}]
    assert diarization.label_transcript(segments, []) == "hello world"


def test_label_transcript_single_speaker():
    segments = [{"start": 0.0, "end": 1.0, "text": "hello"}, {"start": 1.0, "end": 2.0, "text": "world"}]
    turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    assert diarization.label_transcript(segments, turns) == "**SPEAKER_00:** hello world"


def test_label_transcript_alternating_speakers_produce_separate_paragraphs():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hi there"},
        {"start": 1.0, "end": 2.0, "text": "hello back"},
        {"start": 2.0, "end": 3.0, "text": "how are you"},
    ]
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00"},
    ]
    result = diarization.label_transcript(segments, turns)
    assert result == (
        "**SPEAKER_00:** hi there\n\n"
        "**SPEAKER_01:** hello back\n\n"
        "**SPEAKER_00:** how are you"
    )


def test_label_transcript_consecutive_same_speaker_segments_merge():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "part one"},
        {"start": 1.0, "end": 2.0, "text": "part two"},
        {"start": 5.0, "end": 6.0, "text": "different speaker now"},
    ]
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_01"},
    ]
    result = diarization.label_transcript(segments, turns)
    assert result == "**SPEAKER_00:** part one part two\n\n**SPEAKER_01:** different speaker now"


def test_label_transcript_segment_with_no_overlapping_turn_uses_nearest():
    # Segment falls in a gap between two turns -- should attach to the
    # turn whose start is closest, not silently drop the text.
    segments = [{"start": 10.0, "end": 11.0, "text": "orphan segment"}]
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 9.0, "end": 9.5, "speaker": "SPEAKER_01"},
    ]
    result = diarization.label_transcript(segments, turns)
    assert result == "**SPEAKER_01:** orphan segment"


def test_label_transcript_skips_blank_segments():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "  "},
        {"start": 1.0, "end": 2.0, "text": "real text"},
    ]
    turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    assert diarization.label_transcript(segments, turns) == "**SPEAKER_00:** real text"
