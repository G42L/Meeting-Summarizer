"""
Tests for whisper_engine.py. External processes (whisper-cli,
download-ggml-model.sh) and the faster-whisper model loader are mocked --
these tests never touch the network or a real model.
"""
import subprocess
import sys

import pytest

from src import whisper_engine


# ---------------------------------------------------------------------
# pick_default_model
# ---------------------------------------------------------------------

def _models_info(downloaded_names):
    return [{"name": n, "downloaded": n in downloaded_names} for n in
             ["large-v3", "medium", "medium.en", "small", "tiny", "tiny.en"]]


def test_pick_default_model_empty_list_returns_none():
    assert whisper_engine.pick_default_model([]) is None


def test_pick_default_model_prefers_highest_priority_tier():
    info = _models_info({"medium", "medium.en", "small", "tiny"})
    assert whisper_engine.pick_default_model(info) == "medium"


def test_pick_default_model_falls_back_to_en_variant_within_tier():
    info = _models_info({"medium.en", "tiny"})
    assert whisper_engine.pick_default_model(info) == "medium.en"


def test_pick_default_model_falls_back_to_any_downloaded_model():
    info = [{"name": "some-weird-local-model", "downloaded": True}]
    assert whisper_engine.pick_default_model(info) == "some-weird-local-model"


def test_pick_default_model_falls_back_to_first_entry_if_nothing_downloaded():
    info = _models_info(set())
    assert whisper_engine.pick_default_model(info) == "large-v3"


# ---------------------------------------------------------------------
# get_whisper_model_path / get_whisper_models_info
# ---------------------------------------------------------------------

@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper_engine.Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


def test_get_whisper_model_path_missing_returns_none(fake_home):
    assert whisper_engine.get_whisper_model_path("tiny") is None


def test_get_whisper_model_path_found(fake_home):
    models_dir = fake_home / "whisper.cpp" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "ggml-tiny.bin").write_bytes(b"fake")
    path = whisper_engine.get_whisper_model_path("tiny")
    assert path == str(models_dir / "ggml-tiny.bin")


def test_get_whisper_models_info_marks_downloaded_and_lists_extras(fake_home):
    models_dir = fake_home / "whisper.cpp" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "ggml-tiny.bin").write_bytes(b"fake")
    (models_dir / "ggml-my-custom-finetune.bin").write_bytes(b"fake")

    info = whisper_engine.get_whisper_models_info()
    by_name = {m["name"]: m for m in info}

    assert by_name["tiny"]["downloaded"] is True
    assert by_name["base"]["downloaded"] is False
    assert "my-custom-finetune" in by_name
    assert by_name["my-custom-finetune"]["downloaded"] is True


# ---------------------------------------------------------------------
# download_whisper_model
# ---------------------------------------------------------------------

def test_download_whisper_model_missing_script(fake_home):
    success, error = whisper_engine.download_whisper_model("tiny")
    assert success is False
    assert "download-ggml-model.sh" in error


def test_download_whisper_model_success(fake_home, monkeypatch):
    models_dir = fake_home / "whisper.cpp" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "download-ggml-model.sh").write_bytes(b"#!/bin/bash\n")

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    success, error = whisper_engine.download_whisper_model("tiny")
    assert success is True
    assert error is None


def test_download_whisper_model_reports_stderr_on_failure(fake_home, monkeypatch):
    models_dir = fake_home / "whisper.cpp" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "download-ggml-model.sh").write_bytes(b"#!/bin/bash\n")

    def fake_run(*a, **k):
        raise subprocess.CalledProcessError(
            1, a[0], output="", stderr="curl: (6) Could not resolve host"
        )
    monkeypatch.setattr(subprocess, "run", fake_run)

    success, error = whisper_engine.download_whisper_model("tiny")
    assert success is False
    assert "Could not resolve host" in error


def test_download_whisper_model_falls_back_to_stdout_when_no_stderr(fake_home, monkeypatch):
    models_dir = fake_home / "whisper.cpp" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "download-ggml-model.sh").write_bytes(b"#!/bin/bash\n")

    def fake_run(*a, **k):
        raise subprocess.CalledProcessError(1, a[0], output="Invalid model name: bogus", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    success, error = whisper_engine.download_whisper_model("bogus")
    assert success is False
    assert "Invalid model name" in error


# ---------------------------------------------------------------------
# transcribe_cli
# ---------------------------------------------------------------------

class _FakeQueueWorker:
    stop_current = False


@pytest.fixture
def cli_ready_home(fake_home):
    """A fake ~/whisper.cpp with a model and the whisper-cli binary present,
    so transcribe_cli gets past its existence checks."""
    models_dir = fake_home / "whisper.cpp" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "ggml-tiny.bin").write_bytes(b"fake")
    bin_dir = fake_home / "whisper.cpp" / "build" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "whisper-cli").write_bytes(b"fake")
    return fake_home


def test_transcribe_cli_model_missing_and_download_fails(fake_home, tmp_path, monkeypatch):
    logs = []
    monkeypatch.setattr(
        whisper_engine, "download_whisper_model", lambda name: (False, "no network")
    )
    result = whisper_engine.transcribe_cli("audio.wav", "tiny", str(tmp_path), logs.append)
    assert result is None
    assert any("no network" in line for line in logs)


def test_transcribe_cli_binary_missing(fake_home, tmp_path):
    models_dir = fake_home / "whisper.cpp" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "ggml-tiny.bin").write_bytes(b"fake")
    logs = []
    result = whisper_engine.transcribe_cli("audio.wav", "tiny", str(tmp_path), logs.append)
    assert result is None
    assert any("whisper-cli not found" in line for line in logs)


def test_transcribe_cli_success(cli_ready_home, tmp_path, monkeypatch):
    (tmp_path / "meeting.txt").write_text("hello world transcript")

    class FakeProc:
        returncode = 0
        def poll(self):
            return 0  # finished immediately
        def communicate(self):
            return ("", "")

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc())
    logs = []
    result = whisper_engine.transcribe_cli("audio.wav", "tiny", str(tmp_path), logs.append)
    assert result == "hello world transcript"


def test_transcribe_cli_nonzero_exit_reports_stderr(cli_ready_home, tmp_path, monkeypatch):
    class FakeProc:
        returncode = 1
        def poll(self):
            return 1
        def communicate(self):
            return ("", "segfault or similar")

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc())
    logs = []
    result = whisper_engine.transcribe_cli("audio.wav", "tiny", str(tmp_path), logs.append)
    assert result is None
    assert any("segfault" in line for line in logs)


def test_transcribe_cli_cancel_terminates_process(cli_ready_home, tmp_path, monkeypatch):
    class FakeProc:
        def __init__(self):
            self.terminated = False
        def poll(self):
            return None  # never finishes on its own
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return 0
        def communicate(self):
            return ("", "")

    fake_proc = FakeProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake_proc)

    qw = _FakeQueueWorker()
    calls = {"n": 0}

    def fake_sleep(s):
        calls["n"] += 1
        if calls["n"] == 1:
            qw.stop_current = True
        if calls["n"] > 5:
            raise AssertionError("poll loop did not exit after cancellation")

    monkeypatch.setattr(whisper_engine.time, "sleep", fake_sleep)

    logs = []
    result = whisper_engine.transcribe_cli("audio.wav", "tiny", str(tmp_path), logs.append, qw)
    assert result is None
    assert fake_proc.terminated is True
    assert any("Cancelled" in line for line in logs)


# ---------------------------------------------------------------------
# transcribe_faster
# ---------------------------------------------------------------------

def test_transcribe_faster_falls_back_to_cli_if_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    called = {"used_cli": False}

    def fake_transcribe_cli(*a, **k):
        called["used_cli"] = True
        return "cli transcript"

    monkeypatch.setattr(whisper_engine, "transcribe_cli", fake_transcribe_cli)
    logs = []
    result = whisper_engine.transcribe_faster("audio.wav", "tiny", str(tmp_path), logs.append)
    assert result == "cli transcript"
    assert called["used_cli"] is True


def test_transcribe_faster_success(tmp_path, monkeypatch):
    class FakeSegment:
        def __init__(self, text):
            self.text = text

    class FakeInfo:
        language = "en"
        language_probability = 0.98

    class FakeModel:
        def __init__(self, *a, **k):
            pass
        def transcribe(self, audio_file, beam_size=5):
            return [FakeSegment(" hello "), FakeSegment(" world ")], FakeInfo()

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)

    logs = []
    result = whisper_engine.transcribe_faster("audio.wav", "tiny", str(tmp_path), logs.append)
    # transcribe_faster joins raw (unstripped) segment texts with " ", so
    # " hello " + " " + " world " has three spaces in the middle.
    assert result == " hello   world "
    assert (tmp_path / "transcript.txt").read_text(encoding="utf-8") == result


def test_transcribe_faster_cancel_stops_mid_segments(tmp_path, monkeypatch):
    class FakeSegment:
        def __init__(self, text):
            self.text = text

    class FakeInfo:
        language = "en"
        language_probability = 0.9

    class FakeModel:
        def __init__(self, *a, **k):
            pass
        def transcribe(self, audio_file, beam_size=5):
            return [FakeSegment("one"), FakeSegment("two"), FakeSegment("three")], FakeInfo()

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)

    qw = _FakeQueueWorker()
    qw.stop_current = True  # already cancelled before the first segment
    logs = []
    result = whisper_engine.transcribe_faster("audio.wav", "tiny", str(tmp_path), logs.append, qw)
    assert result is None


# ---------------------------------------------------------------------
# transcribe_faster_with_segments
# ---------------------------------------------------------------------

def test_transcribe_faster_with_segments_falls_back_to_none_if_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    logs = []
    transcript, segments = whisper_engine.transcribe_faster_with_segments("audio.wav", "tiny", str(tmp_path), logs.append)
    assert transcript is None
    assert segments is None
    assert any("faster-whisper not installed" in line for line in logs)


def test_transcribe_faster_with_segments_returns_timestamps(tmp_path, monkeypatch):
    class FakeSegment:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    class FakeInfo:
        language = "en"
        language_probability = 0.98

    class FakeModel:
        def __init__(self, *a, **k):
            pass
        def transcribe(self, audio_file, beam_size=5):
            return [FakeSegment("hello", 0.0, 1.0), FakeSegment("world", 1.0, 2.0)], FakeInfo()

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)

    logs = []
    transcript, segments = whisper_engine.transcribe_faster_with_segments("audio.wav", "tiny", str(tmp_path), logs.append)
    assert transcript == "hello world"
    assert segments == [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
    ]
    assert (tmp_path / "transcript.txt").read_text(encoding="utf-8") == transcript


def test_transcribe_faster_with_segments_cancel_stops_mid_segments(tmp_path, monkeypatch):
    class FakeSegment:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    class FakeInfo:
        language = "en"
        language_probability = 0.9

    class FakeModel:
        def __init__(self, *a, **k):
            pass
        def transcribe(self, audio_file, beam_size=5):
            return [FakeSegment("one", 0, 1), FakeSegment("two", 1, 2)], FakeInfo()

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)

    qw = _FakeQueueWorker()
    qw.stop_current = True
    logs = []
    transcript, segments = whisper_engine.transcribe_faster_with_segments("audio.wav", "tiny", str(tmp_path), logs.append, qw)
    assert transcript is None
    assert segments is None
