"""
Tests for tts_engine.py. The real Piper voice download/synthesis flow isn't
exercised here (no network, no onnxruntime model load in CI) -- is_available()
and synthesize() are tested only for their graceful-degradation paths.
"""
import sys

from src import tts_engine


# ---------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------

def test_is_available_false_when_piper_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "piper", None)
    assert tts_engine.is_available() is False


# ---------------------------------------------------------------------
# get_voice_path / get_tts_voices_info
# ---------------------------------------------------------------------

def test_get_voice_path_none_when_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_engine, "VOICES_DIR", tmp_path)
    assert tts_engine.get_voice_path("en_US-lessac-medium") is None


def test_get_voice_path_found_when_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_engine, "VOICES_DIR", tmp_path)
    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"fake")
    assert tts_engine.get_voice_path("en_US-lessac-medium") == tmp_path / "en_US-lessac-medium.onnx"


def test_get_tts_voices_info_reports_downloaded_state(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_engine, "VOICES_DIR", tmp_path)
    (tmp_path / "en_US-amy-medium.onnx").write_bytes(b"fake")

    info = {v["name"]: v for v in tts_engine.get_tts_voices_info()}
    assert info["en_US-amy-medium"]["downloaded"] is True
    assert info["en_US-lessac-medium"]["downloaded"] is False


# ---------------------------------------------------------------------
# synthesize -- graceful failure paths only (no real model here)
# ---------------------------------------------------------------------

def test_synthesize_returns_none_when_piper_not_installed(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "piper", None)
    logs = []
    result = tts_engine.synthesize("hello", "en_US-lessac-medium", tmp_path / "out.wav", logs.append)
    assert result is None
    assert any("not installed" in line for line in logs)


def test_synthesize_returns_none_when_voice_download_fails(monkeypatch, tmp_path):
    import types
    fake_module = types.ModuleType("piper")
    fake_module.PiperVoice = object()
    monkeypatch.setitem(sys.modules, "piper", fake_module)
    monkeypatch.setattr(tts_engine, "VOICES_DIR", tmp_path)
    monkeypatch.setattr(tts_engine, "download_voice", lambda voice_name, log: (False, "network error"))

    logs = []
    result = tts_engine.synthesize("hello", "en_US-lessac-medium", tmp_path / "out.wav", logs.append)
    assert result is None
    assert any("Failed to download voice" in line for line in logs)
