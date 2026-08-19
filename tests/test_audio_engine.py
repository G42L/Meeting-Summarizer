"""
Tests for audio_engine.py -- pure logic, no real audio device is opened.
AudioSource.start()/stop() are monkeypatched out wherever a source needs to
be "active" for mixer tests, since those methods talk to real hardware via
miniaudio/sounddevice.
"""
import numpy as np
import pytest

from src import audio_engine


# ---------------------------------------------------------------------
# _resample_linear
# ---------------------------------------------------------------------

def test_resample_linear_same_rate_is_passthrough():
    samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = audio_engine._resample_linear(samples, 16000, 16000)
    assert result is samples


def test_resample_linear_empty_input():
    samples = np.array([], dtype=np.float32)
    result = audio_engine._resample_linear(samples, 44100, 16000)
    assert len(result) == 0


def test_resample_linear_changes_length_proportionally():
    # 1 second of audio at 44100 Hz resampled to 16000 Hz should yield
    # roughly 16000 samples (allow rounding slack).
    samples = np.sin(np.linspace(0, 2 * np.pi * 440, 44100)).astype(np.float32)
    result = audio_engine._resample_linear(samples, 44100, 16000)
    assert abs(len(result) - 16000) <= 1


# ---------------------------------------------------------------------
# _pick_format
# ---------------------------------------------------------------------

def test_pick_format_prefers_engine_rate_and_mono():
    formats = [
        {"samplerate": 44100, "channels": 2},
        {"samplerate": 16000, "channels": 1},
        {"samplerate": 48000, "channels": 2},
    ]
    rate, channels = audio_engine._pick_format(formats)
    assert rate == audio_engine.ENGINE_SAMPLE_RATE
    assert channels == 1


def test_pick_format_falls_back_to_44100_then_first_available():
    formats = [{"samplerate": 48000, "channels": 2}, {"samplerate": 44100, "channels": 2}]
    rate, _ = audio_engine._pick_format(formats)
    assert rate == 44100

    formats = [{"samplerate": 48000, "channels": 2}]
    rate, _ = audio_engine._pick_format(formats)
    assert rate == 48000


def test_pick_format_handles_empty_list():
    rate, channels = audio_engine._pick_format([])
    assert rate == 44100
    assert channels == 2


# ---------------------------------------------------------------------
# AudioSource._handle_chunk (buffering, gain, mute, preview)
# ---------------------------------------------------------------------

@pytest.fixture
def source():
    return audio_engine.AudioSource(
        "test-source", device_id=None, samplerate=16000, channels=1, gain=2.0,
    )


def test_handle_chunk_applies_gain_and_clips(source):
    mono = np.array([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    source._handle_chunk(mono)
    buffered = source.pull_available()
    assert np.allclose(buffered, np.clip(mono * 2.0, -1.0, 1.0), atol=1e-5)


def test_handle_chunk_preview_is_ungained_and_unclipped(source):
    mono = np.array([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    source._handle_chunk(mono)
    preview = source.preview_snapshot()
    assert np.allclose(preview, mono, atol=1e-5)


def test_handle_chunk_muted_source_buffers_nothing_but_still_updates_preview(source):
    source.muted = True
    mono = np.array([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    source._handle_chunk(mono)
    assert source.pull_available().size == 0
    assert len(source.preview_snapshot()) == 4


def test_handle_chunk_updates_level(source):
    mono = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    source._handle_chunk(mono)
    assert source.level == pytest.approx(1.0, abs=1e-5)


def test_pull_available_drains_and_clears_buffer(source):
    source._handle_chunk(np.array([0.1], dtype=np.float32))
    first = source.pull_available()
    assert first.size == 1
    second = source.pull_available()
    assert second.size == 0


# ---------------------------------------------------------------------
# AudioMixerEngine
# ---------------------------------------------------------------------

@pytest.fixture
def mixer(monkeypatch):
    # AudioSource.start()/stop() would otherwise try to open a real device.
    monkeypatch.setattr(audio_engine.AudioSource, "start", lambda self: None)
    monkeypatch.setattr(audio_engine.AudioSource, "stop", lambda self: None)
    return audio_engine.AudioMixerEngine()


def _add_active_source(mixer, name, gain=1.0):
    source = mixer.add_source(name, device_id=None, samplerate=16000, channels=1, gain=gain)
    # is_active checks for a live device/stream handle; fake one so tick()
    # treats this source as running without needing a real capture device.
    source._miniaudio_device = object()
    return source


def test_add_source_duplicate_name_raises(mixer):
    _add_active_source(mixer, "mic")
    with pytest.raises(ValueError):
        _add_active_source(mixer, "mic")


def test_start_without_sources_raises(mixer):
    with pytest.raises(RuntimeError):
        mixer.start()


def test_tick_mixes_two_sources_and_pads_shorter_one(mixer):
    a = _add_active_source(mixer, "a")
    b = _add_active_source(mixer, "b")
    a._handle_chunk(np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32))
    b._handle_chunk(np.array([0.2, 0.2], dtype=np.float32))

    mixer.tick()
    preview = mixer.get_mixed_preview()
    # b's chunk is zero-padded to match a's length, then summed.
    assert preview[:2] == pytest.approx([0.3, 0.3], abs=1e-5)
    assert preview[2:4] == pytest.approx([0.1, 0.1], abs=1e-5)


def test_tick_only_accumulates_into_recording_while_running(mixer):
    a = _add_active_source(mixer, "a")
    a._handle_chunk(np.array([0.5, 0.5], dtype=np.float32))
    mixer.tick()  # not recording yet -- should NOT land in get_mixed_audio()
    assert mixer.get_mixed_audio().size == 0

    mixer.start()
    a._handle_chunk(np.array([0.5, 0.5], dtype=np.float32))
    mixer.tick()
    assert mixer.get_mixed_audio().size == 2


def test_pause_stops_accumulation_without_discarding_prior_audio(mixer):
    a = _add_active_source(mixer, "a")
    mixer.start()
    a._handle_chunk(np.array([0.5, 0.5], dtype=np.float32))
    mixer.tick()
    assert mixer.get_mixed_audio().size == 2

    mixer.pause()
    assert mixer.is_paused is True
    a._handle_chunk(np.array([0.9, 0.9, 0.9], dtype=np.float32))
    mixer.tick()  # paused -- should not land in the saved recording
    assert mixer.get_mixed_audio().size == 2


def test_resume_appends_to_previously_recorded_audio(mixer):
    a = _add_active_source(mixer, "a")
    mixer.start()
    a._handle_chunk(np.array([0.5, 0.5], dtype=np.float32))
    mixer.tick()

    mixer.pause()
    a._handle_chunk(np.array([0.9], dtype=np.float32))
    mixer.tick()  # discarded while paused

    mixer.resume()
    assert mixer.is_paused is False
    a._handle_chunk(np.array([0.1, 0.1], dtype=np.float32))
    mixer.tick()
    # 2 samples from before the pause + 2 samples from after resume, the
    # single sample ticked while paused is nowhere in there.
    assert mixer.get_mixed_audio().size == 4


def test_stop_clears_paused_flag(mixer):
    _add_active_source(mixer, "a")
    mixer.start()
    mixer.pause()
    assert mixer.is_paused is True
    mixer.stop()
    assert mixer.is_paused is False


def test_muted_source_excluded_from_mix_but_still_drained(mixer):
    a = _add_active_source(mixer, "a")
    a.muted = True
    a._handle_chunk(np.array([0.9, 0.9], dtype=np.float32))
    mixer.tick()
    assert mixer.get_mixed_preview() == []
    # the muted source's buffer should have been drained regardless, so it
    # doesn't leak into the mix once unmuted later.
    assert a.pull_available().size == 0


def test_set_gain_and_muted_helpers(mixer):
    _add_active_source(mixer, "a")
    mixer.set_gain("a", 3.0)
    assert mixer.sources["a"].gain == 3.0
    mixer.set_gain("a", -1.0)  # clamped to 0
    assert mixer.sources["a"].gain == 0.0
    mixer.set_muted("a", True)
    assert mixer.sources["a"].muted is True


# ---------------------------------------------------------------------
# full_mute -- the "short press" mode (SourceRow's MuteButton) additionally
# silences a source's own dedicated VU meter, on top of the plain `muted`
# flag (the "long press" mode) that only excludes it from the mix.
# ---------------------------------------------------------------------

def test_audio_source_full_mute_defaults_to_false():
    source = audio_engine.AudioSource("test", device_id=None, samplerate=16000, channels=1)
    assert source.full_mute is False


def test_set_muted_with_full_mute_true(mixer):
    _add_active_source(mixer, "a")
    mixer.set_muted("a", True, full_mute=True)
    assert mixer.sources["a"].muted is True
    assert mixer.sources["a"].full_mute is True


def test_set_muted_full_mute_defaults_to_false(mixer):
    _add_active_source(mixer, "a")
    mixer.set_muted("a", True)
    assert mixer.sources["a"].full_mute is False


def test_set_muted_unmuting_forces_full_mute_off(mixer):
    _add_active_source(mixer, "a")
    mixer.set_muted("a", True, full_mute=True)
    mixer.set_muted("a", False, full_mute=True)  # unmuting -- full_mute should not stick
    assert mixer.sources["a"].muted is False
    assert mixer.sources["a"].full_mute is False


def test_get_source_level_returns_zero_when_full_mute(mixer):
    a = _add_active_source(mixer, "a")
    a.level = 0.75
    mixer.set_muted("a", True, full_mute=True)
    assert mixer.get_source_level("a") == 0.0


def test_get_source_level_stays_real_when_muted_but_not_full_mute(mixer):
    """Long-press mode: excluded from the mix, but the row's own VU meter
    should keep showing real activity."""
    a = _add_active_source(mixer, "a")
    a.level = 0.75
    mixer.set_muted("a", True, full_mute=False)
    assert mixer.get_source_level("a") == 0.75


def test_get_source_level_unknown_source_returns_zero(mixer):
    assert mixer.get_source_level("does-not-exist") == 0.0


def test_get_source_errors_only_reports_sources_with_errors(mixer):
    a = _add_active_source(mixer, "a")
    _add_active_source(mixer, "b")
    a.error = "device disconnected"
    errors = mixer.get_source_errors()
    assert errors == {"a": "device disconnected"}


def test_remove_source_stops_and_drops_it(mixer):
    _add_active_source(mixer, "a")
    mixer.remove_source("a")
    assert "a" not in mixer.sources
