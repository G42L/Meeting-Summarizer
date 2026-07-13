"""
Tests for the UI-layer logic in main.py that's meaningfully testable
without a real display -- conftest.py already sets QT_QPA_PLATFORM=offscreen
and provides a session-scoped QApplication, so real widgets (MainWindow,
MuteButton, etc.) can be constructed directly. Nothing here opens a real
audio device, hits a real LLM server, or touches the filesystem outside
tmp_path/monkeypatch.
"""
from pathlib import Path

import pytest

from src import main


@pytest.fixture
def main_window():
    """
    MainWindow.__init__ starts a real QThread running QueueWorker.run()'s
    poll loop. Without closing the window afterward (which is what
    closeEvent's mixer.shutdown()/queue_worker.stop()/queue_thread.wait()
    normally does), that thread is left running forever -- with enough
    leaked threads across a test session this reliably crashes the
    interpreter (verified directly: a fatal abort inside pipeline.py's
    run() during teardown). This fixture makes sure every test's window
    is torn down the same way the app itself would on exit.
    """
    win = main.MainWindow()
    yield win
    win.close()


# ---------------------------------------------------------------------
# MuteButton -- short click vs. long press state machine
#
# Real mouse events + a real 500ms timer would make these slow and
# timing-flaky, so transitions are exercised through the same internal
# hooks the mouse handlers call (_set_mode / _trigger_long_press), which
# is exactly the state-machine logic being tested -- mousePressEvent/
# mouseReleaseEvent are just plumbing that decides *which* of these to
# call and are simple enough to read by inspection.
# ---------------------------------------------------------------------

def test_mute_button_starts_unmuted():
    btn = main.MuteButton()
    assert btn._mode == "none"


def test_mute_button_short_click_toggles_none_and_short():
    btn = main.MuteButton()
    btn._set_mode("short" if btn._mode == "none" else "none")
    assert btn._mode == "short"
    btn._set_mode("short" if btn._mode == "none" else "none")
    assert btn._mode == "none"


def test_mute_button_long_press_toggles_none_and_long():
    btn = main.MuteButton()
    btn._trigger_long_press()
    assert btn._mode == "long"
    btn._trigger_long_press()
    assert btn._mode == "none"


def test_mute_button_short_click_while_long_muted_unmutes_first():
    btn = main.MuteButton()
    btn._trigger_long_press()
    assert btn._mode == "long"
    btn._set_mode("short" if btn._mode == "none" else "none")
    assert btn._mode == "none"


def test_mute_button_long_press_while_short_muted_switches_directly():
    btn = main.MuteButton()
    btn._set_mode("short")
    btn._trigger_long_press()
    assert btn._mode == "long"


def test_mute_button_mode_changed_signal_emits_new_mode():
    btn = main.MuteButton()
    seen = []
    btn.mode_changed.connect(seen.append)
    btn._set_mode("short")
    btn._trigger_long_press()
    assert seen == ["short", "long"]


def test_mute_button_icon_and_color_defined_for_every_mode():
    for mode in ("none", "short", "long"):
        assert mode in main.MuteButton.ICON_FOR_MODE
        assert mode in main.MuteButton.COLOR_FOR_MODE


def test_source_row_relays_mute_mode_as_muted_and_full_mute_flags():
    row = main.SourceRow("mic-1", "Mic 1")
    received = []
    row.mute_changed.connect(lambda name, muted, full: received.append((name, muted, full)))

    row._on_mute_mode_changed("short")
    row._on_mute_mode_changed("long")
    row._on_mute_mode_changed("none")

    assert received == [
        ("mic-1", True, True),    # short press -> muted + full_mute
        ("mic-1", True, False),   # long press -> muted, VU meter stays active
        ("mic-1", False, False),  # back to unmuted
    ]


# ---------------------------------------------------------------------
# themed_icon / LOG_ICONS / LOG_ICON_COLORS
# ---------------------------------------------------------------------

def test_themed_icon_returns_non_null_icon():
    from PyQt5.QtGui import QColor
    icon = main.themed_icon("check-circle-outside", QColor("#1e8e3e"), size=16)
    assert not icon.isNull()
    pixmap = icon.pixmap(16, 16)
    assert not pixmap.isNull()


def test_every_log_icon_has_a_color_mapped():
    """Guards against a future LOG_ICONS entry silently rendering with no
    color (themed_icon would KeyError, but only at render time -- this
    catches it immediately instead)."""
    for icon_name in set(main.LOG_ICONS.values()):
        assert icon_name in main.LOG_ICON_COLORS, f"{icon_name!r} has no LOG_ICON_COLORS entry"


def test_log_icon_svg_files_exist_on_disk():
    for icon_name in set(main.LOG_ICONS.values()):
        assert (Path(main.ICONS_DIR) / f"{icon_name}.svg").exists()


# ---------------------------------------------------------------------
# FlatIconButton / HoverColorIconButton
# ---------------------------------------------------------------------

def test_hover_color_icon_button_swaps_icon_on_hover():
    from PyQt5.QtCore import QEvent

    btn = main.HoverColorIconButton("cross-circle", hover_color="#d93025")
    normal_icon = btn.icon()

    btn.enterEvent(QEvent(QEvent.Enter))
    hover_icon = btn.icon()
    assert hover_icon.cacheKey() != normal_icon.cacheKey()

    btn.leaveEvent(QEvent(QEvent.Leave))
    assert btn.icon().cacheKey() == normal_icon.cacheKey()


def test_flat_icon_button_has_no_border_styling():
    btn = main.FlatIconButton()
    assert "border: none" in btn.styleSheet()


# ---------------------------------------------------------------------
# MainWindow.open_folder -- always the general transcripts root, never the
# last job's specific session folder (that's the History sidebar's job).
# ---------------------------------------------------------------------

def test_open_folder_opens_transcripts_root_not_last_session(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / "transcripts" / "2026-01-01 10.00.00"
    session_dir.mkdir(parents=True)
    (session_dir / "summary.md").write_text("fake")

    win = main_window
    win.last_md_path = str(session_dir / "summary.md")

    captured = {}
    win._open_path = lambda path: captured.setdefault("path", path)
    win.open_folder()

    assert captured["path"] == str(tmp_path / "transcripts")


def test_open_folder_falls_back_to_cwd_when_transcripts_dir_missing(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    win = main_window
    win.last_md_path = None

    captured = {}
    win._open_path = lambda path: captured.setdefault("path", path)
    win.open_folder()

    assert captured["path"] == str(tmp_path)


# ---------------------------------------------------------------------
# Whisper-cli / diarization / HF token mutual exclusion
# (on_use_cli_toggled, on_diarization_toggled)
# ---------------------------------------------------------------------

def test_diarization_toggle_enables_hf_token_field_and_label(main_window):
    win = main_window
    assert win.hf_token_edit.isEnabled() is False
    assert win.hf_token_label.isEnabled() is False

    win.diarization_check.setChecked(True)
    assert win.hf_token_edit.isEnabled() is True
    assert win.hf_token_label.isEnabled() is True

    win.diarization_check.setChecked(False)
    assert win.hf_token_edit.isEnabled() is False
    assert win.hf_token_label.isEnabled() is False


def test_use_cli_toggled_disables_diarization_and_token_while_checked(main_window):
    win = main_window
    win.diarization_check.setChecked(True)

    win.use_cli_check.setChecked(True)
    assert win.diarization_check.isChecked() is False
    assert win.diarization_check.isEnabled() is False
    assert win.hf_token_edit.isEnabled() is False
    assert win.hf_token_label.isEnabled() is False


def test_use_cli_toggled_off_restores_diarization_and_token_if_previously_checked(main_window):
    win = main_window
    win.diarization_check.setChecked(True)
    win.use_cli_check.setChecked(True)

    win.use_cli_check.setChecked(False)
    assert win.diarization_check.isChecked() is True
    assert win.diarization_check.isEnabled() is True
    assert win.hf_token_edit.isEnabled() is True
    assert win.hf_token_label.isEnabled() is True


def test_use_cli_toggled_off_leaves_diarization_off_if_it_was_off(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    win = main_window
    assert win.diarization_check.isChecked() is False

    win.use_cli_check.setChecked(True)
    win.use_cli_check.setChecked(False)

    assert win.diarization_check.isChecked() is False
    assert win.hf_token_edit.isEnabled() is False
    assert win.hf_token_label.isEnabled() is False


# ---------------------------------------------------------------------
# Summary Style folded into the LLM Backend group (no separate group box)
# ---------------------------------------------------------------------

def test_summary_style_has_no_own_group_box(main_window):
    assert not hasattr(main_window, "prompt_style_group")


def test_prompt_style_combo_lives_inside_llm_group(main_window):
    win = main_window
    assert win.prompt_style_combo.window() is win
    # Walk up the parent chain and confirm llm_group is an ancestor.
    widget = win.prompt_style_combo.parentWidget()
    ancestors = []
    while widget is not None:
        ancestors.append(widget)
        widget = widget.parentWidget()
    assert win.llm_group in ancestors
