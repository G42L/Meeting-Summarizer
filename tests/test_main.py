"""
Tests for the UI-layer logic in main.py that's meaningfully testable
without a real display -- conftest.py already sets QT_QPA_PLATFORM=offscreen
and provides a session-scoped QApplication, so real widgets (MainWindow,
MuteButton, etc.) can be constructed directly. Nothing here opens a real
audio device, hits a real LLM server, or touches the filesystem outside
tmp_path/monkeypatch.
"""
import random
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMessageBox, QInputDialog, QFileDialog, QApplication

from src import main, vu_meters


def _isolate_qsettings(monkeypatch, tmp_path):
    """
    Redirects every QSettings("MeetingTranscriber", "MeetingTranscriber")
    call inside main.py to a throwaway ini file for the duration of one
    test, instead of the real ~/.config store -- needed by any test that
    might trigger a settings *write* (e.g. accepting the HF token prompt),
    since main.MainWindow.closeEvent() unconditionally saves diarization/
    prompt-style settings on teardown, and an unisolated write here would
    leak into the developer's real config and potentially poison whatever
    test runs next (a fresh MainWindow() loads real settings back in).
    """
    settings_path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(main, "QSettings", lambda *a, **k: QSettings(settings_path, QSettings.Format.IniFormat))


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
    from PyQt6.QtGui import QColor
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
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QEnterEvent

    btn = main.HoverColorIconButton("cross-circle", hover_color="#d93025")
    normal_icon = btn.icon()

    btn.enterEvent(QEnterEvent(QPointF(0, 0), QPointF(0, 0), QPointF(0, 0)))
    hover_icon = btn.icon()
    assert hover_icon.cacheKey() != normal_icon.cacheKey()

    btn.leaveEvent(QEvent(QEvent.Type.Leave))
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

# These four tests are only exercising the enable/disable *wiring* between
# diarization_check / use_cli_check / the HF token field -- not the
# model-availability check + token-prompt dance in on_diarization_toggled
# (covered separately below), which would otherwise pop a real blocking
# QInputDialog in a headless test run. is_available() is patched False so
# checking the box is a no-op past the enable/disable toggling.

def test_diarization_toggle_enables_hf_token_field_and_label(main_window, monkeypatch):
    win = main_window
    monkeypatch.setattr(main.diarization, "is_available", lambda: False)
    assert win.hf_token_edit.isEnabled() is False
    assert win.hf_token_label.isEnabled() is False

    win.diarization_check.setChecked(True)
    assert win.hf_token_edit.isEnabled() is True
    assert win.hf_token_label.isEnabled() is True

    win.diarization_check.setChecked(False)
    assert win.hf_token_edit.isEnabled() is False
    assert win.hf_token_label.isEnabled() is False


def test_use_cli_toggled_disables_diarization_and_token_while_checked(main_window, monkeypatch):
    win = main_window
    monkeypatch.setattr(main.diarization, "is_available", lambda: False)
    win.diarization_check.setChecked(True)

    win.use_cli_check.setChecked(True)
    assert win.diarization_check.isChecked() is False
    assert win.diarization_check.isEnabled() is False
    assert win.hf_token_edit.isEnabled() is False
    assert win.hf_token_label.isEnabled() is False


def test_use_cli_toggled_off_restores_diarization_and_token_if_previously_checked(main_window, monkeypatch):
    win = main_window
    monkeypatch.setattr(main.diarization, "is_available", lambda: False)
    win.diarization_check.setChecked(True)
    win.use_cli_check.setChecked(True)

    win.use_cli_check.setChecked(False)
    assert win.diarization_check.isChecked() is True
    assert win.diarization_check.isEnabled() is True
    assert win.hf_token_edit.isEnabled() is True
    assert win.hf_token_label.isEnabled() is True


def test_use_cli_toggled_off_leaves_diarization_off_if_it_was_off(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main.diarization, "is_available", lambda: False)
    win = main_window
    assert win.diarization_check.isChecked() is False

    win.use_cli_check.setChecked(True)
    win.use_cli_check.setChecked(False)

    assert win.diarization_check.isChecked() is False
    assert win.hf_token_edit.isEnabled() is False
    assert win.hf_token_label.isEnabled() is False


# ---------------------------------------------------------------------
# on_diarization_toggled -- model-availability check + token-prompt flow
# ---------------------------------------------------------------------

def test_diarization_toggle_skips_prompt_when_model_already_cached(main_window, monkeypatch, tmp_path):
    _isolate_qsettings(monkeypatch, tmp_path)
    win = main_window
    win.hf_token_edit.setText("")  # a prior test's leftover token, if any, must not affect this one
    monkeypatch.setattr(main.diarization, "is_available", lambda: True)
    monkeypatch.setattr(main.diarization, "is_model_cached_locally", lambda: True)
    prompted = []
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: prompted.append(1) or ("", False))

    win.diarization_check.setChecked(True)

    assert not prompted
    assert win.diarization_check.isChecked() is True


def test_diarization_toggle_prompts_when_model_not_cached(main_window, monkeypatch, tmp_path):
    _isolate_qsettings(monkeypatch, tmp_path)
    win = main_window
    win.hf_token_edit.setText("")
    monkeypatch.setattr(main.diarization, "is_available", lambda: True)
    monkeypatch.setattr(main.diarization, "is_model_cached_locally", lambda: False)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("hf_typedtoken", True))

    win.diarization_check.setChecked(True)

    assert win.diarization_check.isChecked() is True
    assert win.hf_token_edit.text() == "hf_typedtoken"
    win.hf_token_edit.setText("")  # don't leak this token into whatever closeEvent() saves on teardown


def test_diarization_toggle_unchecks_when_prompt_cancelled(main_window, monkeypatch, tmp_path):
    _isolate_qsettings(monkeypatch, tmp_path)
    win = main_window
    win.hf_token_edit.setText("")
    monkeypatch.setattr(main.diarization, "is_available", lambda: True)
    monkeypatch.setattr(main.diarization, "is_model_cached_locally", lambda: False)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))

    win.diarization_check.setChecked(True)

    assert win.diarization_check.isChecked() is False


def test_diarization_toggle_skips_prompt_when_token_already_on_file(main_window, monkeypatch, tmp_path):
    _isolate_qsettings(monkeypatch, tmp_path)
    win = main_window
    win.hf_token_edit.setText("hf_alreadyset")
    monkeypatch.setattr(main.diarization, "is_available", lambda: True)
    # is_model_cached_locally deliberately not patched -- if it were called
    # despite a token already being present, this would blow up, proving
    # the early-return happens before that check.
    monkeypatch.setattr(
        main.diarization, "is_model_cached_locally",
        lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    prompted = []
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: prompted.append(1) or ("", False))

    win.diarization_check.setChecked(True)

    assert not prompted
    assert win.diarization_check.isChecked() is True
    win.hf_token_edit.setText("")  # don't leak this token into whatever closeEvent() saves on teardown


def test_diarization_toggle_skips_check_when_pyannote_not_available(main_window, monkeypatch, tmp_path):
    _isolate_qsettings(monkeypatch, tmp_path)
    win = main_window
    win.hf_token_edit.setText("")
    monkeypatch.setattr(main.diarization, "is_available", lambda: False)
    prompted = []
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: prompted.append(1) or ("", False))

    win.diarization_check.setChecked(True)

    assert not prompted
    assert win.diarization_check.isChecked() is True


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


# ---------------------------------------------------------------------
# Icons/widgets refreshing on OS theme (palette) change while running.
#
# Two independent bugs lived here: our baked icon *bitmaps* (themed_icon()/
# MainWindow._icon()) never got re-rendered on a theme flip, since nothing
# listened for QEvent.Type.PaletteChange; and, less obviously, BASE_STYLESHEET's
# palette(...) references turned out NOT to re-evaluate live despite
# looking dynamic -- Qt's QSS engine caches the resolved color per widget
# and only recomputes it on an explicit repolish (verified directly via
# screenshots: QPushButton/QComboBox/QLineEdit kept the *old* theme's
# colors indefinitely after a live palette change, while natively-drawn
# widgets like QGroupBox's fill updated fine on their own). These tests
# call changeEvent() directly (after actually changing the widget's
# palette) rather than relying on the platform to deliver a real
# theme-change event, since that's the untestable, OS-specific part --
# what's being tested here is our own response to the event, not Qt's
# delivery of it.
# ---------------------------------------------------------------------

def _first_opaque_pixel_color(icon, size=18):
    """Fully-opaque (alpha==255) pixel specifically, not just alpha>0 --
    anti-aliased edge pixels are partially blended with the (transparent)
    background and don't reflect the tint color exactly."""
    image = icon.pixmap(size, size).toImage()
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() == 255:
                return color
    return None


def test_icon_refreshes_on_palette_change(main_window):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QColor, QPalette

    win = main_window
    old_icon = win.load_btn.icon()

    new_color = QColor(10, 200, 10)  # distinct from whatever the default was
    new_palette = QPalette(win.palette())
    new_palette.setColor(QPalette.ColorRole.WindowText, new_color)
    win.setPalette(new_palette)
    win.changeEvent(QEvent(QEvent.Type.PaletteChange))

    new_icon = win.load_btn.icon()
    assert new_icon.cacheKey() != old_icon.cacheKey()
    sampled = _first_opaque_pixel_color(new_icon)
    assert sampled is not None
    assert (sampled.red(), sampled.green(), sampled.blue()) == (10, 200, 10)


def test_record_btn_icon_survives_palette_change_while_recording(main_window):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QColor, QPalette

    win = main_window
    win._record_icon_name = "stop-circle"  # simulate the mid-recording state
    win.record_btn.setIcon(win._icon("stop-circle"))

    new_palette = QPalette(win.palette())
    new_palette.setColor(QPalette.ColorRole.WindowText, QColor(5, 5, 5))
    win.setPalette(new_palette)
    win.changeEvent(QEvent(QEvent.Type.PaletteChange))

    assert win._record_icon_name == "stop-circle"
    expected_image = win._icon("stop-circle").pixmap(18, 18).toImage()
    actual_image = win.record_btn.icon().pixmap(18, 18).toImage()
    assert actual_image == expected_image


def test_stylesheet_button_color_updates_on_palette_change(main_window):
    """Regression test for the QSS-caching bug: a QPushButton's
    `background-color: palette(button)` rule must actually repaint with
    the new color after a live theme flip, not just the icon on top of
    it. Without MainWindow._repolish_widget_tree(), this stayed stuck on
    whatever color was resolved the first time the stylesheet was
    applied."""
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QColor, QPalette
    from PyQt6.QtWidgets import QApplication

    win = main_window
    win.show()
    app = QApplication.instance()
    app.processEvents()
    original_palette = QPalette(app.palette())
    try:
        # QSS's palette(...) functions resolve against the QApplication's
        # palette, not the individual widget's -- matches what
        # apply_linux_color_scheme() actually does in production
        # (app.setPalette(...)), unlike the icon-refresh tests above which
        # only need the widget's own palette.
        new_palette = QPalette(app.palette())
        new_palette.setColor(QPalette.ColorRole.Button, QColor(1, 222, 3))
        new_palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        app.setPalette(new_palette)
        win.changeEvent(QEvent(QEvent.Type.PaletteChange))
        app.processEvents()

        # A few pixels in from the left edge, away from the rounded corner
        # clip and the 1px border, so this samples the button's actual QSS
        # fill -- which should now reflect the new palette's Button color
        # if the QSS repolish actually ran.
        image = win.clear_log_btn.grab().toImage()
        fill = image.pixelColor(10, image.height() // 2)
        assert (fill.red(), fill.green(), fill.blue()) == (1, 222, 3)
    finally:
        app.setPalette(original_palette)
        win.changeEvent(QEvent(QEvent.Type.PaletteChange))


# ---------------------------------------------------------------------
# Log console autoscroll-to-bottom.
#
# _flush_console() re-renders the whole console document on every debounced
# batch of new lines (see append_log/append_summary), which requires
# manually restoring the scroll position afterward since setHtml() doesn't
# preserve it on its own. The original implementation inferred "was the
# user at the bottom" by comparing scrollbar.value()/maximum() immediately
# around the setHtml() call -- reported unreliable on macOS specifically
# (the console appeared to jump to the top on every new line even when it
# hadn't been scrolled up), so this was replaced with a persistently
# tracked _log_autoscroll flag fed by the scrollbar's own valueChanged
# signal instead of an ad hoc before/after comparison.
# ---------------------------------------------------------------------

def test_log_stays_pinned_to_bottom_as_lines_are_appended(main_window):
    win = main_window
    win.show()
    win.resize(900, 950)
    from PyQt6.QtWidgets import QApplication
    QApplication.instance().processEvents()

    scrollbar = win.log_text.verticalScrollBar()
    for i in range(30):
        win.append_log(f"Log line {i} with enough text to require scrolling.")
        win._console_render_timer.stop()
        win._flush_console()

    assert scrollbar.value() == scrollbar.maximum()
    assert scrollbar.maximum() > 0  # sanity: content actually overflowed


def test_log_does_not_yank_view_back_down_after_manual_scroll_up(main_window):
    win = main_window
    win.show()
    win.resize(900, 950)
    from PyQt6.QtWidgets import QApplication
    QApplication.instance().processEvents()

    scrollbar = win.log_text.verticalScrollBar()
    for i in range(30):
        win.append_log(f"Log line {i} with enough text to require scrolling.")
        win._console_render_timer.stop()
        win._flush_console()

    scrollbar.setValue(0)  # simulate the user manually scrolling up to read
    assert win._log_autoscroll is False

    for i in range(30, 35):
        win.append_log(f"Log line {i} with enough text to require scrolling.")
        win._console_render_timer.stop()
        win._flush_console()

    assert scrollbar.value() == 0


def test_removed_source_row_refresher_pruned_without_breaking_others(main_window):
    from PyQt6 import sip
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QColor, QPalette

    win = main_window
    row_a = main.SourceRow("a", "a", icon=win._icon("mic"))
    row_b = main.SourceRow("b", "b", icon=win._icon("mic"))
    win._track_icon(lambda label=row_a.icon_label: label.setPixmap(win._icon("mic").pixmap(16, 16)))
    win._track_icon(row_a.remove_btn.refresh_normal_icon)
    win._track_icon(lambda label=row_b.icon_label: label.setPixmap(win._icon("mic").pixmap(16, 16)))
    win._track_icon(row_b.remove_btn.refresh_normal_icon)

    sip.delete(row_a)  # simulates the widget having actually been destroyed,
                        # like a removed SourceRow -- not just gone out of scope

    new_palette = QPalette(win.palette())
    new_palette.setColor(QPalette.ColorRole.WindowText, QColor(1, 2, 3))
    win.setPalette(new_palette)
    win.changeEvent(QEvent(QEvent.Type.PaletteChange))  # must not raise

    assert not row_b.icon_label.pixmap().isNull()
    assert not row_b.remove_btn.icon().isNull()


# ---------------------------------------------------------------------
# apply_linux_color_scheme. This sandbox turned out to genuinely have a
# working xdg-desktop-portal reporting "prefer dark" (an initial
# `busctl --user list | grep portal` check missed it, likely because the
# portal is D-Bus-activatable rather than a persistently listed service --
# a raw QDBusInterface.call confirmed the real reply), so both the real
# success path and the fallback paths (D-Bus unreachable, non-Linux) are
# actually verifiable here, not just the fallback as originally assumed
# while planning this fix.
# ---------------------------------------------------------------------

def _spy_on_set_palette(app, monkeypatch):
    """
    QApplication is a single global instance shared by the whole test
    session, so comparing whole QPalette objects before/after (rather
    than spying on the call itself) is fragile -- Qt's resolve-mask
    bookkeeping can differ between an untouched palette and one that was
    explicitly re-set to identical colors, and other tests in this file
    also touch the same global app. Spying on setPalette directly is a
    more direct, order-independent way to assert "this code path never
    tried to change the palette at all".
    """
    calls = []
    monkeypatch.setattr(app, "setPalette", lambda *a, **k: calls.append(a))
    return calls


def test_apply_linux_color_scheme_noop_when_dbus_not_connected(monkeypatch):
    from PyQt6.QtDBus import QDBusConnection
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    monkeypatch.setattr(main.sys, "platform", "linux")
    calls = _spy_on_set_palette(app, monkeypatch)

    class FakeBus:
        def isConnected(self):
            return False

    monkeypatch.setattr(QDBusConnection, "sessionBus", staticmethod(lambda: FakeBus()))

    main.apply_linux_color_scheme(app)  # must not raise

    assert calls == []


def test_apply_linux_color_scheme_is_noop_on_non_linux(monkeypatch):
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    monkeypatch.setattr(main.sys, "platform", "darwin")
    calls = _spy_on_set_palette(app, monkeypatch)

    main.apply_linux_color_scheme(app)

    assert calls == []


def test_apply_linux_color_scheme_detects_dark_via_portal(monkeypatch):
    """
    Exercises the actual success path (a portal reporting "prefer dark")
    via a mocked QDBusInterface/QDBusConnection, rather than relying on
    whatever the real host's D-Bus session happens to report -- an
    earlier version of this test asserted against the live sandbox's own
    portal state directly, which was real during development but made the
    test inherently non-portable: it failed the moment it ran on any
    other machine/CI environment without that same portal configuration.
    """
    from PyQt6 import QtDBus
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    monkeypatch.setattr(main.sys, "platform", "linux")
    original_palette = QPalette(app.palette())

    class FakeBus:
        def isConnected(self):
            return True

        def connect(self, *args, **kwargs):
            return True

    class FakeReply:
        def arguments(self):
            return [1]  # 1 == prefer dark

    class FakeInterface:
        def __init__(self, *args, **kwargs):
            pass

        def isValid(self):
            return True

        def call(self, *args, **kwargs):
            return FakeReply()

    monkeypatch.setattr(QtDBus.QDBusConnection, "sessionBus", staticmethod(lambda: FakeBus()))
    monkeypatch.setattr(QtDBus, "QDBusInterface", FakeInterface)

    try:
        main.apply_linux_color_scheme(app)
        window_text = app.palette().color(QPalette.ColorRole.WindowText)
        assert (window_text.red(), window_text.green(), window_text.blue()) == (230, 230, 230)
    finally:
        app.setPalette(original_palette)


# ---------------------------------------------------------------------
# _dark_palette's Disabled color group -- setColor(role, color) with no
# explicit ColorGroup applies to every group at once, so without
# overriding Disabled separately, setEnabled(False) widgets silently stop
# looking greyed-out the moment this palette is active (regression: the
# HF token field and "already added" source picker entries rendered
# identically whether enabled or disabled).
# ---------------------------------------------------------------------

def test_dark_palette_disabled_text_is_dimmer_than_active():
    from PyQt6.QtGui import QPalette

    palette = main._dark_palette()
    active = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText)
    disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
    assert disabled != active

    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        assert palette.color(QPalette.ColorGroup.Disabled, role) != palette.color(QPalette.ColorGroup.Active, role)


def test_hf_token_field_visibly_greys_out_under_dark_palette(main_window, monkeypatch):
    """
    Reproduces the actual bug: isEnabled() toggling correctly was never
    the problem, the rendered color not changing was. Applies the real
    dark palette to the app (as apply_linux_color_scheme does on a
    dark-mode Linux desktop) and asserts the HF token label's resolved
    text color actually differs between the disabled and enabled states.
    """
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QApplication

    monkeypatch.setattr(main.diarization, "is_available", lambda: False)

    app = QApplication.instance()
    original_palette = QPalette(app.palette())
    try:
        app.setPalette(main._dark_palette())

        win = main_window
        assert win.hf_token_label.isEnabled() is False
        assert win.hf_token_edit.isEnabled() is False
        color_disabled = win.hf_token_label.palette().color(QPalette.ColorRole.WindowText)

        win.diarization_check.setChecked(True)
        assert win.hf_token_label.isEnabled() is True
        assert win.hf_token_edit.isEnabled() is True
        color_enabled = win.hf_token_label.palette().color(QPalette.ColorRole.WindowText)

        assert color_disabled != color_enabled

        win.diarization_check.setChecked(False)
        assert win.hf_token_label.isEnabled() is False
        assert win.hf_token_edit.isEnabled() is False
    finally:
        app.setPalette(original_palette)


def test_source_picker_greys_out_already_added_device_under_dark_palette(main_window, monkeypatch):
    """
    Same regression as the HF token field, but for the "Add source"
    combo's already-added entries. Qt's item views paint a disabled item
    using QPalette::Disabled rather than any per-item color, so there's
    nothing to read off the QStandardItem itself -- the real assertion is
    that (a) refresh_source_picker() still flips ItemIsEnabled correctly
    on add/remove, and (b) the app's Disabled palette group (proven
    distinct by test_dark_palette_disabled_text_is_dimmer_than_active) is
    what the combo's view will actually paint that item with.
    """
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    original_palette = QPalette(app.palette())
    try:
        app.setPalette(main._dark_palette())

        win = main_window
        monkeypatch.setattr(
            main.audio_engine,
            "list_all_sources",
            lambda: [{"name": "Fake Mic", "is_loopback": False, "device_id": 0, "samplerate": 48000, "channels": 1, "wasapi_loopback": False}],
        )
        win.refresh_source_picker()
        model = win.source_picker_combo.model()
        assert model.item(0).isEnabled() is True

        win.mixer.sources["Fake Mic"] = object()  # simulate it having been added
        win.refresh_source_picker()
        assert model.item(0).isEnabled() is False

        win.mixer.sources.pop("Fake Mic")  # simulate it having been removed
        win.refresh_source_picker()
        assert model.item(0).isEnabled() is True
    finally:
        app.setPalette(original_palette)


# ---------------------------------------------------------------------
# VU-style switching -- switch_vu_style() drives the whole
# create/replace-widget/paint/update_level cycle. These tests exercise it
# for every registered style (vu_meters.VU_METER_STYLES), both in
# registry order and in a shuffled order (styles aren't independent of
# each other in switch_vu_style -- it tears down whatever the *previous*
# style's widget was, so the interesting failure mode is style A leaving
# something behind that breaks style B, which sequential-only testing
# can't surface).
# ---------------------------------------------------------------------

def _render(widget):
    """Force a real paintEvent (offscreen render, not repaint()'s
    visibility-gated no-op) and let any exception propagate."""
    widget.resize(300, 130)
    pixmap = QPixmap(widget.width(), widget.height())
    widget.render(pixmap)


def _assert_switched_cleanly(main_window, index):
    name, widget_cls, *_ = vu_meters.VU_METER_STYLES[index]
    assert isinstance(main_window.vumeter, widget_cls), name
    # Exactly one widget in the container -- the old one was removed, not
    # just covered up.
    assert main_window.vu_container_layout.count() == 1, name
    assert main_window.vu_container_layout.itemAt(0).widget() is main_window.vumeter
    _render(main_window.vumeter)


def test_switching_through_every_vu_style_in_sequence(main_window):
    for index in range(len(vu_meters.VU_METER_STYLES)):
        main_window.switch_vu_style(index)
        _assert_switched_cleanly(main_window, index)


def test_switching_through_every_vu_style_in_random_order(main_window):
    order = list(range(len(vu_meters.VU_METER_STYLES)))
    random.Random(20260714).shuffle(order)
    for index in order:
        main_window.switch_vu_style(index)
        _assert_switched_cleanly(main_window, index)


def test_switching_through_every_vu_style_forwards_live_audio_level(main_window, monkeypatch):
    """Functional check, not just non-crashing: with a mocked 'live
    recording' feeding a loud mono buffer through the mixer, every style
    switch must actually receive that level via update_level(), the same
    codepath switch_vu_style() uses for a real recording in progress."""
    calls = []
    for name, widget_cls, *_ in vu_meters.VU_METER_STYLES:
        original_update_level = widget_cls.update_level

        def make_wrapper(name=name, original=original_update_level):
            def wrapper(self, rms):
                calls.append((name, rms))
                return original(self, rms)
            return wrapper

        monkeypatch.setattr(widget_cls, "update_level", make_wrapper())

    loud_mono_buffer = [0.8] * 1600  # well above silence, below clipping
    monkeypatch.setattr(main_window.mixer, "get_mixed_preview", lambda: loud_mono_buffer)
    main_window.loaded_samples = None
    main_window.is_recording = True

    order = list(range(len(vu_meters.VU_METER_STYLES)))
    random.Random(42).shuffle(order)
    for index in order:
        calls.clear()
        main_window.switch_vu_style(index)
        name, widget_cls, *_ = vu_meters.VU_METER_STYLES[index]

        assert calls, f"{name} never received update_level() on switch"
        call_name, rms = calls[-1]
        assert call_name == name
        assert rms > 0.0, f"{name} got a silent level despite the mocked loud buffer"

        _render(main_window.vumeter)


# ---------------------------------------------------------------------
# History search (HistorySidebar search box + MainWindow.refresh_history)
# ---------------------------------------------------------------------

def _make_session(tmp_path, name, summary_body):
    session_dir = tmp_path / "transcripts" / name
    session_dir.mkdir(parents=True)
    (session_dir / "summary.md").write_text(
        f"# Meeting Summary\n\n## Summary\n\n{summary_body}\n\n---\n\n## Full Transcript\n\n",
        encoding="utf-8",
    )
    return session_dir


def test_refresh_history_populates_list_with_no_search_query(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    _make_session(tmp_path, "2026-01-01 10.00.00", "Discussed the roadmap.")
    _make_session(tmp_path, "2026-01-02 10.00.00", "Reviewed the budget.")

    main_window.refresh_history()

    assert main_window.history_sidebar.list_widget.count() == 2


def test_search_filters_by_text_found_only_in_summary_body(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    _make_session(tmp_path, "2026-01-01 10.00.00", "Discussed the unique roadmap keyword zzyzx.")
    _make_session(tmp_path, "2026-01-02 10.00.00", "Reviewed the budget.")

    main_window.refresh_history()
    main_window.history_sidebar.search_box.setText("zzyzx")
    main_window.history_sidebar._apply_search_filter()

    assert main_window.history_sidebar.list_widget.count() == 1
    item = main_window.history_sidebar.list_widget.item(0)
    assert item.data(main.Qt.ItemDataRole.UserRole)["timestamp"] == "2026-01-01 10.00.00"


def test_search_is_case_insensitive(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    _make_session(tmp_path, "2026-01-01 10.00.00", "Discussed the Roadmap Keyword.")

    main_window.refresh_history()
    main_window.history_sidebar.search_box.setText("ROADMAP")
    main_window.history_sidebar._apply_search_filter()

    assert main_window.history_sidebar.list_widget.count() == 1


def test_search_with_no_matches_yields_empty_list(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    _make_session(tmp_path, "2026-01-01 10.00.00", "Discussed the roadmap.")

    main_window.refresh_history()
    main_window.history_sidebar.search_box.setText("nothing matches this")
    main_window.history_sidebar._apply_search_filter()

    assert main_window.history_sidebar.list_widget.count() == 0


def test_search_debounce_timer_starts_on_text_change(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    main_window.refresh_history()

    main_window.history_sidebar.search_box.setText("a")
    assert main_window.history_sidebar._search_debounce_timer.isActive()


# ---------------------------------------------------------------------
# SummaryViewerDialog export / copy
# ---------------------------------------------------------------------

def test_export_as_writes_editor_content_including_unsaved_edit(tmp_path, monkeypatch):
    md_path = tmp_path / "summary.md"
    md_path.write_text("original content", encoding="utf-8")
    dialog = main.SummaryViewerDialog(str(md_path), "2026-01-01 10.00.00", "original content")

    dialog.editor.setPlainText("unsaved edit not yet on disk")
    out_path = tmp_path / "exported.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), ""))

    dialog.export_as()

    assert out_path.read_text(encoding="utf-8") == "unsaved edit not yet on disk"
    # original file on disk is untouched by export
    assert md_path.read_text(encoding="utf-8") == "original content"


def test_export_as_cancelled_writes_nothing(tmp_path, monkeypatch):
    md_path = tmp_path / "summary.md"
    md_path.write_text("content", encoding="utf-8")
    dialog = main.SummaryViewerDialog(str(md_path), "2026-01-01 10.00.00", "content")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    dialog.export_as()  # should not raise, nothing written

    assert list(tmp_path.iterdir()) == [md_path]


def test_export_as_shows_warning_on_write_failure(tmp_path, monkeypatch):
    md_path = tmp_path / "summary.md"
    md_path.write_text("content", encoding="utf-8")
    dialog = main.SummaryViewerDialog(str(md_path), "2026-01-01 10.00.00", "content")

    bad_path = tmp_path / "no_such_dir" / "out.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(bad_path), ""))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog.export_as()

    assert len(warnings) == 1


def test_copy_to_clipboard_sets_editor_text(tmp_path, monkeypatch):
    md_path = tmp_path / "summary.md"
    md_path.write_text("content", encoding="utf-8")
    dialog = main.SummaryViewerDialog(str(md_path), "2026-01-01 10.00.00", "content")
    dialog.editor.setPlainText("copy me")

    copied = []
    fake_clipboard = type("FakeClipboard", (), {"setText": lambda self, text: copied.append(text)})()
    monkeypatch.setattr(QApplication, "clipboard", staticmethod(lambda: fake_clipboard))

    dialog.copy_to_clipboard()

    assert copied == ["copy me"]


# ---------------------------------------------------------------------
# Session rename / delete (context menu handlers)
# ---------------------------------------------------------------------

def test_rename_history_session_writes_meta_json_and_refreshes(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    session_dir = _make_session(tmp_path, "2026-01-01 10.00.00", "Some summary.")
    main_window.refresh_history()
    session = main_window.history_sidebar.list_widget.item(0).data(main.Qt.ItemDataRole.UserRole)

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("New Name", True))
    main_window._rename_history_session(session)

    assert (session_dir / "meta.json").exists()
    item = main_window.history_sidebar.list_widget.item(0)
    assert "New Name" in item.text()


def test_rename_history_session_cancelled_writes_nothing(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    session_dir = _make_session(tmp_path, "2026-01-01 10.00.00", "Some summary.")
    main_window.refresh_history()
    session = main_window.history_sidebar.list_widget.item(0).data(main.Qt.ItemDataRole.UserRole)

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Ignored", False))
    main_window._rename_history_session(session)

    assert not (session_dir / "meta.json").exists()


def test_delete_history_session_confirmed_removes_folder(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    session_dir = _make_session(tmp_path, "2026-01-01 10.00.00", "Some summary.")
    main_window.refresh_history()
    session = main_window.history_sidebar.list_widget.item(0).data(main.Qt.ItemDataRole.UserRole)

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    main_window._delete_history_session(session)

    assert not session_dir.exists()
    assert main_window.history_sidebar.list_widget.count() == 0


def test_delete_history_session_declined_keeps_folder(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    session_dir = _make_session(tmp_path, "2026-01-01 10.00.00", "Some summary.")
    main_window.refresh_history()
    session = main_window.history_sidebar.list_widget.item(0).data(main.Qt.ItemDataRole.UserRole)

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    main_window._delete_history_session(session)

    assert session_dir.exists()
    assert main_window.history_sidebar.list_widget.count() == 1


def test_context_menu_does_nothing_on_empty_area_click(tmp_path, monkeypatch, main_window):
    monkeypatch.chdir(tmp_path)
    main_window.refresh_history()

    from PyQt6.QtCore import QPoint
    exec_calls = []
    monkeypatch.setattr(main.QMenu, "exec", lambda self, *a, **k: exec_calls.append(1))

    main_window._show_history_context_menu(QPoint(5, 5))

    assert exec_calls == []
