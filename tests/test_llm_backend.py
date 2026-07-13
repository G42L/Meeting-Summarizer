"""
Tests for llm_backend.py. All network I/O (requests.get/requests.post) is
mocked -- these tests never touch a real LLM server.
"""
import json

import pytest
import requests

from src import llm_backend


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

class FakeGetResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


class FakeStreamResponse:
    """Mimics `with requests.post(..., stream=True) as r:` usage."""
    def __init__(self, status_code=200, lines=(), text=""):
        self.status_code = status_code
        self._lines = lines
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        for line in self._lines:
            yield line.encode() if isinstance(line, str) else line


class FakeQueueWorker:
    stop_current = False


# ---------------------------------------------------------------------
# detect_backends
# ---------------------------------------------------------------------

def test_detect_backends_ollama_success(monkeypatch):
    def fake_get(url, timeout=None):
        if "11434" in url:
            return FakeGetResponse(200, {"models": [{"name": "llama3"}]})
        return FakeGetResponse(500)

    monkeypatch.setattr(requests, "get", fake_get)
    logs = []
    backends = llm_backend.detect_backends(log=logs.append)
    assert "Ollama" in backends
    assert backends["Ollama"]["models"] == ["llama3"]
    assert any("Ollama detected" in line for line in logs)


def test_detect_backends_unreachable_is_silent_without_log():
    def fake_get(url, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    import requests as req
    orig = req.get
    req.get = fake_get
    try:
        backends = llm_backend.detect_backends()  # no log callback
        assert backends == {}
    finally:
        req.get = orig


def test_detect_backends_vllm_openai_style(monkeypatch):
    def fake_get(url, timeout=None):
        if "8000" in url:
            return FakeGetResponse(200, {"data": [{"id": "mistral-7b"}]})
        return FakeGetResponse(500)

    monkeypatch.setattr(requests, "get", fake_get)
    backends = llm_backend.detect_backends()
    assert backends["vLLM"]["models"] == ["mistral-7b"]
    assert backends["vLLM"]["api_type"] == "openai"


def test_lm_studio_models_prefers_native_endpoint(monkeypatch):
    def fake_get(url, timeout=None):
        if "/api/v0/models" in url:
            return FakeGetResponse(200, {"data": [
                {"id": "loaded-model", "state": "loaded"},
                {"id": "unloaded-model", "state": "not-loaded"},
            ]})
        raise AssertionError("should not fall back to /v1/models when native succeeds")

    monkeypatch.setattr(requests, "get", fake_get)
    all_models, used_native = llm_backend._lm_studio_models("http://localhost:1234")
    assert used_native is True
    assert {"id": "loaded-model", "usable": True} in all_models
    assert {"id": "unloaded-model", "usable": False} in all_models


def test_lm_studio_models_falls_back_to_v1_models(monkeypatch):
    def fake_get(url, timeout=None):
        if "/api/v0/models" in url:
            return FakeGetResponse(404)
        if "/v1/models" in url:
            return FakeGetResponse(200, {"data": [{"id": "some-model"}]})
        raise AssertionError("unexpected URL")

    monkeypatch.setattr(requests, "get", fake_get)
    all_models, used_native = llm_backend._lm_studio_models("http://localhost:1234")
    assert used_native is False
    assert all_models == [{"id": "some-model", "usable": True}]


def test_lm_studio_models_both_endpoints_fail_raises(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: FakeGetResponse(500))
    with pytest.raises(RuntimeError):
        llm_backend._lm_studio_models("http://localhost:1234")


# ---------------------------------------------------------------------
# summarize dispatch
# ---------------------------------------------------------------------

def test_summarize_unknown_api_type_logs_and_returns_none():
    logs = []
    result = llm_backend.summarize(
        "transcript", {"url": "http://x", "api_type": "carrier-pigeon"}, "model",
        on_chunk=lambda c: None, log=logs.append,
    )
    assert result is None
    assert any("Unknown API type" in line for line in logs)


def test_summarize_uses_default_prompt_template_when_none_given(monkeypatch):
    captured = {}

    def fake_summarize_ollama(prompt, backend_url, model, on_chunk, log, queue_worker=None):
        captured["prompt"] = prompt
        return "summary"

    monkeypatch.setattr(llm_backend, "_summarize_ollama", fake_summarize_ollama)
    llm_backend.summarize(
        "the transcript", {"url": "http://x", "api_type": "ollama"}, "model",
        on_chunk=lambda c: None, log=print,
    )
    assert captured["prompt"] == llm_backend.DEFAULT_PROMPT_TEMPLATE.replace("{transcript}", "the transcript")


def test_summarize_uses_custom_prompt_template_when_given(monkeypatch):
    captured = {}

    def fake_summarize_ollama(prompt, backend_url, model, on_chunk, log, queue_worker=None):
        captured["prompt"] = prompt
        return "summary"

    monkeypatch.setattr(llm_backend, "_summarize_ollama", fake_summarize_ollama)
    llm_backend.summarize(
        "the transcript", {"url": "http://x", "api_type": "ollama"}, "model",
        on_chunk=lambda c: None, log=print,
        prompt_template="Only action items:\n{transcript}",
    )
    assert captured["prompt"] == "Only action items:\nthe transcript"


def test_summarize_custom_template_survives_stray_braces_in_transcript(monkeypatch):
    """str.replace, not str.format -- a transcript containing literal { }
    (e.g. someone reading out code) must not raise a KeyError/IndexError."""
    captured = {}

    def fake_summarize_ollama(prompt, backend_url, model, on_chunk, log, queue_worker=None):
        captured["prompt"] = prompt
        return "summary"

    monkeypatch.setattr(llm_backend, "_summarize_ollama", fake_summarize_ollama)
    llm_backend.summarize(
        "so then I said {foo: bar}", {"url": "http://x", "api_type": "ollama"}, "model",
        on_chunk=lambda c: None, log=print,
    )
    assert "{foo: bar}" in captured["prompt"]


# ---------------------------------------------------------------------
# _summarize_ollama
# ---------------------------------------------------------------------

def test_summarize_ollama_assembles_chunks(monkeypatch):
    lines = [
        json.dumps({"response": "Hello "}),
        json.dumps({"response": "world"}),
        "",  # blank lines from the server should be skipped
    ]
    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, stream=None, timeout=None: FakeStreamResponse(200, lines),
    )
    chunks = []
    result = llm_backend._summarize_ollama("prompt", "http://x", "model", chunks.append, print)
    assert result == "Hello world"
    assert chunks == ["Hello ", "world"]


def test_summarize_ollama_passes_stream_timeout(monkeypatch):
    seen = {}

    def fake_post(url, json=None, stream=None, timeout=None):
        seen["timeout"] = timeout
        return FakeStreamResponse(200, [])

    monkeypatch.setattr(requests, "post", fake_post)
    llm_backend._summarize_ollama("prompt", "http://x", "model", lambda c: None, print)
    assert seen["timeout"] == llm_backend._STREAM_TIMEOUT


def test_summarize_ollama_non_200_logs_and_returns_none(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, stream=None, timeout=None: FakeStreamResponse(500, [], text="boom"),
    )
    logs = []
    result = llm_backend._summarize_ollama("prompt", "http://x", "model", lambda c: None, logs.append)
    assert result is None
    assert any("500" in line for line in logs)


def test_summarize_ollama_timeout_reports_clearly(monkeypatch):
    def fake_post(*a, **k):
        raise requests.exceptions.Timeout()
    monkeypatch.setattr(requests, "post", fake_post)
    logs = []
    result = llm_backend._summarize_ollama("prompt", "http://x", "model", lambda c: None, logs.append)
    assert result is None
    assert any("hung" in line for line in logs)


def test_summarize_ollama_cancel_stops_mid_stream(monkeypatch):
    def infinite_lines():
        n = 0
        while True:
            n += 1
            yield json.dumps({"response": f"tok{n} "})

    class InfiniteStreamResponse(FakeStreamResponse):
        def iter_lines(self):
            for line in infinite_lines():
                yield line.encode()

    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, stream=None, timeout=None: InfiniteStreamResponse(200),
    )

    qw = FakeQueueWorker()
    chunks = []

    def on_chunk(c):
        chunks.append(c)
        if len(chunks) == 3:
            qw.stop_current = True

    result = llm_backend._summarize_ollama("prompt", "http://x", "model", on_chunk, print, qw)
    assert result is None
    assert len(chunks) <= 4  # loop must exit promptly after cancellation, not run forever


# ---------------------------------------------------------------------
# _summarize_openai
# ---------------------------------------------------------------------

def test_summarize_openai_assembles_chunks_and_stops_at_done(monkeypatch):
    def sse(data):
        return f"data: {json.dumps(data)}"

    lines = [
        sse({"choices": [{"delta": {"content": "Hi "}}]}),
        sse({"choices": [{"delta": {"content": "there"}}]}),
        "data: [DONE]",
        sse({"choices": [{"delta": {"content": "should not appear"}}]}),
    ]
    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, headers=None, stream=None, timeout=None: FakeStreamResponse(200, lines),
    )
    chunks = []
    result = llm_backend._summarize_openai("prompt", "http://x", "model", chunks.append, print)
    assert result == "Hi there"
    assert chunks == ["Hi ", "there"]


def test_summarize_openai_skips_malformed_json_lines(monkeypatch):
    lines = ["data: {not valid json", "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]})]
    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, headers=None, stream=None, timeout=None: FakeStreamResponse(200, lines),
    )
    chunks = []
    result = llm_backend._summarize_openai("prompt", "http://x", "model", chunks.append, print)
    assert result == "ok"


def test_summarize_openai_cancel_stops_mid_stream(monkeypatch):
    def infinite_lines():
        n = 0
        while True:
            n += 1
            yield "data: " + json.dumps({"choices": [{"delta": {"content": f"tok{n} "}}]})

    class InfiniteStreamResponse(FakeStreamResponse):
        def iter_lines(self):
            for line in infinite_lines():
                yield line.encode()

    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, headers=None, stream=None, timeout=None: InfiniteStreamResponse(200),
    )

    qw = FakeQueueWorker()
    chunks = []

    def on_chunk(c):
        chunks.append(c)
        if len(chunks) == 3:
            qw.stop_current = True

    result = llm_backend._summarize_openai("prompt", "http://x", "model", on_chunk, print, qw)
    assert result is None
    assert len(chunks) <= 4


# ---------------------------------------------------------------------
# save_markdown
# ---------------------------------------------------------------------

def test_save_markdown_writes_expected_content(tmp_path):
    path = llm_backend.save_markdown(
        tmp_path, "meeting.wav", "the transcript text", "the summary text",
        {"name": "Ollama"}, "llama3", "medium",
    )
    content = open(path, encoding="utf-8").read()
    assert "# Meeting Summary" in content
    assert "the summary text" in content
    assert "the transcript text" in content
    assert "llama3" in content
    assert "medium" in content


# ---------------------------------------------------------------------
# _extract_summary_snippet
# ---------------------------------------------------------------------

def test_extract_summary_snippet_pulls_text_between_summary_and_transcript():
    md = (
        "# Meeting Summary\n\n**Generated:** now\n\n"
        "---\n\n## Summary\n\nKey decision: ship it.\n\n"
        "---\n\n## Full Transcript\n\n```\nblah\n```\n"
    )
    assert llm_backend._extract_summary_snippet(md) == "Key decision: ship it."


def test_extract_summary_snippet_truncates_long_summaries():
    long_summary = "word " * 50
    md = f"---\n\n## Summary\n\n{long_summary}\n\n---\n\n## Full Transcript\n\n"
    snippet = llm_backend._extract_summary_snippet(md, max_len=20)
    assert len(snippet) <= 21  # 20 chars + the ellipsis
    assert snippet.endswith("…")


def test_extract_summary_snippet_missing_marker_returns_empty_string():
    assert llm_backend._extract_summary_snippet("no summary heading here") == ""


def test_extract_summary_snippet_missing_trailing_dashes_still_works():
    md = "## Summary\n\nJust a summary with no trailing separator."
    assert llm_backend._extract_summary_snippet(md) == "Just a summary with no trailing separator."


def test_extract_summary_snippet_strips_markdown_formatting():
    """The summary is itself LLM-generated Markdown -- the list preview
    should be plain text, not '### **Meeting Summary**' verbatim."""
    md = "## Summary\n\n### **Overview:** The team `shipped` it.\n\n---\n\n## Full Transcript\n\n"
    assert llm_backend._extract_summary_snippet(md) == "Overview: The team shipped it."


# ---------------------------------------------------------------------
# list_past_sessions
# ---------------------------------------------------------------------

def test_list_past_sessions_empty_dir_returns_empty_list(tmp_path):
    assert llm_backend.list_past_sessions(tmp_path / "does-not-exist") == []


def test_list_past_sessions_orders_newest_first(tmp_path):
    for name in ["2026-01-01 10.00.00", "2026-01-03 10.00.00", "2026-01-02 10.00.00"]:
        (tmp_path / name).mkdir()

    sessions = llm_backend.list_past_sessions(tmp_path)
    assert [s["timestamp"] for s in sessions] == [
        "2026-01-03 10.00.00", "2026-01-02 10.00.00", "2026-01-01 10.00.00",
    ]


def test_list_past_sessions_reads_summary_snippet_when_present(tmp_path):
    session_dir = tmp_path / "2026-01-01 10.00.00"
    session_dir.mkdir()
    (session_dir / "summary.md").write_text(
        "---\n\n## Summary\n\nDiscussed the roadmap.\n\n---\n\n## Full Transcript\n\n", encoding="utf-8"
    )

    sessions = llm_backend.list_past_sessions(tmp_path)
    assert len(sessions) == 1
    assert sessions[0]["summary_snippet"] == "Discussed the roadmap."
    assert sessions[0]["md_path"] == str(session_dir / "summary.md")


def test_list_past_sessions_folder_without_summary_still_listed(tmp_path):
    (tmp_path / "2026-01-01 10.00.00").mkdir()

    sessions = llm_backend.list_past_sessions(tmp_path)
    assert len(sessions) == 1
    assert sessions[0]["summary_snippet"] == ""
    assert sessions[0]["md_path"] is None


def test_list_past_sessions_ignores_non_directory_entries(tmp_path):
    (tmp_path / "2026-01-01 10.00.00").mkdir()
    (tmp_path / "stray_file.txt").write_text("not a session", encoding="utf-8")

    sessions = llm_backend.list_past_sessions(tmp_path)
    assert len(sessions) == 1
