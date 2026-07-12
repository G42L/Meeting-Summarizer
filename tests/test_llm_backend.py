"""
Tests for llm_backend.py. All network I/O (requests.get/requests.post) is
mocked -- these tests never touch a real LLM server.
"""
import json

import pytest
import requests

import llm_backend


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
