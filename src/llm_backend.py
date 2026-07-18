#!/usr/bin/env python3
"""
llm_backend.py
---------------
Talks to whichever local LLM server is running (Ollama, vLLM, LM Studio,
llama.cpp) to turn a transcript into a summary.

Named llm_backend.py rather than ollama.py, since it detects and talks to
four different backends, only one of which is Ollama -- and `import ollama`
is a real pip package name, so a local ollama.py would shadow it.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import requests


def _lm_studio_models(url):
    """
    LM Studio needs a model actually *loaded* (not just downloaded) before
    it can serve requests for it. The generic OpenAI-compatible
    /v1/models endpoint lists every downloaded model regardless of load
    state -- that's why "all installed models" was showing up even though
    only one (or none) was actually usable.

    LM Studio also exposes its own native /api/v0/models endpoint (LM
    Studio 0.3.6+), which includes a per-model "state": "loaded" /
    "not-loaded" field. We use that instead.

    Returns (all_models, used_native):
      all_models    list of {"id": str, "usable": bool} for every
                    downloaded model. usable=True only for ones currently
                    loaded into memory (when used_native is True).
      used_native   whether /api/v0/models answered (has real load-state
                    info) or we fell back to plain /v1/models on an older
                    LM Studio -- in the fallback case we can't tell load
                    state at all, so everything is reported usable=True,
                    same as the app's behaviour before this feature.
    """
    r = requests.get(f"{url}/api/v0/models", timeout=1.5)
    if r.status_code == 200:
        data = r.json().get("data", [])
        all_models = [{"id": m["id"], "usable": m.get("state") == "loaded"} for m in data]
        return all_models, True

    r = requests.get(f"{url}/v1/models", timeout=1.5)
    if r.status_code == 200:
        data = r.json().get("data", [])
        all_models = [{"id": m["id"], "usable": True} for m in data]
        return all_models, False

    raise RuntimeError(f"HTTP {r.status_code}")


def detect_backends(log=None):
    """
    Probe known local LLM server ports. Returns {backend_name: info_dict}.

    If `log` is given (a callable taking a string), reports what it found
    or didn't find for each backend, and why. That's the useful bit for
    "I have LM Studio installed but it's not showing up": having it
    *installed* doesn't start its server -- LM Studio only opens the
    localhost:1234 endpoint once you've opened the app, loaded a model,
    and clicked "Start Server" on its Developer/Local Server tab. Without
    `log`, failures are silent (same behaviour as before).
    """
    def report(msg):
        if log:
            log(msg)

    backends = {}
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            backends["Ollama"] = {
                "url": "http://localhost:11434", "api_type": "ollama", "models": models,
                "all_models": [{"id": m, "usable": True} for m in models],
            }
            report(f"✅ Ollama detected at localhost:11434 ({len(models)} model(s))")
        else:
            report(f"⚠️ Ollama replied with HTTP {r.status_code} at localhost:11434 -- ignoring it")
    except Exception as e:
        report(f"— Ollama not reachable at localhost:11434 ({e})")

    for name, url in [
        ("vLLM", "http://localhost:8000"),
        ("LM Studio", "http://localhost:1234"),
        ("llama.cpp", "http://localhost:8080"),
    ]:
        try:
            if name == "LM Studio":
                all_models, used_native = _lm_studio_models(url)
                usable = [m["id"] for m in all_models if m["usable"]]
                # List LM Studio as soon as it's reachable, even with nothing
                # loaded -- that's what lets the model dropdown show the
                # downloaded-but-not-loaded ones greyed out instead of hiding
                # the backend entirely.
                backends[name] = {"url": url, "api_type": "openai", "models": usable, "all_models": all_models}
                if usable and used_native:
                    report(f"✅ {name} detected at {url} ({len(usable)} loaded model(s) of {len(all_models)} downloaded)")
                elif usable:
                    report(f"✅ {name} detected at {url} ({len(all_models)} model(s) -- "
                            f"older LM Studio without /api/v0/models, can't tell which is actually loaded)")
                elif all_models:
                    report(f"⚠️ {name} is running at {url} but no model is loaded "
                           f"({len(all_models)} downloaded, none loaded) -- load one in the LM Studio app")
                else:
                    report(f"⚠️ {name} is running at {url} but has no downloaded models")
                continue

            r = requests.get(f"{url}/v1/models", timeout=1.5)
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])]
                backends[name] = {
                    "url": url, "api_type": "openai", "models": models,
                    "all_models": [{"id": m, "usable": True} for m in models],
                }
                report(f"✅ {name} detected at {url} ({len(models)} model(s))")
            else:
                report(f"⚠️ {name} replied with HTTP {r.status_code} at {url} -- ignoring it")
        except Exception as e:
            report(
                f"— {name} not reachable at {url} ({e}). If it's installed, make sure "
                f"its local server is actually started (and a model loaded) inside the app."
            )
    return backends


# (connect_timeout, read_timeout) for streaming requests. read_timeout is
# the max gap allowed between bytes from the server, not a cap on total
# response time -- a local LLM streaming tokens normally should never hit
# it; a backend that accepted the connection but then hangs will.
_STREAM_TIMEOUT = (5, 60)

DEFAULT_PROMPT_TEMPLATE = (
    "Summarize the following meeting transcript. "
    "Provide a concise summary with key points, decisions, and action items.\n\n"
    "Transcript:\n{transcript}"
)

# Built-in summary styles offered in the UI, each a `{transcript}`-templated
# prompt. Keys are shown to the user as-is (e.g. in a combo box).
PROMPT_TEMPLATES = {
    "Standard Minutes": DEFAULT_PROMPT_TEMPLATE,
    "Action Items Only": (
        "From the following meeting transcript, extract only the action items "
        "as a bullet list, including an owner if one is mentioned. Do not include "
        "any other summary text.\n\nTranscript:\n{transcript}"
    ),
    "Executive Digest": (
        "Write a single concise paragraph (executive digest) of the following "
        "meeting transcript, covering only the single most important decision "
        "or outcome.\n\nTranscript:\n{transcript}"
    ),
}

# Checkbox items offered by the "Select Pre-defined" summary style, in the
# order they should appear both in the UI and in the generated prompt. Keys
# are stored in QSettings; values are the human-readable phrases spliced
# into the prompt sentence.
PREDEFINED_SUMMARY_ITEMS = {
    "key_points": "key points",
    "decisions": "decisions",
    "action_items": "action items",
    "follow_ups": "follow-ups",
    "open_questions": "open questions",
    "risks": "risks and blockers",
    "deadlines": "deadlines",
    "attendees": "attendees and roles",
}

# Prompt used by the "Pure Ollama" option: no summarization instructions at
# all, just the raw transcript, so the model's own default behavior decides
# the output. Mutually exclusive with PREDEFINED_SUMMARY_ITEMS checkboxes.
PURE_OLLAMA_TEMPLATE = "{transcript}"


def build_predefined_template(selected_keys):
    """
    Build a `{transcript}`-templated summarize prompt requesting exactly the
    items named by `selected_keys` (a collection of PREDEFINED_SUMMARY_ITEMS
    keys), phrased in canonical order regardless of selection order.
    Returns None if `selected_keys` is empty.
    """
    phrases = [phrase for key, phrase in PREDEFINED_SUMMARY_ITEMS.items() if key in selected_keys]
    if not phrases:
        return None
    if len(phrases) == 1:
        items_text = phrases[0]
    elif len(phrases) == 2:
        items_text = f"{phrases[0]} and {phrases[1]}"
    else:
        items_text = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    return (
        "Summarize the following meeting transcript. "
        f"Provide a concise summary with {items_text}.\n\n"
        "Transcript:\n{transcript}"
    )


def summarize(transcript, backend_info, llm_model, on_chunk, log, queue_worker=None, prompt_template=None):
    """
    Stream a summary out of the given backend. `on_chunk(str)` is called
    for every incremental piece of text as it arrives (for live display).
    Returns the full summary string, or None on failure.
    `queue_worker`, if given, is polled for `.stop_current` so a job can be
    cancelled mid-stream, not just before it starts.
    `prompt_template`, if given, overrides DEFAULT_PROMPT_TEMPLATE -- must
    contain a `{transcript}` placeholder. Uses str.replace rather than
    str.format so stray `{`/`}` characters in a real transcript or a
    user-written custom template can't raise a KeyError.
    """
    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    prompt = template.replace("{transcript}", transcript)
    backend_url = backend_info["url"]
    api_type = backend_info["api_type"]

    if api_type == "ollama":
        return _summarize_ollama(prompt, backend_url, llm_model, on_chunk, log, queue_worker)
    elif api_type == "openai":
        return _summarize_openai(prompt, backend_url, llm_model, on_chunk, log, queue_worker)
    else:
        log(f"Unknown API type: {api_type}")
        return None


def _cancelled(queue_worker):
    return queue_worker is not None and getattr(queue_worker, "stop_current", False)


def _summarize_ollama(prompt, backend_url, model, on_chunk, log, queue_worker=None):
    payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        with requests.post(f"{backend_url}/api/generate", json=payload, stream=True, timeout=_STREAM_TIMEOUT) as r:
            if r.status_code != 200:
                log(f"Ollama error: {r.status_code} {r.text}")
                return None
            summary = ""
            for line in r.iter_lines():
                if _cancelled(queue_worker):
                    log("🛑 Cancelled during summarization.")
                    return None
                if line:
                    data = json.loads(line.decode())
                    if "response" in data:
                        chunk = data["response"]
                        summary += chunk
                        on_chunk(chunk)
            return summary
    except requests.exceptions.Timeout:
        log(f"Ollama error: no response from server for {_STREAM_TIMEOUT[1]}s -- it may be hung.")
        return None
    except Exception as e:
        log(f"Ollama error: {e}")
        return None


def _summarize_openai(prompt, backend_url, model, on_chunk, log, queue_worker=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    try:
        with requests.post(f"{backend_url}/v1/chat/completions", json=payload, headers=headers, stream=True, timeout=_STREAM_TIMEOUT) as r:
            if r.status_code != 200:
                log(f"API error: {r.status_code} {r.text}")
                return None
            summary = ""
            for line in r.iter_lines():
                if _cancelled(queue_worker):
                    log("🛑 Cancelled during summarization.")
                    return None
                if not line:
                    continue
                line_str = line.decode()
                data_str = line_str[6:] if line_str.startswith("data: ") else line_str
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        summary += delta
                        on_chunk(delta)
                except json.JSONDecodeError:
                    pass
            return summary
    except requests.exceptions.Timeout:
        log(f"API error: no response from server for {_STREAM_TIMEOUT[1]}s -- it may be hung.")
        return None
    except Exception as e:
        log(f"OpenAI error: {e}")
        return None


def save_markdown(output_dir, audio_file, transcript, summary, backend_info, model, whisper_model):
    md_file = Path(output_dir) / "summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Meeting Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Audio File:** {audio_file}\n\n")
        f.write(f"**Whisper Model:** {whisper_model}\n\n")
        f.write(f"**LLM Backend:** {backend_info.get('name', 'Unknown')}\n")
        f.write(f"**LLM Model:** {model}\n\n")
        f.write("---\n\n## Summary\n\n")
        f.write(summary)
        f.write("\n\n---\n\n## Full Transcript\n\n")
        f.write("```\n")
        f.write(transcript)
        f.write("\n```\n\n")
        f.write("---\n*Generated by Meeting Transcriber*")
    return str(md_file)


def _extract_summary_snippet(markdown_text, max_len=100):
    """
    Pulls a short preview out of the '## Summary' section written by
    save_markdown, for display in a one-line list item. Returns "" if the
    marker isn't found (e.g. a hand-edited or foreign .md file).
    """
    marker = "## Summary\n\n"
    idx = markdown_text.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end = markdown_text.find("\n\n---", start)
    snippet = markdown_text[start:end if end != -1 else start + max_len]
    # The summary itself is LLM-generated Markdown (headings, bold, ...) --
    # strip that formatting since this is a plain one-line list preview,
    # not a rendered document.
    snippet = re.sub(r'[#*_`]', '', snippet)
    snippet = " ".join(snippet.split())  # collapse newlines/whitespace for a one-line preview
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rstrip() + "…"
    return snippet


def _read_display_name(folder):
    """Read the custom display name from a session's meta.json, if any."""
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        return ""
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return (data.get("display_name") or "").strip()


def set_session_display_name(folder, display_name):
    """Write/update a session's custom display name in its meta.json."""
    meta_path = Path(folder) / "meta.json"
    data = {}
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    data["display_name"] = display_name
    meta_path.write_text(json.dumps(data), encoding="utf-8")


def list_past_sessions(transcripts_dir):
    """
    Scan a transcripts directory for past session folders (each created by
    save_markdown), newest first by folder name (timestamps sort correctly
    as strings since they're formatted YYYY-MM-DD HH.MM.SS). Sorting is
    always by folder name/timestamp, never by display_name, so a rename
    never disturbs chronological order.

    Returns a list of dicts: {folder: Path, timestamp: str,
    display_name: str, summary_snippet: str, md_path: str or None}.
    display_name is "" unless the user has renamed the session via
    set_session_display_name. Folders without a summary.md (e.g. a job
    that errored before saving) are still listed, with summary_snippet=""
    and md_path=None, since the audio/transcript may still be worth
    revisiting.
    """
    transcripts_dir = Path(transcripts_dir)
    if not transcripts_dir.exists():
        return []

    sessions = []
    for folder in sorted(transcripts_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not folder.is_dir():
            continue
        md_file = folder / "summary.md"
        snippet = ""
        md_path = None
        if md_file.exists():
            md_path = str(md_file)
            try:
                snippet = _extract_summary_snippet(md_file.read_text(encoding="utf-8"))
            except OSError:
                pass
        sessions.append({
            "folder": folder,
            "timestamp": folder.name,
            "display_name": _read_display_name(folder),
            "summary_snippet": snippet,
            "md_path": md_path,
        })
    return sessions
