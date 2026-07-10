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


def summarize(transcript, backend_info, llm_model, on_chunk, log):
    """
    Stream a summary out of the given backend. `on_chunk(str)` is called
    for every incremental piece of text as it arrives (for live display).
    Returns the full summary string, or None on failure.
    """
    prompt = (
        "Summarize the following meeting transcript. "
        "Provide a concise summary with key points, decisions, and action items.\n\n"
        f"Transcript:\n{transcript}"
    )
    backend_url = backend_info["url"]
    api_type = backend_info["api_type"]

    if api_type == "ollama":
        return _summarize_ollama(prompt, backend_url, llm_model, on_chunk, log)
    elif api_type == "openai":
        return _summarize_openai(prompt, backend_url, llm_model, on_chunk, log)
    else:
        log(f"Unknown API type: {api_type}")
        return None


def _summarize_ollama(prompt, backend_url, model, on_chunk, log):
    payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        with requests.post(f"{backend_url}/api/generate", json=payload, stream=True) as r:
            if r.status_code != 200:
                log(f"Ollama error: {r.status_code} {r.text}")
                return None
            summary = ""
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    if "response" in data:
                        chunk = data["response"]
                        summary += chunk
                        on_chunk(chunk)
            return summary
    except Exception as e:
        log(f"Ollama error: {e}")
        return None


def _summarize_openai(prompt, backend_url, model, on_chunk, log):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    try:
        with requests.post(f"{backend_url}/v1/chat/completions", json=payload, headers=headers, stream=True) as r:
            if r.status_code != 200:
                log(f"API error: {r.status_code} {r.text}")
                return None
            summary = ""
            for line in r.iter_lines():
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
