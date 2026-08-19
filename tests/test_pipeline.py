"""
Tests for pipeline.py. whisper_engine.transcribe / llm_backend.summarize /
llm_backend.save_markdown are all mocked -- these tests never touch real
audio, a real Whisper model, or a real LLM server. Signal emissions are
captured by connecting each pyqtSignal to a plain list.append, which is a
direct (synchronous) connection since everything here runs on one thread.
"""
from src import pipeline


# ---------------------------------------------------------------------
# new_output_dir
# ---------------------------------------------------------------------

def test_new_output_dir_creates_timestamped_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = pipeline.new_output_dir()
    assert out.exists()
    assert out.parent == tmp_path / "transcripts"
    assert len(out.name.split(" ")) == 2  # "YYYY-MM-DD HH.MM.SS"


# ---------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------

def test_job_ids_increment():
    j1 = pipeline.Job("a.wav", "tiny", {}, "model", False)
    j2 = pipeline.Job("b.wav", "tiny", {}, "model", False)
    assert j2.id == j1.id + 1


def test_job_stores_fields():
    job = pipeline.Job("a.wav", "medium", {"name": "Ollama"}, "llama3", True, output_dir="/tmp/x")
    assert job.audio_file_path == "a.wav"
    assert job.whisper_model == "medium"
    assert job.backend_info == {"name": "Ollama"}
    assert job.llm_model == "llama3"
    assert job.use_whisper_cli is True
    assert job.output_dir == "/tmp/x"


def test_job_review_and_diarization_fields_default_off():
    job = pipeline.Job("a.wav", "tiny", {}, "model", False)
    assert job.review_transcript is False
    assert job.enable_diarization is False
    assert job.hf_token is None
    assert job.enable_tts is False


def test_job_stores_review_and_diarization_fields():
    job = pipeline.Job(
        "a.wav", "tiny", {}, "model", False,
        review_transcript=True, enable_diarization=True, hf_token="hf_abc",
        enable_tts=True,
    )
    assert job.review_transcript is True
    assert job.enable_diarization is True
    assert job.hf_token == "hf_abc"
    assert job.enable_tts is True


# ---------------------------------------------------------------------
# QueueWorker (pure flag/queue bookkeeping -- run()'s loop isn't exercised
# here, it's a thin infinite-poll wrapper around ProcessingWorker.process)
# ---------------------------------------------------------------------

def test_queue_worker_add_job_and_stop_flags():
    qw = pipeline.QueueWorker()
    job = pipeline.Job("a.wav", "tiny", {}, "model", False)
    qw.add_job(job)
    assert len(qw.queue) == 1
    assert qw.queue[0] is job

    assert qw.stop_current is False
    qw.stop_current_job()
    assert qw.stop_current is True

    assert qw.stop_requested is False
    qw.stop()
    assert qw.stop_requested is True


# ---------------------------------------------------------------------
# ProcessingWorker.process
# ---------------------------------------------------------------------

class FakeQueueWorkerFlag:
    def __init__(self, stop_current=False):
        self.stop_current = stop_current


def _connect_collectors(worker):
    events = {"log": [], "progress": [], "summary_chunk": [], "finished": [], "error": []}
    worker.log.connect(events["log"].append)
    worker.progress.connect(events["progress"].append)
    worker.summary_chunk.connect(events["summary_chunk"].append)
    worker.finished.connect(events["finished"].append)
    worker.error.connect(events["error"].append)
    return events


def test_process_cancelled_before_start(tmp_path):
    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job("a.wav", "tiny", {}, "model", False, output_dir=str(tmp_path))
    worker.process(job, FakeQueueWorkerFlag(stop_current=True))

    assert events["error"] == ["Cancelled by user."]
    assert events["finished"] == []


def test_process_audio_file_missing(tmp_path):
    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(tmp_path / "missing.wav"), "tiny", {}, "model", False, output_dir=str(tmp_path))
    worker.process(job, FakeQueueWorkerFlag())

    assert len(events["error"]) == 1
    assert "not found" in events["error"][0]


def test_process_transcription_failure(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: None)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path))
    worker.process(job, FakeQueueWorkerFlag())

    assert events["error"] == ["Transcription failed."]


def test_process_summarization_failure(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "the transcript")
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: None)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path))
    worker.process(job, FakeQueueWorkerFlag())

    assert events["error"] == ["Summarization failed."]


def test_process_success_emits_finished_with_md_path(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    md_path = str(tmp_path / "summary.md")

    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "the transcript")
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: "the summary")
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: md_path)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {"name": "Ollama"}, "model", False, output_dir=str(tmp_path))
    worker.process(job, FakeQueueWorkerFlag())

    assert events["error"] == []
    assert events["finished"] == [md_path]
    assert events["progress"][-1] == 100


def test_process_passes_queue_worker_to_summarize_for_cancellation(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "transcript")

    received = {}

    def fake_summarize(transcript, backend_info, llm_model, on_chunk, log, queue_worker=None, prompt_template=None):
        received["queue_worker"] = queue_worker
        received["prompt_template"] = prompt_template
        return "summary"

    monkeypatch.setattr(pipeline.llm_backend, "summarize", fake_summarize)
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")

    worker = pipeline.ProcessingWorker()
    _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), prompt_template="custom {transcript}")
    qw = FakeQueueWorkerFlag()
    worker.process(job, qw)

    assert received["queue_worker"] is qw
    assert received["prompt_template"] == "custom {transcript}"


def test_process_uses_new_output_dir_when_none_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "transcript")
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: "summary")
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")

    worker = pipeline.ProcessingWorker()
    _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=None)
    worker.process(job, FakeQueueWorkerFlag())

    assert worker.output_dir.parent == tmp_path / "transcripts"
    assert worker.output_dir.exists()


def test_process_exception_is_caught_and_reported_as_error(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", boom)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path))
    worker.process(job, FakeQueueWorkerFlag())

    assert events["error"] == ["disk on fire"]


# ---------------------------------------------------------------------
# Transcript review (job.review_transcript)
#
# process()'s review wait is a plain `while ...: QThread.msleep(100)` poll
# loop -- same idiom already used for Cancel. Monkeypatching QThread.msleep
# lets these tests simulate "the GUI answered instantly" without a real
# thread or a real sleep.
# ---------------------------------------------------------------------

def test_process_emits_transcript_ready_and_uses_edited_result(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "original transcript")

    received = {}

    def fake_summarize(transcript, backend_info, llm_model, on_chunk, log, queue_worker=None, prompt_template=None):
        received["transcript"] = transcript
        return "summary"

    monkeypatch.setattr(pipeline.llm_backend, "summarize", fake_summarize)
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    transcript_ready_events = []
    worker.transcript_ready.connect(transcript_ready_events.append)

    def fake_msleep(ms):
        # First poll tick: pretend the GUI thread just answered "Continue"
        # with edited text.
        worker.review_result = "edited transcript"

    monkeypatch.setattr(pipeline.QThread, "msleep", fake_msleep)

    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), review_transcript=True)
    worker.process(job, FakeQueueWorkerFlag())

    assert transcript_ready_events == ["original transcript"]
    assert received["transcript"] == "edited transcript"
    assert events["error"] == []
    assert events["finished"] == ["md.md"]


def test_process_review_cancelled_aborts_job(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "original transcript")

    summarize_calls = []
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: summarize_calls.append(1))

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)

    def fake_msleep(ms):
        worker.review_cancelled = True

    monkeypatch.setattr(pipeline.QThread, "msleep", fake_msleep)

    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), review_transcript=True)
    worker.process(job, FakeQueueWorkerFlag())

    assert summarize_calls == []  # never reached summarization
    assert events["error"] == ["Cancelled during transcript review."]


def test_process_review_stops_promptly_on_queue_cancel(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "original transcript")

    qw = FakeQueueWorkerFlag()
    calls = {"n": 0}

    def fake_msleep(ms):
        calls["n"] += 1
        qw.stop_current = True
        if calls["n"] > 5:
            raise AssertionError("review poll loop did not exit after cancellation")

    monkeypatch.setattr(pipeline.QThread, "msleep", fake_msleep)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), review_transcript=True)
    worker.process(job, qw)

    assert events["error"] == ["Cancelled by user."]


# ---------------------------------------------------------------------
# Diarization (job.enable_diarization)
# ---------------------------------------------------------------------

def test_process_diarization_labels_transcript_when_available(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    monkeypatch.setattr(
        pipeline.whisper_engine, "transcribe_faster_with_segments",
        lambda *a, **k: ("hi", segments),
    )
    monkeypatch.setattr(pipeline.diarization, "is_available", lambda: True)
    turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    monkeypatch.setattr(pipeline.diarization, "diarize", lambda *a, **k: turns)
    monkeypatch.setattr(pipeline.diarization, "label_transcript", lambda segs, trns: "**SPEAKER_00:** hi")

    received = {}
    monkeypatch.setattr(
        pipeline.llm_backend, "summarize",
        lambda transcript, *a, **k: received.setdefault("transcript", transcript) or "summary",
    )
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), enable_diarization=True)
    worker.process(job, FakeQueueWorkerFlag())

    assert received["transcript"] == "**SPEAKER_00:** hi"
    assert events["error"] == []


def test_process_diarization_falls_back_to_plain_transcript_when_unavailable(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    monkeypatch.setattr(
        pipeline.whisper_engine, "transcribe_faster_with_segments",
        lambda *a, **k: ("hi", segments),
    )
    monkeypatch.setattr(pipeline.diarization, "is_available", lambda: False)

    received = {}
    monkeypatch.setattr(
        pipeline.llm_backend, "summarize",
        lambda transcript, *a, **k: received.setdefault("transcript", transcript) or "summary",
    )
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), enable_diarization=True)
    worker.process(job, FakeQueueWorkerFlag())

    assert received["transcript"] == "hi"  # plain transcript, no labeling
    assert events["error"] == []


def test_process_diarization_skipped_when_using_whisper_cli(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "cli transcript")

    def fail_if_called(*a, **k):
        raise AssertionError("transcribe_faster_with_segments should not be called for whisper-cli jobs")

    monkeypatch.setattr(pipeline.whisper_engine, "transcribe_faster_with_segments", fail_if_called)
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: "summary")
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", True, output_dir=str(tmp_path), enable_diarization=True)
    worker.process(job, FakeQueueWorkerFlag())


# ---------------------------------------------------------------------
# Text-to-speech (job.enable_tts)
# ---------------------------------------------------------------------

def test_process_tts_not_run_when_disabled(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "the transcript")
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: "the summary")
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")

    def fail_if_called(*a, **k):
        raise AssertionError("tts_engine.synthesize should not be called when enable_tts is False")

    monkeypatch.setattr(pipeline.tts_engine, "synthesize", fail_if_called)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), enable_tts=False)
    worker.process(job, FakeQueueWorkerFlag())

    assert events["finished"] == ["md.md"]
    assert events["error"] == []


def test_process_tts_synthesizes_summary_when_enabled(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "the transcript")
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: "the summary")
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")
    monkeypatch.setattr(pipeline.tts_engine, "is_available", lambda: True)

    received = {}

    def fake_synthesize(text, voice_name, output_path, log):
        received["text"] = text
        received["voice_name"] = voice_name
        received["output_path"] = output_path
        return output_path

    monkeypatch.setattr(pipeline.tts_engine, "synthesize", fake_synthesize)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), enable_tts=True)
    worker.process(job, FakeQueueWorkerFlag())

    assert received["text"] == "the summary"
    assert received["voice_name"] == pipeline.tts_engine.DEFAULT_VOICE
    assert received["output_path"] == tmp_path / "summary.wav"
    assert events["finished"] == ["md.md"]
    assert events["error"] == []


def test_process_tts_failure_does_not_fail_job(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "the transcript")
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: "the summary")
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")
    monkeypatch.setattr(pipeline.tts_engine, "is_available", lambda: True)
    monkeypatch.setattr(pipeline.tts_engine, "synthesize", lambda *a, **k: None)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), enable_tts=True)
    worker.process(job, FakeQueueWorkerFlag())

    assert events["finished"] == ["md.md"]
    assert events["error"] == []


def test_process_tts_skipped_when_piper_not_installed(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(pipeline.whisper_engine, "transcribe", lambda *a, **k: "the transcript")
    monkeypatch.setattr(pipeline.llm_backend, "summarize", lambda *a, **k: "the summary")
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")
    monkeypatch.setattr(pipeline.tts_engine, "is_available", lambda: False)

    def fail_if_called(*a, **k):
        raise AssertionError("tts_engine.synthesize should not be called when piper-tts isn't installed")

    monkeypatch.setattr(pipeline.tts_engine, "synthesize", fail_if_called)

    worker = pipeline.ProcessingWorker()
    events = _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path), enable_tts=True)
    worker.process(job, FakeQueueWorkerFlag())

    assert events["finished"] == ["md.md"]
    assert events["error"] == []

    assert events["error"] == []
    assert events["finished"] == ["md.md"]
