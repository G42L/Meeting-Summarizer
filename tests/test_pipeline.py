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

    def fake_summarize(transcript, backend_info, llm_model, on_chunk, log, queue_worker=None):
        received["queue_worker"] = queue_worker
        return "summary"

    monkeypatch.setattr(pipeline.llm_backend, "summarize", fake_summarize)
    monkeypatch.setattr(pipeline.llm_backend, "save_markdown", lambda *a, **k: "md.md")

    worker = pipeline.ProcessingWorker()
    _connect_collectors(worker)
    job = pipeline.Job(str(audio), "tiny", {}, "model", False, output_dir=str(tmp_path))
    qw = FakeQueueWorkerFlag()
    worker.process(job, qw)

    assert received["queue_worker"] is qw


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
