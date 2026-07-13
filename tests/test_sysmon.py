"""
Tests for sysmon.py. Real hardware/drivers (NVML, sysfs, system_profiler)
are all mocked -- these tests must pass identically on a CI runner with no
GPU at all. sysmon._NVML_OK / sysmon.pynvml are monkeypatched per test
rather than relying on whatever this machine's real detection found.
"""
import subprocess
import sys

import pytest

from src import sysmon


# ---------------------------------------------------------------------
# sample_cpu_ram
# ---------------------------------------------------------------------

class FakeVirtualMemory:
    def __init__(self, total, available, percent):
        self.total = total
        self.available = available
        self.percent = percent


def test_sample_cpu_ram(monkeypatch):
    gib = 1024 ** 3
    monkeypatch.setattr(sysmon.psutil, "cpu_percent", lambda interval=None: 42.0)
    monkeypatch.setattr(
        sysmon.psutil, "virtual_memory",
        lambda: FakeVirtualMemory(total=16 * gib, available=8 * gib, percent=50.0),
    )
    result = sysmon.sample_cpu_ram()
    assert result["cpu_percent"] == 42.0
    assert result["ram_total_gb"] == pytest.approx(16.0)
    assert result["ram_used_gb"] == pytest.approx(8.0)
    assert result["ram_percent"] == 50.0


# ---------------------------------------------------------------------
# list_gpus / sample_gpu -- NVML (NVIDIA) backend
# ---------------------------------------------------------------------

class FakeUtil:
    def __init__(self, gpu):
        self.gpu = gpu


class FakeMemInfo:
    def __init__(self, used, total):
        self.used = used
        self.total = total


class FakeNvml:
    def __init__(self, names):
        self._names = names

    def nvmlDeviceGetCount(self):
        return len(self._names)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i  # handle is opaque to our code; use the index as a stand-in

    def nvmlDeviceGetName(self, handle):
        return self._names[handle]

    def nvmlDeviceGetUtilizationRates(self, handle):
        return FakeUtil(gpu=31.0)

    def nvmlDeviceGetMemoryInfo(self, handle):
        mib = 1024 ** 2
        return FakeMemInfo(used=2000 * mib, total=16000 * mib)

    def nvmlShutdown(self):
        self.shutdown_called = True


def test_list_gpus_nvml_backend(monkeypatch):
    fake = FakeNvml(["NVIDIA GeForce RTX 5070 Ti"])
    monkeypatch.setattr(sysmon, "_NVML_OK", True)
    monkeypatch.setattr(sysmon, "pynvml", fake)
    monkeypatch.setattr(sysmon.sys, "platform", "win32")  # skip the Linux sysfs branch

    gpus = sysmon.list_gpus()
    assert len(gpus) == 1
    assert gpus[0]["vendor"] == "NVIDIA"
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert gpus[0]["backend"] == "nvml"


def test_list_gpus_nvml_decodes_bytes_name(monkeypatch):
    class FakeNvmlBytes(FakeNvml):
        def nvmlDeviceGetName(self, handle):
            return self._names[handle].encode()

    fake = FakeNvmlBytes(["NVIDIA Fake GPU"])
    monkeypatch.setattr(sysmon, "_NVML_OK", True)
    monkeypatch.setattr(sysmon, "pynvml", fake)
    monkeypatch.setattr(sysmon.sys, "platform", "win32")

    gpus = sysmon.list_gpus()
    assert gpus[0]["name"] == "NVIDIA Fake GPU"


def test_sample_gpu_nvml(monkeypatch):
    fake = FakeNvml(["NVIDIA GeForce RTX 5070 Ti"])
    monkeypatch.setattr(sysmon, "pynvml", fake)
    gpu = {"backend": "nvml", "handle": 0}
    stats = sysmon.sample_gpu(gpu)
    assert stats["load_percent"] == 31.0
    assert stats["vram_used_mb"] == pytest.approx(2000.0)
    assert stats["vram_total_mb"] == pytest.approx(16000.0)


def test_sample_gpu_nvml_error_returns_all_none(monkeypatch):
    class BrokenNvml:
        def nvmlDeviceGetUtilizationRates(self, handle):
            raise RuntimeError("device lost")

    monkeypatch.setattr(sysmon, "pynvml", BrokenNvml())
    stats = sysmon.sample_gpu({"backend": "nvml", "handle": 0})
    assert stats == {"load_percent": None, "vram_used_mb": None, "vram_total_mb": None}


def test_list_gpus_nvml_detection_failure_is_swallowed(monkeypatch):
    class BrokenNvml:
        def nvmlDeviceGetCount(self):
            raise RuntimeError("driver not loaded")

    monkeypatch.setattr(sysmon, "_NVML_OK", True)
    monkeypatch.setattr(sysmon, "pynvml", BrokenNvml())
    monkeypatch.setattr(sysmon.sys, "platform", "win32")
    assert sysmon.list_gpus() == []


def test_shutdown_calls_nvml_shutdown_when_available(monkeypatch):
    fake = FakeNvml([])
    monkeypatch.setattr(sysmon, "_NVML_OK", True)
    monkeypatch.setattr(sysmon, "pynvml", fake)
    sysmon.shutdown()
    assert getattr(fake, "shutdown_called", False) is True


def test_shutdown_noop_when_nvml_unavailable(monkeypatch):
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    sysmon.shutdown()  # must not raise


# ---------------------------------------------------------------------
# list_gpus / sample_gpu -- Linux sysfs backend (AMD / Intel)
# ---------------------------------------------------------------------

def _make_sysfs_device(tmp_path, name, vendor_id, driver, gpu_busy=None, vram_used=None, vram_total=None):
    device_dir = tmp_path / f"sys_{name}" / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "uevent").write_text(f"DRIVER={driver}\n")
    (device_dir / "vendor").write_text(vendor_id + "\n")
    if gpu_busy is not None:
        (device_dir / "gpu_busy_percent").write_text(str(gpu_busy))
    if vram_used is not None:
        (device_dir / "mem_info_vram_used").write_text(str(vram_used))
    if vram_total is not None:
        (device_dir / "mem_info_vram_total").write_text(str(vram_total))
    return device_dir


def test_list_gpus_sysfs_amd_backend(tmp_path, monkeypatch):
    device_dir = _make_sysfs_device(
        tmp_path, "amd", vendor_id="0x1002", driver="amdgpu",
        gpu_busy=17, vram_used=1000 * 1024 * 1024, vram_total=8000 * 1024 * 1024,
    )
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    monkeypatch.setattr(sysmon.sys, "platform", "linux")
    monkeypatch.setattr(sysmon.glob, "glob", lambda pattern: [str(device_dir)])

    gpus = sysmon.list_gpus()
    assert len(gpus) == 1
    assert gpus[0]["vendor"] == "AMD"
    assert gpus[0]["backend"] == "sysfs"

    stats = sysmon.sample_gpu(gpus[0])
    assert stats["load_percent"] == 17.0
    assert stats["vram_used_mb"] == pytest.approx(1000.0)
    assert stats["vram_total_mb"] == pytest.approx(8000.0)


def test_list_gpus_sysfs_intel_without_stats_reports_na(tmp_path, monkeypatch):
    device_dir = _make_sysfs_device(tmp_path, "intel", vendor_id="0x8086", driver="i915")
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    monkeypatch.setattr(sysmon.sys, "platform", "linux")
    monkeypatch.setattr(sysmon.glob, "glob", lambda pattern: [str(device_dir)])

    gpus = sysmon.list_gpus()
    assert len(gpus) == 1
    assert gpus[0]["vendor"] == "Intel"

    stats = sysmon.sample_gpu(gpus[0])
    assert stats == {"load_percent": None, "vram_used_mb": None, "vram_total_mb": None}


def test_list_gpus_sysfs_skips_nvidia_driver_devices(tmp_path, monkeypatch):
    # NVIDIA GPUs are already covered via NVML -- the sysfs scan must not
    # double-list a card whose driver is "nvidia".
    device_dir = _make_sysfs_device(tmp_path, "nv", vendor_id="0x10de", driver="nvidia")
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    monkeypatch.setattr(sysmon.sys, "platform", "linux")
    monkeypatch.setattr(sysmon.glob, "glob", lambda pattern: [str(device_dir)])

    assert sysmon.list_gpus() == []


def test_list_gpus_sysfs_skipped_on_non_linux(monkeypatch):
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    monkeypatch.setattr(sysmon.sys, "platform", "win32")
    monkeypatch.setattr(sysmon.glob, "glob", lambda pattern: (_ for _ in ()).throw(
        AssertionError("glob should not be called on non-Linux platforms")
    ))
    assert sysmon.list_gpus() == []


# ---------------------------------------------------------------------
# list_gpus -- macOS backend
# ---------------------------------------------------------------------

def test_list_gpus_macos_apple_silicon(monkeypatch):
    payload = '{"SPDisplaysDataType": [{"_name": "Apple M2 Pro", "sppci_model": "Apple M2 Pro", "sppci_cores": "19"}]}'
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    monkeypatch.setattr(sysmon.sys, "platform", "darwin")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=payload),
    )

    gpus = sysmon.list_gpus()
    assert len(gpus) == 1
    assert gpus[0]["vendor"] == "Apple"
    assert "19-core" in gpus[0]["name"]
    assert gpus[0]["backend"] == "none"

    stats = sysmon.sample_gpu(gpus[0])
    assert stats == {"load_percent": None, "vram_used_mb": None, "vram_total_mb": None}


def test_list_gpus_macos_multiple_gpus_guesses_vendor_from_name(monkeypatch):
    payload = (
        '{"SPDisplaysDataType": ['
        '{"_name": "Intel UHD Graphics 630", "sppci_model": "Intel UHD Graphics 630"},'
        '{"_name": "AMD Radeon Pro 5500M", "sppci_model": "AMD Radeon Pro 5500M"}'
        ']}'
    )
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    monkeypatch.setattr(sysmon.sys, "platform", "darwin")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=payload),
    )

    gpus = sysmon.list_gpus()
    vendors = {g["vendor"] for g in gpus}
    assert vendors == {"Intel", "AMD"}


def test_list_gpus_macos_missing_system_profiler_is_swallowed(monkeypatch):
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    monkeypatch.setattr(sysmon.sys, "platform", "darwin")

    def fake_run(*a, **k):
        raise FileNotFoundError("no such command")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert sysmon.list_gpus() == []


def test_list_gpus_macos_malformed_json_is_swallowed(monkeypatch):
    monkeypatch.setattr(sysmon, "_NVML_OK", False)
    monkeypatch.setattr(sysmon.sys, "platform", "darwin")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="not json"),
    )
    assert sysmon.list_gpus() == []
