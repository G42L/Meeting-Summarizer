"""
sysmon.py - CPU/RAM/GPU/VRAM sampling for the System monitor panel.
Pure Python, no PyQt5 import -- main.py is the only thing that turns these
numbers into widgets (same boundary as audio_engine.py/whisper_engine.py).

GPU support is best-effort and platform-dependent:
  - NVIDIA: full load% + VRAM via nvidia-ml-py (pynvml), works on Linux/Windows
    wherever the NVIDIA driver is installed -- no nvidia-smi binary needed.
  - AMD (Linux, amdgpu driver): full load% + VRAM via sysfs
    (/sys/class/drm/cardN/device/{gpu_busy_percent,mem_info_vram_*}).
  - Intel / other (Linux): detected via the PCI vendor ID in sysfs, but most
    kernels don't expose busy%/VRAM for integrated GPUs there, so those
    fields come back as None ("N/A" in the UI) -- the GPU is still listed.
  - macOS, or AMD/Intel on Windows: not detected. There's no vendor-neutral,
    pip-installable API for those combinations.
"""
import sys
import glob
import os

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

_PCI_VENDOR_NAMES = {
    "0x10de": "NVIDIA",
    "0x1002": "AMD",
    "0x8086": "Intel",
}


def sample_cpu_ram():
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": (vm.total - vm.available) / (1024 ** 3),
        "ram_total_gb": vm.total / (1024 ** 3),
        "ram_percent": vm.percent,
    }


def list_gpus():
    """Detect all GPUs once at startup. Returns a list of dicts; pass one of
    these back into sample_gpu() on every tick."""
    gpus = []

    if _NVML_OK:
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                gpus.append({
                    "vendor": "NVIDIA", "name": name,
                    "backend": "nvml", "handle": handle,
                })
        except Exception:
            pass

    if sys.platform.startswith("linux"):
        for device_dir in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            driver = ""
            try:
                with open(os.path.join(device_dir, "uevent")) as f:
                    for line in f:
                        if line.startswith("DRIVER="):
                            driver = line.strip().split("=", 1)[1]
            except OSError:
                continue
            if driver == "nvidia":
                continue  # already covered via NVML above

            try:
                with open(os.path.join(device_dir, "vendor")) as f:
                    vendor_id = f.read().strip()
            except OSError:
                continue
            vendor = _PCI_VENDOR_NAMES.get(vendor_id, vendor_id)
            name = f"{vendor} Graphics ({driver or os.path.basename(device_dir)})"
            gpus.append({
                "vendor": vendor, "name": name,
                "backend": "sysfs", "handle": device_dir,
            })

    return gpus


def sample_gpu(gpu):
    """Returns dict: load_percent, vram_used_mb, vram_total_mb -- any of
    which may be None if that stat isn't available for this GPU."""
    if gpu["backend"] == "nvml":
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu["handle"])
            mem = pynvml.nvmlDeviceGetMemoryInfo(gpu["handle"])
            return {
                "load_percent": float(util.gpu),
                "vram_used_mb": mem.used / (1024 ** 2),
                "vram_total_mb": mem.total / (1024 ** 2),
            }
        except Exception:
            return {"load_percent": None, "vram_used_mb": None, "vram_total_mb": None}

    if gpu["backend"] == "sysfs":
        device_dir = gpu["handle"]
        load = _read_int(os.path.join(device_dir, "gpu_busy_percent"))
        vram_used = _read_int(os.path.join(device_dir, "mem_info_vram_used"))
        vram_total = _read_int(os.path.join(device_dir, "mem_info_vram_total"))
        return {
            "load_percent": float(load) if load is not None else None,
            "vram_used_mb": vram_used / (1024 ** 2) if vram_used is not None else None,
            "vram_total_mb": vram_total / (1024 ** 2) if vram_total is not None else None,
        }

    return {"load_percent": None, "vram_used_mb": None, "vram_total_mb": None}


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def shutdown():
    """Call once on app exit to release the NVML handle cleanly."""
    if _NVML_OK:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
