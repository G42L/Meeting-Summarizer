#!/usr/bin/env python3
"""
audio_engine.py
----------------
The Audio Mixer Engine: lets you capture an arbitrary number of audio
sources at once (microphone, MS Teams / system audio via loopback, a
second mic, etc.) and mixes them down into a single mono stream that
gets saved as one WAV file for Whisper.

Two classes matter to the rest of the app:

    AudioSource         one physical/virtual device being recorded
    AudioMixerEngine     owns a dict of AudioSource, mixes them live

------------------------------------------------------------------
WHY THIS FILE USES TWO DIFFERENT AUDIO LIBRARIES (read before editing)
------------------------------------------------------------------
This used to be built entirely on `sounddevice` (a wrapper around the
PortAudio C library). That worked, but PortAudio ships bundled with the
pip package only on Windows and macOS -- on Linux you have to separately
`apt install libportaudio2` before it'll even import. For a "simple
cross-platform app" that's exactly the kind of extra step we want to
avoid.

`miniaudio` (also a Python/cffi wrapper, this time around the miniaudio
C library) ships a fully self-contained wheel: it does NOT need any
system package installed ahead of time, on any of the three OSes --
verified by importing it in an environment with zero audio libraries
present. On Linux it reaches ALSA/PulseAudio via dlopen() at runtime,
and those are already part of any working Linux desktop, so there is
nothing left for the user to install.

So: **miniaudio is used for every normal capture device** (every
microphone, plus BlackHole on macOS and the PulseAudio ".monitor"
devices on Linux -- both of which are just ordinary capture devices as
far as any audio library is concerned; see the loopback note below).

The one thing miniaudio's Python binding does *not* expose is Windows
WASAPI loopback capture (recording what's being played OUT of a device,
e.g. "everything Teams is playing"). The underlying C library supports
it, but the pip package's high-level API doesn't wire it up, and
reaching around that means poking at undocumented internals that could
break on a miniaudio version bump -- too fragile for what's supposed to
be the *simple* option. Windows already gets PortAudio bundled by pip
automatically (no `apt install` equivalent needed there), so there's no
dependency-reduction upside to forcing that path onto miniaudio too.
`sounddevice` is kept, but ONLY imported (lazily) when a Windows
loopback source is actually added -- if you never add one, `sounddevice`
is never touched and doesn't need to be installed at all.

  * Windows system-audio capture: `sounddevice` + WASAPI loopback
    (`sd.WasapiSettings(loopback=True)`, opened on the *output* device).
  * macOS system-audio capture: install BlackHole (a free virtual audio
    driver -- not a pip package: https://existential.audio/blackhole/),
    route Teams's output to it (directly, or via a Multi-Output Device
    in Audio MIDI Setup so you still hear it too). It then just shows up
    as a normal capture device, handled by miniaudio like a microphone.
  * Linux system-audio capture: every PulseAudio/PipeWire output (sink)
    automatically has a matching ".monitor" source. It shows up as a
    normal capture device (name usually contains "Monitor of ..."),
    handled by miniaudio like a microphone. Nothing to install.
"""

import platform
import threading
from collections import deque

import numpy as np
import miniaudio

# The rate everything gets mixed down to. Whisper wants 16 kHz mono, so
# mixing at that rate means the saved WAV is already Whisper-ready with
# no extra resampling step at transcription time.
ENGINE_SAMPLE_RATE = 16000

# Devices whose *name* signals they're a virtual loopback/monitor device
# on macOS or Linux (case-insensitive substring match). Add to this list
# if you install a different virtual audio driver.
_LOOPBACK_NAME_HINTS = (
    "blackhole", "soundflower", "loopback",
    "monitor of", "pulse monitor", "stereo mix", "what u hear",
)


def get_platform():
    """'windows' | 'macos' | 'linux' -- used to pick the loopback strategy."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


def _resample_linear(samples, src_rate, dst_rate):
    """
    Linear-interpolation resampler. Not audiophile-grade, but Whisper
    resamples/downmixes internally anyway and speech content survives
    linear interpolation just fine -- this keeps the engine dependency-free
    (no scipy). If you later want higher fidelity, swap this one function
    for scipy.signal.resample_poly and nothing else needs to change.
    """
    if src_rate == dst_rate or len(samples) == 0:
        return samples
    duration = len(samples) / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _pick_format(formats):
    """
    Each miniaudio device advertises a list of supported (format,
    samplerate, channels) combos. Pick something sane: 16 kHz if the
    device happens to support it directly (skips resampling entirely),
    otherwise a common rate; mono if available, otherwise stereo
    (we downmix to mono ourselves either way).
    """
    samplerates = sorted({f["samplerate"] for f in formats if f.get("samplerate")})
    channel_opts = sorted({f["channels"] for f in formats if f.get("channels")})

    if not samplerates:
        samplerate = 44100
    elif ENGINE_SAMPLE_RATE in samplerates:
        samplerate = ENGINE_SAMPLE_RATE
    elif 44100 in samplerates:
        samplerate = 44100
    else:
        samplerate = samplerates[0]

    if not channel_opts:
        channels = 2
    elif 1 in channel_opts:
        channels = 1
    else:
        channels = channel_opts[0]

    return samplerate, channels


def list_input_devices():
    """
    Every normal capture device miniaudio can see: real microphones, plus
    macOS/Linux virtual loopback devices (they're indistinguishable from a
    real input device at the API level -- see module docstring).

    Returns a list of dicts:
        {device_id, name, samplerate, channels, is_loopback, wasapi_loopback}
    `device_id` is an opaque object -- pass it straight back into
    AudioMixerEngine.add_source(), don't inspect or compare it yourself.
    """
    result = []
    for dev in miniaudio.Devices().get_captures():
        samplerate, channels = _pick_format(dev.get("formats") or [])
        name_lower = dev["name"].lower()
        result.append({
            "device_id": dev["id"],
            "name": dev["name"],
            "samplerate": samplerate,
            "channels": channels,
            "is_loopback": any(hint in name_lower for hint in _LOOPBACK_NAME_HINTS),
            "wasapi_loopback": False,
        })
    return result


def list_loopback_devices():
    """
    System-audio / "what you hear" capture candidates.

    On macOS and Linux these are just entries from list_input_devices()
    that matched a known virtual-device name, so this returns a subset
    of that list (no sounddevice/PortAudio involved).

    On Windows, WASAPI loopback works on any OUTPUT device (it captures
    whatever is being played through it), so this instead lists your
    output devices (speakers, headphones, virtual cables) tagged with
    wasapi_loopback=True. This is the one path that still uses
    `sounddevice`, imported lazily right here -- if you're not on
    Windows, or never call this, sounddevice is never touched.
    """
    if get_platform() != "windows":
        return [d for d in list_input_devices() if d["is_loopback"]]

    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError(
            "Windows system-audio (loopback) capture needs the optional "
            "'sounddevice' package. Install it with: pip install sounddevice\n"
            f"(underlying error: {e})"
        )

    result = []
    devices = sd.query_devices()
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []
    for i, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) <= 0:
            continue
        hostapi_name = ""
        if hostapis and 0 <= dev.get("hostapi", -1) < len(hostapis):
            hostapi_name = hostapis[dev["hostapi"]].get("name", "")
        if "wasapi" not in hostapi_name.lower():
            continue  # loopback only works via the WASAPI host API
        result.append({
            "device_id": i,  # a plain sounddevice device index, not a miniaudio id
            "name": f"{dev['name']} (loopback)",
            "samplerate": dev.get("default_samplerate") or ENGINE_SAMPLE_RATE,
            "channels": min(2, dev["max_output_channels"]),
            "is_loopback": True,
            "wasapi_loopback": True,
        })
    return result


def list_all_sources():
    """
    Everything the UI should offer in the "Add Source" picker: normal
    inputs plus (on Windows) WASAPI loopback candidates. Deduplicated by
    (name, wasapi_loopback) -- device_id objects from miniaudio aren't
    guaranteed to compare equal across separate enumeration calls, so
    name is the safe thing to dedupe on here.
    """
    combined = list(list_input_devices())
    seen = {(d["name"], d["wasapi_loopback"]) for d in combined}
    try:
        loopbacks = list_loopback_devices()
    except RuntimeError:
        loopbacks = []  # sounddevice not installed and not on this platform's critical path
    for d in loopbacks:
        key = (d["name"], d["wasapi_loopback"])
        if key not in seen:
            combined.append(d)
            seen.add(key)
    return combined


class AudioSource:
    """
    One capture device. Downmixes to mono, resamples to
    ENGINE_SAMPLE_RATE, applies gain/mute, and buffers the result in a
    deque that AudioMixerEngine drains periodically.

    Two backends live behind this one class:
      * wasapi_loopback=True  -> sounddevice (Windows loopback only)
      * everything else       -> miniaudio (mic, BlackHole, pulse monitor)
    Whichever backend is in use, its audio callback runs on that
    library's own native thread -- that's why self._lock guards the
    deque, and why the callback must stay fast (no logging, no GUI calls).
    """

    def __init__(self, name, device_id, samplerate, channels,
                 is_loopback=False, wasapi_loopback=False, gain=1.0):
        self.name = name
        self.device_id = device_id
        self.native_samplerate = int(samplerate)
        self.channels = max(1, int(channels))
        self.is_loopback = is_loopback
        self.wasapi_loopback = wasapi_loopback

        self.gain = gain
        self.muted = False

        self.level = 0.0          # latest RMS (0..1-ish), for the per-source VU meter
        self.error = None         # last stream error message, if any

        self._lock = threading.Lock()
        self._buffer = deque()    # resampled, gain-applied float32 chunks awaiting mixing
        self._preview = deque(maxlen=ENGINE_SAMPLE_RATE)  # this source's own live waveform

        self._miniaudio_device = None
        self._miniaudio_generator = None
        self._sd_stream = None

    # ---------------- stream lifecycle ----------------

    def start(self):
        self.error = None
        try:
            if self.wasapi_loopback:
                self._start_sounddevice_loopback()
            else:
                self._start_miniaudio()
        except Exception as e:
            self.error = str(e)
            raise

    def _start_miniaudio(self):
        gen = self._miniaudio_capture_generator()
        next(gen)  # prime it up to its first `yield`, per miniaudio's contract
        self._miniaudio_generator = gen
        self._miniaudio_device = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=self.channels,
            sample_rate=self.native_samplerate,
            buffersize_msec=100,
            device_id=self.device_id,
        )
        self._miniaudio_device.start(gen)

    def _start_sounddevice_loopback(self):
        import sounddevice as sd  # lazy: only Windows loopback sources need this
        extra = sd.WasapiSettings(loopback=True)
        self._sd_stream = sd.InputStream(
            device=self.device_id,
            channels=self.channels,
            samplerate=self.native_samplerate,
            dtype="float32",
            blocksize=1024,
            latency="low",
            extra_settings=extra,
            callback=self._sd_callback,
        )
        self._sd_stream.start()

    def stop(self):
        if self._miniaudio_device is not None:
            try:
                self._miniaudio_device.stop()
                self._miniaudio_device.close()
            except Exception:
                pass
            self._miniaudio_device = None
            self._miniaudio_generator = None
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

    @property
    def is_active(self):
        return self._miniaudio_device is not None or self._sd_stream is not None

    # ---------------- audio callbacks ----------------

    def _miniaudio_capture_generator(self):
        """
        miniaudio's capture protocol: a generator that receives raw PCM
        bytes via .send() each time a chunk is ready, and must keep
        yielding to ask for the next one. Runs on miniaudio's audio
        thread, same rules as any audio callback: keep it fast.
        """
        while True:
            raw_bytes = yield
            mono = self._decode_and_downmix(raw_bytes)
            self._handle_chunk(mono)

    def _decode_and_downmix(self, raw_bytes):
        samples = np.frombuffer(raw_bytes, dtype=np.float32)
        if self.channels > 1:
            samples = samples.reshape(-1, self.channels).mean(axis=1)
        return samples.astype(np.float32, copy=False)

    def _sd_callback(self, indata, frames, time_info, status):
        if status:
            self.error = str(status)
        mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata.reshape(-1)
        self._handle_chunk(mono.astype(np.float32, copy=False))

    def _handle_chunk(self, mono):
        """Shared tail end of both backends: level, resample, gain, buffer."""
        if mono.size:
            self.level = float(np.sqrt(np.mean(np.square(mono))))

        resampled = _resample_linear(mono, self.native_samplerate, ENGINE_SAMPLE_RATE)

        with self._lock:
            self._preview.extend(resampled.tolist())
            if not self.muted:
                chunk = resampled * self.gain if self.gain != 1.0 else resampled
                self._buffer.append(np.clip(chunk, -1.0, 1.0))

    # ---------------- consumption (called from the mixer, GUI thread) ----------------

    def pull_available(self):
        """Pop and concatenate everything buffered since the last pull."""
        with self._lock:
            if not self._buffer:
                return np.empty(0, dtype=np.float32)
            chunks = list(self._buffer)
            self._buffer.clear()
        return np.concatenate(chunks)

    def preview_snapshot(self):
        with self._lock:
            return list(self._preview)


class AudioMixerEngine:
    """
    Owns a set of named AudioSource objects, starts/stops them together,
    and periodically mixes whatever they've each produced into one mono
    stream at ENGINE_SAMPLE_RATE.

    Usage:
        engine = AudioMixerEngine()
        engine.add_source("Microphone", device_id=mic["device_id"],
                           samplerate=mic["samplerate"], channels=mic["channels"])
        engine.add_source("Teams (loopback)", device_id=loop["device_id"],
                           samplerate=loop["samplerate"], channels=loop["channels"],
                           is_loopback=True, wasapi_loopback=loop["wasapi_loopback"])
        engine.start()
        ...
        engine.tick()          # call this on a QTimer, e.g. every 33ms
        ...
        audio = engine.stop()  # returns the full mixed np.ndarray
    """

    def __init__(self):
        self.sources = {}                     # name -> AudioSource
        self._mixed_chunks = []               # accumulated across the whole recording
        self._mixed_preview = deque(maxlen=ENGINE_SAMPLE_RATE)
        self._running = False

    # ---------------- source management ----------------

    def add_source(self, name, device_id, samplerate, channels,
                    is_loopback=False, wasapi_loopback=False, gain=1.0):
        if name in self.sources:
            raise ValueError(f"A source named '{name}' is already in the mix.")
        source = AudioSource(
            name=name, device_id=device_id, samplerate=samplerate,
            channels=channels, is_loopback=is_loopback,
            wasapi_loopback=wasapi_loopback, gain=gain,
        )
        self.sources[name] = source
        if self._running:
            source.start()
        return source

    def remove_source(self, name):
        source = self.sources.pop(name, None)
        if source is not None:
            source.stop()

    def set_gain(self, name, gain):
        if name in self.sources:
            self.sources[name].gain = max(0.0, gain)

    def set_muted(self, name, muted):
        if name in self.sources:
            self.sources[name].muted = muted

    # ---------------- recording lifecycle ----------------

    def start(self):
        if not self.sources:
            raise RuntimeError("Add at least one source before starting the mixer.")
        self._mixed_chunks = []
        self._mixed_preview.clear()
        errors = []
        for source in self.sources.values():
            try:
                source.start()
            except Exception as e:
                errors.append(f"{source.name}: {e}")
        self._running = True
        return errors  # sources that failed to open; caller decides how to warn

    def stop(self):
        for source in self.sources.values():
            source.stop()
        self._running = False
        return self.get_mixed_audio()

    @property
    def is_running(self):
        return self._running

    def tick(self):
        """
        Pull whatever each active source has produced since the last
        tick, pad the shorter ones with zeros so they line up, sum them,
        and append the result to the running mix. Call this regularly
        (e.g. from a 33ms QTimer) while recording.

        Returns the RMS of the newly mixed chunk (0.0 if nothing new).
        """
        if not self._running:
            return 0.0

        chunks = []
        for source in self.sources.values():
            if not source.is_active:
                continue
            # Always drain the buffer, even when muted, so audio captured
            # in the instant just before a mute click doesn't sit there and
            # get mixed in later when the source is unmuted again.
            chunk = source.pull_available()
            if source.muted:
                continue
            if chunk.size:
                chunks.append(chunk)

        if not chunks:
            return 0.0

        max_len = max(c.size for c in chunks)
        mixed = np.zeros(max_len, dtype=np.float32)
        for c in chunks:
            if c.size < max_len:
                c = np.pad(c, (0, max_len - c.size))
            mixed += c
        mixed = np.clip(mixed, -1.0, 1.0)

        self._mixed_chunks.append(mixed)
        self._mixed_preview.extend(mixed.tolist())
        return float(np.sqrt(np.mean(np.square(mixed)))) if mixed.size else 0.0

    # ---------------- reading results ----------------

    def get_mixed_audio(self):
        """Full mixed recording so far, as one mono float32 np.ndarray at ENGINE_SAMPLE_RATE."""
        if not self._mixed_chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._mixed_chunks)

    def get_mixed_preview(self):
        """Last ~1s of mixed audio, for the combined waveform widget."""
        return list(self._mixed_preview)

    def get_source_level(self, name):
        source = self.sources.get(name)
        return source.level if source else 0.0

    def get_source_errors(self):
        return {name: s.error for name, s in self.sources.items() if s.error}
