# Meeting Transcriber
A cross‑platform desktop application built with PyQt5 that records or loads audio, transcribes it with Whisper, and generates a structured summary using a local large language model (LLM). All files are saved in timestamped folders under ./transcripts/.


| Light Theme | Dark Theme |
|------------|------------|
| ![Light Theme](images/light-theme.png) | ![Dark Theme](images/dark-theme.png) |
| Main application window using the light theme. | Main application window using the dark theme. |

Table of Contents
1. [Features](#Features)
2. [System Requirements](#System-Requirements)
3. [Installation](#Installation)
4. [Python Dependencies](#Python-Dependencies)
5. [Whisper Backend](#Whisper-Backend)
5. [LLM Backend](#LLM-Backend)
5. [Configuration](#Configuration)
5. [Usage](#Usage)
5. [Recording](#Recording)
5. [Loading an Existing Audio File](#Loading-an-Existing-Audio-File)
5. [Selecting Models](#Selecting-Models)
5. [Processing](#Processing)
5. [Viewing Output](#Viewing-Output)
5. [File Output Structure](#FileOutput-Structure)
5. [Troubleshooting](#Troubleshooting)
5. [License](#License)

# Features
* 🎤 Live recording with real‑time VU meter and scrolling waveform display.
* 📂 Load audio files in common formats (WAV, MP3, M4A, FLAC, OGG, AAC).
* 🗣️ Transcription using either:
    * faster‑whisper (Python, recommended) – automatically used if installed.
    * whisper‑cli from whisper.cpp (C++ implementation) – lower resource usage.
* 🤖 Summarization via local LLM servers supporting:
    * Ollama
    * vLLM
    * LM Studio
    * llama.cpp (OpenAI‑compatible endpoint)
    * and more ....
* 📝 Streaming summary displayed in real time as the LLM generates it.
* 📁 Organised output: every job creates a separate folder with the audio, transcript, and a Markdown summary.
* 💾 Save/export the summary Markdown file.
📂 Open output folder with one click.
* 🔄 Job queue – process multiple audio files sequentially without blocking the UI.

# System Requirements
* Operating System: Windows, macOS, or Linux (tested on POP!_OS & Mac OS ARM).
* Python: 3.8 or higher.
* RAM: At least 4 GB (more recommended for larger Whisper models and LLMs).
* Disk Space: Depends on the models used (see Whisper model sizes below).
* Audio Input: A working microphone (for recording) or any audio file for processing.

# Installation
1. Clone or Download the Source
    ```bash
    git clone https://github.com/yourusername/meeting-transcriber.git
    cd meeting-transcriber
    ```

2. Install Python Dependencies
Create a virtual environment (optional but recommended) and install the required packages:

    ```bash
    pip install PyQt5 sounddevice soundfile numpy requests
    ```

If you want to use the faster‑whisper backend (recommended for ease of use), also install:

```bash
pip install faster-whisper
```

> Note: `faster‑whisper` is optional – if not installed, the application will fall back to `whisper‑cli`.

3. Whisper Backend
You have two options – the GUI will let you choose which one to use.

**Option A: faster‑whisper (Python)**
* Advantages: Easy installation, no separate build required.
* Disadvantages: May use more RAM/CPU than the C++ version.

If you installed faster‑whisper via pip, you are ready to go. Do not check the “Use whisper‑cli” box in the GUI.

**Option B: whisper‑cli (C++ from whisper.cpp)**
* Advantages: Faster on some systems, lower memory footprint.
* Disadvantages: Requires compilation and manual model download.

**Step‑by‑step setup:**

1. Clone the whisper.cpp repository:

```bash
git clone https://github.com/ggerganov/whisper.cpp.git ~/whisper.cpp
cd ~/whisper.cpp
```

2. Build the project:

```bash
make -j
```

(On Windows, you may need to use CMake or follow the whisper.cpp build instructions.)

3. Download the desired Whisper models using the provided script:

```bash
cd models
./download-ggml-model.sh tiny   # or base, small, medium, large-v3, etc.
```

The models will be placed in `~/whisper.cpp/models/` as `ggml-*.bin`files.

4. The GUI will automatically detect downloaded models and allow you to select them.
Check the “Use whisper‑cli” box in the GUI when you want to use this backend.

> The application can auto‑download models through the GUI, but only if the download script (```~/whisper.cpp/models/download-ggml-model.sh```) exists and is executable. If you built whisper.cpp as above, this script is present.

# LLM Backend
The application does not include an LLM – you must run one separately on your machine. Supported backends:

* Ollama (default port 11434)
* vLLM (port 8000)
* LM Studio (port 1234)
* llama.cpp (port 8080)

Example: Setting up Ollama
1. Install Ollama from ollama.com.
2. Pull a model (e.g., `gemma4:26b` or `llama3.2`):

```bash
ollama pull gemma4:26b
```

3. Start the Ollama server (it usually runs automatically in the background). Ensure it is listening on `http://localhost:11434`.

For other backends, refer to their respective documentation to start an OpenAI‑compatible API server.

Once the LLM server is running, click the “Refresh” button next to the Backend dropdown in the GUI – it will automatically detect the running service and list available models.

# Configuration
The application has no separate configuration file; all settings are selected through the GUI:

* Audio Device: choose your microphone from the dropdown.
* Whisper Model: select a model size. The GUI shows disk size, memory usage, and download status.
    * If a model is not downloaded, clicking on it will prompt you to download it (requires whisper.cpp download script).
* LLM Backend: select the detected server and choose a model from its list.
* Use whisper‑cli: toggle between the two Whisper backends.

All choices are remembered per session but reset when the application is closed.

# Usage
Launch the application:

```bash
source venv/bin/activate
python transcribe.py
```

# GUI Overview

1. Audio Input – select your microphone device.
2. Whisper Model – choose the model and toggle whisper‑cli mode.
3. LLM Backend – select your running backend and its model.
4. Audio Monitor – shows live waveform and VU meter during recording.
5. Control Buttons:
    * 🎤 Record – start/stop recording.
    * 📂 Load Audio – load an existing audio file for processing.
    * 🗑️ Clear Log – clear the log/summary display.
6. Progress Bar – shows overall job progress.
7. Log / Summary Output – displays logs and streams the generated summary.
8. Save / Open – save the Markdown summary or open the output folder.

# Recording
1. Select your microphone from the Audio Input dropdown.
2. Choose your Whisper model and LLM backend/model.
3. Click 🎤 Record – the waveform and VU meter will show live input.
4. When finished, click ⏹ Stop – the recording is saved and automatically added to the processing queue.

# Loading an Existing Audio File
* Click 📂 Load Audio and select a file (WAV, MP3, M4A, FLAC, OGG, AAC are supported).
* The file is loaded into the processing queue immediately.

# Selecting Models
* Whisper: The dropdown lists all available models (downloaded or not).
    * If you select a model that is not downloaded, a dialog will ask if you want to download it (requires a working `whisper.cpp` setup).
* LLM: Click **Refresh** to detect running servers. Then pick the desired model from the list.

# Processing
* After recording or loading, a job is added to the queue.
* The queue processes one job at a time in the background.
* While processing, you will see:
    * Progress bar updates.
    * Log messages.
    * The summary streaming in the output area in real time.
* Once finished, a Markdown file is saved in a new timestamped folder.

## Viewing Output
* The summary is displayed in the log area.
* You can click 💾 Save Markdown As… to save a copy elsewhere.
* Click 📂 Open Output Folder to open the folder containing all files for the last processed job.

# File Output Structure
Every job creates a dedicated folder under `./transcripts/` named with the current date and time (e.g., `2026-06-18 14.30.45`). The folder contains:

```text
transcripts/
└── 2026-06-18 14.30.45/
    ├── meeting.wav           (the recorded or loaded audio file)
    ├── transcript.txt        (raw transcription text)
    ├── summary.md            (the final Markdown summary with metadata)
    └── (additional files if whisper-cli is used)
```

The summary.md includes:
* Timestamp
* Audio file path
* Whisper model used
* LLM backend and model
* The generated summary (from the LLM)
* The full transcript (verbatim)

# Troubleshooting
| Problem   |    Possible Solution |
|----------|-------------|
| No audio devices shown |  Ensure your microphone is connected. On Linux, you may need to install `portaudio` or `pulseaudio`. |
| Whisper model not downloading |    Check that `~/whisper.cpp/models/download-ggml-model.sh` exists and is executable. You may need to build whisper.cpp first.   |
| faster‑whisper not found | Install it via `pip install faster-whisper`, or use `whisper‑cl`. |
| LLM backend not detected | Verify the server is running and the correct port is open. Click **Refresh** in the GUI. |
| Transcription fails with whisper-cli | Ensure the model file exists at `~/whisper.cpp/models/ggml-*.bin` and that `whisper-cli` is built. |
| Memory errors during transcription/summarisation | Use smaller models (e.g., `tiny`, `base`) or lower‑memory LLM models. |
| Audio file not loading for visualisation | Only WAV, MP3, M4A, FLAC, OGG, AAC are supported via `soundfile` and `ffmpeg`. Install `ffmpeg` if necessary. |
| Job queue gets stuck | Restart the application. The queue is cleared on exit. |

For further help, please open an issue on the project repository.

# License
This project is released under the MIT License. See the LICENSE file for details.

*Enjoy transcribing and summarizing your meetings! 🚀*
