# Feature To-Do List

## Recording
- [x] Pause/resume recording — currently only start/stop. A long meeting with a break means either recording through silence or stitching two separate files manually.

## Live Transcription
- [x] Rolling/live transcript while recording, instead of only after recording stops.

## Diarization
- [ ] UI to rename raw speaker labels (`SPEAKER_00`, `SPEAKER_01`, from [diarization.py:74](src/diarization.py#L74)) to actual names — every transcript currently needs manual find/replace to be readable.
- [ ] Per-source speaker attribution as a token-free diarization alternative — tag each transcript segment with whichever mixer *source* (not voice) was loudest during that span. No new dependencies, works instantly, but only for "one mic per person" recordings (multi-source mixer already supports this), not a single shared mic.
- [ ] (Low priority) NeMo (NVIDIA) diarization as an ungated alternative to pyannote's gated model — no HF token/terms-acceptance needed, but pulls in the much heavier `nemo_toolkit` and works best with a GPU.

## Export
- [ ] SRT/VTT subtitle export — Whisper already produces timestamped segments, useful for captions, not just minutes.
- [ ] PDF/DOCX export — currently only "Save As" of the raw markdown text ([main.py:623](src/main.py#L623)).

## LLM Backends
- [ ] Optional cloud LLM fallback (e.g. Anthropic/OpenAI/Google/Deepseek/Groq) for users without a capable local machine. Trades off the app's local-privacy premise, so gate it behind a clear on-screen disclaimer when enabled. Depends on the secure secret storage item below — don't ship cloud API keys on the plaintext QSettings pattern.

## Security / Secrets
- [ ] Move secret storage (existing HF token, plus any future cloud LLM API keys) off plaintext QSettings and onto the OS-native encrypted credential store via the `keyring` library, across all three platforms: Secret Service/GNOME Keyring or KWallet on Linux, Keychain on macOS, Credential Manager on Windows. Needs a graceful fallback for environments with no keyring backend available (minimal/headless Linux, bare WSL, some CI containers) — warn and refuse to save rather than silently falling back to plaintext.

## Layout
Two-column layout instead of one long vertical stack of group boxes, rolled out in two steps:
- [x] Step 1: Audio Sources on the left, Audio Monitor (mixed output) to its right.
- [x] Step 2: Whisper Model on the left, LLM Backend to its right.

## Post-summary Actions - Nice to have
- [ ] Ways to push the summary out — email, clipboard-to-Slack, calendar follow-up, task-tracker for action items. Currently read/copy/save only.
