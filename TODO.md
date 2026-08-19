# Feature To-Do List

## Recording
- [x] Pause/resume recording — currently only start/stop. A long meeting with a break means either recording through silence or stitching two separate files manually.

## Live Transcription
- [ ] Rolling/live transcript while recording, instead of only after recording stops.

## Diarization
- [ ] UI to rename raw speaker labels (`SPEAKER_00`, `SPEAKER_01`, from [diarization.py:74](src/diarization.py#L74)) to actual names — every transcript currently needs manual find/replace to be readable.

## Export
- [ ] SRT/VTT subtitle export — Whisper already produces timestamped segments, useful for captions, not just minutes.
- [ ] PDF/DOCX export — currently only "Save As" of the raw markdown text ([main.py:623](src/main.py#L623)).

## LLM Backends
- [ ] Optional cloud LLM fallback (e.g. Anthropic/OpenAI/Google/Deepseek/Groq) for users without a capable local machine. Trades off the app's local-privacy premise, so gate it behind a clear on-screen disclaimer when enabled.

## Post-summary Actions - Nice to have
- [ ] Ways to push the summary out — email, clipboard-to-Slack, calendar follow-up, task-tracker for action items. Currently read/copy/save only.
