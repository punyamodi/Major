# MedAide+ Ollama UI

This folder provides a standalone custom UI to test the **final MedAide+ architecture** with local Ollama models.

## Running the App

### Start

From repository root:

```powershell
& .\.venv\Scripts\python.exe .\medaide_plus_ollama_ui\app.py
```

Open:

http://localhost:7861

### Stop

To stop the application, press `Ctrl+C` in the terminal where the app is running.

## Notes

- Uses `medaide_plus/config/config.yaml` as base, but forces `llm.provider = ollama`.
- Writes runtime config to `medaide_plus_ollama_ui/runtime_config.yaml`.
- Uses MedAide+ data paths from `medaide_plus/data/...`.
- Includes run diagnostics, provider health visibility, and metadata tabs.
- Requires a live Ollama server and model availability; if Ollama is unreachable, the run fails with an explicit error.
- Does not use mock/demo responses in production mode.
