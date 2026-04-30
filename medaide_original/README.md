# MedAide Original Architecture (Recreated)

This folder contains a runnable recreation of the original MedAide architecture using only:

- **M1** AMQU (query decomposition)
- **M2** HDIO (intent classification)
- **M3** DMACN (single-agent execution)

It intentionally excludes MedAide+ enhancements (M4–M7) to preserve the original 3-module flow.

## Prerequisites
- Python environment with dependencies installed from medaide_plus/requirements.txt
- An LLM provider configured in medaide_original/config.yaml
	- Default is Ollama at http://localhost:11434
	- Alternatively set OPENAI_API_KEY and switch llm.provider to openai

## Setup

From repository root:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r .\medaide_plus\requirements.txt
```

If you use Ollama, make sure the model is available:

```bash
ollama pull qwen3:8b
```

## Run

From repository root:

```powershell
& .\.venv\Scripts\python.exe .\medaide_original\run_demo.py
```

This run path is production-strict: it requires a real configured LLM provider and a non-empty persisted knowledge base.

## Configuration
- Config file: medaide_original/config.yaml
- Knowledge base: medaide_plus/data/knowledge_base.json
- Logs: medaide_plus/logs/medaide_original.log

## Programmatic usage

```python
from medaide_original.pipeline import MedAideOriginalPipeline

pipeline = MedAideOriginalPipeline()
result = pipeline.run_sync("I have recurring headaches and nausea. What should I check first?")
print(result.final_response)
```
