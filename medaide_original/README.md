# MedAide Original Architecture (Recreated)

This folder contains a runnable recreation of the original MedAide architecture using only:

- **M1** AMQU (query decomposition)
- **M2** HDIO (intent classification)
- **M3** DMACN (single-agent execution)

It intentionally excludes MedAide+ enhancements (M4–M7) to preserve the original 3-module flow.

## Run

From repository root:

```powershell
& .\.venv\Scripts\python.exe .\medaide_original\run_demo.py
```

This run path is production-strict: it requires a real configured LLM provider and a non-empty persisted knowledge base.

## Programmatic usage

```python
from medaide_original.pipeline import MedAideOriginalPipeline

pipeline = MedAideOriginalPipeline()
result = pipeline.run_sync("I have recurring headaches and nausea. What should I check first?")
print(result.final_response)
```
