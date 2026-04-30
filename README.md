# MedAide+ Workspace

This repository contains multiple MedAide variants, a local UI, and paper artifacts.

## Common setup (shared environment)
1. Create and activate a virtual environment.
2. Install dependencies from medaide_plus/requirements.txt.
3. Optional: install the spaCy model for better NER.

### Windows (PowerShell)
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r .\medaide_plus\requirements.txt
python -m spacy download en_core_web_sm
```

### Linux/Mac
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r medaide_plus/requirements.txt
python -m spacy download en_core_web_sm
```

## LLM provider prerequisites
- Default configs use Ollama at http://localhost:11434.
- Ensure Ollama is running and the model is pulled (default: qwen3:8b).

```bash
ollama pull qwen3:8b
```

To use OpenAI or other providers, edit:
- medaide_plus/config/config.yaml
- medaide_original/config.yaml
and set the related environment variables (for example, OPENAI_API_KEY).

## Run MedAide+ (main pipeline)
Start the Gradio demo:

```powershell
& .\.venv\Scripts\python.exe .\medaide_plus\demo\app.py
```

Open http://localhost:7860

For API usage, tests, and evaluation commands, see medaide_plus/README.md.

## Run MedAide Original (3-module baseline)
```powershell
& .\.venv\Scripts\python.exe .\medaide_original\run_demo.py
```

## Run MedAide+ Ollama UI (local model UI)
```powershell
& .\.venv\Scripts\python.exe .\medaide_plus_ollama_ui\app.py
```

Open http://localhost:7861

## Tests and evaluation (MedAide+)
```powershell
cd .\medaide_plus
pytest tests -v
python -m evaluation.benchmark_runner --limit 20
```

## Paper utilities
These scripts are for updating LaTeX tables and automating benchmark runs:

```powershell
python .\update_paper_results.py
python .\watch_and_run_gpt.py
```

Notes:
- update_paper_results.py expects a medaide_plus_trial folder with Ollama results and a MedAidePlus_Phase4_Paper.tex file at the repo root.
- watch_and_run_gpt.py also requires pdflatex on PATH.
Adjust the paths at the top of the scripts if your layout differs.

## Folder guide
- medaide_plus: main MedAide+ pipeline, evaluation, and demo
- medaide_original: original 3-module baseline
- medaide_plus_ollama_ui: local Ollama UI
- Final: slides and final reports (documents only)
- Submissions: paper PDFs, LaTeX sources, and submission artifacts
