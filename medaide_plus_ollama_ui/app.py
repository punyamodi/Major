"""Standalone custom Ollama UI for MedAide+."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import yaml

ROOT = Path(__file__).resolve().parent.parent
MEDAIDE_ROOT = ROOT / "medaide_plus"
BASE_CONFIG_PATH = MEDAIDE_ROOT / "config" / "config.yaml"
RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent / "runtime_config.yaml"

# Ensure medaide_plus package is importable regardless of launch cwd.
sys.path.insert(0, str(MEDAIDE_ROOT))

_pipeline = None
_pipeline_key: Optional[Tuple[str, str, int, int, float]] = None

SAMPLE_QUERIES = [
    "I have chest discomfort with sweating and pain in my left arm for 2 hours. What should I do now?",
    "My blood pressure has been 150/95 for a week. What lifestyle changes and medicines are usually considered?",
    "I am taking warfarin and was prescribed fluconazole. Is this combination risky?",
    "My fasting sugar is 165 mg/dL and HbA1c is 8.1%. How should I interpret this?",
]

QUERY_TEMPLATES = {
    "Emergency triage": "I have crushing chest pain with sweating and shortness of breath for 30 minutes. What should I do immediately?",
    "Medication safety": "I take warfarin 5 mg daily and was prescribed fluconazole. Is this interaction dangerous?",
    "Lab interpretation": "My HbA1c is 8.4% and fasting glucose is 170 mg/dL. What does this mean and what should I do next?",
    "Follow-up planning": "I completed chemotherapy 3 months ago. When should I schedule follow-up scans and what warning symptoms matter most?",
}

MODEL_PRESETS = {
    "qwen3:8b (Balanced, Recommended)": ("qwen3:8b", 2048, 1024, 0.3),
    "gemma3:4b (Fast)": ("gemma3:4b", 1536, 768, 0.25),
    "phi4-reasoning:14b (High reasoning)": ("phi4-reasoning:14b", 2048, 1024, 0.2),
    "gpt-oss:20b (Large local model)": ("gpt-oss:20b", 2048, 1024, 0.2),
}

CUSTOM_CSS = """
.hero {
  border-radius: 18px;
  padding: 22px 24px;
  background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 70%, #2563eb 100%);
  color: #f8fafc;
  margin-bottom: 16px;
}
.hero h1 { margin: 0 0 8px 0; font-size: 30px; letter-spacing: 0.1px; }
.hero p { margin: 0; color: #dbeafe; line-height: 1.4; }
.panel {
  border: 1px solid #d4e3f3;
  border-radius: 16px;
  background: linear-gradient(180deg, #f8fbff 0%, #f4f8ff 100%);
  padding: 14px;
}
.status {
  border: 1px solid #d4e3f3;
  border-radius: 16px;
  background: #ffffff; 
  padding: 8px;
  color: #000000 !important; 
  --body-text-color: #000000; 
  --body-text-color-subdued: #334155;
}
.status .result-text,
.status .result-text * {
  color: #000000 !important; 
}
.status .result-text a {
  color: #1d4ed8 !important;
}
.status .result-text code {
  background: #f1f5f9;
  color: #0f172a !important;
  border-radius: 6px;
  padding: 1px 4px;
}
.status .result-text pre code {
  display: block;
  padding: 10px;
}
.status-cards {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}
.status-card {
  border-radius: 12px;
  border: 1px solid #dbe8f8;
  background: #f7fbff;
  padding: 10px 12px;
}
.status-label {
  font-size: 12px;
  color: #5b6f8e;
  margin-bottom: 4px;
}
.status-value {
  font-size: 15px;
  font-weight: 600;
  color: #0f172a;
}
.error-banner {
  border: 1px solid #f3c2c2;
  border-radius: 12px;
  background: #fff5f5;
  color: #7f1d1d;
  padding: 10px 12px;
  margin-bottom: 10px;
}
"""

THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")


def _build_runtime_config(
    ollama_base_url: str,
    ollama_model: str,
    num_ctx: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config.setdefault("llm", {})
    config["llm"]["provider"] = "ollama"
    config["llm"].setdefault("ollama", {})
    config["llm"]["ollama"].update(
        {
            "base_url": ollama_base_url.strip() or "http://localhost:11434",
            "model": ollama_model.strip() or "qwen3:8b",
            "num_ctx": int(num_ctx),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
    )

    config.setdefault("modules", {})
    config["modules"].setdefault("amqu", {})
    config["modules"]["amqu"].setdefault("use_flan_t5", False)
    config["modules"]["amqu"].setdefault("use_consistency_filter", False)
    config["modules"].setdefault("hdio", {})
    config["modules"]["hdio"].setdefault("use_biobert", False)
    config["modules"]["hdio"].setdefault("use_sentence_transformer", False)
    config["modules"]["hdio"].setdefault("use_gat", False)
    config["modules"].setdefault("plmm", {})
    config["modules"].setdefault("hdfg", {})
    config["modules"]["hdfg"].setdefault("enabled", False)
    config["modules"]["plmm"]["graph_storage_path"] = str(
        MEDAIDE_ROOT / "data" / "patient_graphs"
    )

    config.setdefault("knowledge_base", {})
    config["knowledge_base"]["kb_path"] = str(MEDAIDE_ROOT / "data" / "knowledge_base.json")
    config["knowledge_base"]["faiss_index_path"] = str(MEDAIDE_ROOT / "data" / "faiss_index")

    config.setdefault("logging", {})
    logs_dir = MEDAIDE_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    config["logging"]["file"] = str(logs_dir / "medaide_plus_ollama_ui.log")

    return config


def get_pipeline(
    ollama_base_url: str,
    ollama_model: str,
    num_ctx: int,
    max_tokens: int,
    temperature: float,
):
    global _pipeline, _pipeline_key

    key = (
        ollama_base_url.strip() or "http://localhost:11434",
        ollama_model.strip() or "qwen3:8b",
        int(num_ctx),
        int(max_tokens),
        float(temperature),
    )

    if _pipeline is not None and _pipeline_key == key:
        return _pipeline

    runtime_config = _build_runtime_config(
        ollama_base_url=key[0],
        ollama_model=key[1],
        num_ctx=key[2],
        max_tokens=key[3],
        temperature=key[4],
    )
    with open(RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_config, f, sort_keys=False, allow_unicode=False)

    from medaide_plus.pipeline import MedAidePlusPipeline

    _pipeline = MedAidePlusPipeline(
        config_path=str(RUNTIME_CONFIG_PATH),
        patient_id="ollama_ui_patient",
    )
    _pipeline_key = key
    return _pipeline


def _detect_provider_health(pipe) -> Tuple[bool, str]:
    provider = getattr(pipe, "_llm_provider", None)
    if provider is None:
        return False, "No active provider object."
    try:
        healthy = asyncio.run(provider.health_check())
        if healthy:
            return True, "Ollama endpoint is reachable."
        return False, "Ollama endpoint unavailable."
    except Exception as e:
        return False, f"Health check failed: {e}"


def _build_status_cards(metadata: Dict[str, Any]) -> str:
    def _fmt(key: str, default: str = "-") -> str:
        value = metadata.get(key, default)
        return str(value) if value is not None else default

    latency = metadata.get("latency_ms")
    latency_value = f"{latency:.1f} ms" if isinstance(latency, (int, float)) else "-"
    confidence = metadata.get("confidence")
    confidence_value = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "-"
    hallucination = metadata.get("hallucination_rate")
    hallucination_value = f"{hallucination:.3f}" if isinstance(hallucination, (int, float)) else "-"

    return (
        '<div class="status-cards">'
        f'<div class="status-card"><div class="status-label">Runtime</div><div class="status-value">{_fmt("runtime_mode")}</div></div>'
        f'<div class="status-card"><div class="status-label">Model</div><div class="status-value">{_fmt("model")}</div></div>'
        f'<div class="status-card"><div class="status-label">Tier</div><div class="status-value">{_fmt("tier")}</div></div>'
        f'<div class="status-card"><div class="status-label">Latency</div><div class="status-value">{latency_value}</div></div>'
        f'<div class="status-card"><div class="status-label">Confidence</div><div class="status-value">{confidence_value}</div></div>'
        f'<div class="status-card"><div class="status-label">Hallucination Rate</div><div class="status-value">{hallucination_value}</div></div>'
        "</div>"
    )


def _error_banner(message: str) -> str:
    return f'<div class="error-banner"><strong>Run failed:</strong> {message}</div>'


def _apply_model_preset(preset_name: str):
    return MODEL_PRESETS.get(preset_name, MODEL_PRESETS["qwen3:8b (Balanced, Recommended)"])


def _load_template(template_name: str) -> str:
    return QUERY_TEMPLATES.get(template_name, "")


def run_query(
    query: str,
    patient_id: str,
    session_id: str,
    ollama_model: str,
    ollama_base_url: str,
    num_ctx: int,
    max_tokens: int,
    temperature: float,
):
    if not query.strip():
        return (
            "Please enter a medical query to run.",
            "No verified response yet.",
            "No run yet.",
            "{}",
            "Diagnostics unavailable until first run.",
            _error_banner("Enter a query before running."),
        )

    patient_id = patient_id.strip() or "ollama_ui_patient"
    session_id = session_id.strip() or patient_id

    try:
        pipe = get_pipeline(
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            num_ctx=int(num_ctx),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        provider_healthy, provider_status = _detect_provider_health(pipe)
        if not provider_healthy:
            raise RuntimeError(
                f"Ollama is not reachable at {ollama_base_url}. "
                "Start the Ollama server and ensure the model is available."
            )
        result = asyncio.run(pipe.run(query=query, patient_id=patient_id, session_id=session_id))
        runtime_mode = "Live Ollama response"

        metadata = {
            "provider": "ollama",
            "model": ollama_model,
            "base_url": ollama_base_url,
            "provider_healthy": provider_healthy,
            "runtime_mode": runtime_mode,
            "tier": result.tier,
            "n_agents_used": result.n_agents_used,
            "intents": result.intents,
            "top_category": result.top_category,
            "confidence": round(result.confidence, 4),
            "hallucination_rate": round(result.hallucination_rate, 4),
            "latency_ms": round(result.latency_ms, 2),
            "patient_id": patient_id,
            "session_id": session_id,
        }

        summary = (
            f"### Run Summary\n"
            f"- **Runtime mode:** {runtime_mode}\n"
            f"- **Tier:** {result.tier}\n"
            f"- **Agents:** {result.n_agents_used}\n"
            f"- **Top category:** {result.top_category}\n"
            f"- **Latency:** {result.latency_ms:.1f} ms\n"
            f"- **Confidence:** {result.confidence:.3f}\n"
            f"- **Hallucination rate:** {result.hallucination_rate:.3f}"
        )

        diagnostics = (
            f"### Diagnostics\n"
            f"- **Provider status:** {provider_status}\n"
            f"- **Runtime config file:** `{RUNTIME_CONFIG_PATH}`\n"
            f"- **Model:** `{ollama_model}`\n"
            f"- **Base URL:** `{ollama_base_url}`\n"
            f"- **num_ctx / max_tokens / temperature:** `{int(num_ctx)} / {int(max_tokens)} / {float(temperature):.2f}`"
        )

        return (
            result.final_response,
            result.annotated_response,
            summary,
            json.dumps(metadata, indent=2),
            diagnostics,
            _build_status_cards(metadata),
        )
    except Exception as e:
        return (
            f"Error while running MedAide+ with Ollama: {e}",
            "No verified response available because the run failed.",
            "Run failed.",
            "{}",
            "Check provider availability and runtime config.",
            _error_banner(str(e)),
        )


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="MedAide+ Ollama UI") as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>MedAide+ Ollama Studio</h1>
              <p>Custom runtime UI for local-model testing with full MedAide+ pipeline metadata.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2, elem_classes=["panel"]):
                query = gr.Textbox(
                    label="Medical Query",
                    lines=7,
                    placeholder="Describe symptoms, medications, history, and the exact clinical question.",
                )
                gr.Examples(
                    examples=[[s] for s in SAMPLE_QUERIES],
                    inputs=query,
                    label="Quick prompts",
                )
                with gr.Row():
                    template_selector = gr.Dropdown(
                        label="Clinical template",
                        choices=list(QUERY_TEMPLATES.keys()),
                        value="Emergency triage",
                    )
                    use_template_btn = gr.Button("Use template", variant="secondary")
                with gr.Row():
                    patient_id = gr.Textbox(label="Patient ID", value="patient_001")
                    session_id = gr.Textbox(label="Session ID", value="session_001")

                with gr.Accordion("Ollama Runtime Settings", open=True):
                    preset_selector = gr.Dropdown(
                        label="Model preset",
                        choices=list(MODEL_PRESETS.keys()),
                        value="qwen3:8b (Balanced, Recommended)",
                    )
                    ollama_model = gr.Textbox(label="Model", value="qwen3:8b")
                    ollama_base_url = gr.Textbox(label="Base URL", value="http://localhost:11434")
                    num_ctx = gr.Number(label="Context window (num_ctx)", value=2048, precision=0)
                    max_tokens = gr.Number(label="Max tokens", value=1024, precision=0)
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                    )

                with gr.Row():
                    submit = gr.Button("Run MedAide+", variant="primary", size="lg")
                    clear = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=3, elem_classes=["status"]):
                status_cards = gr.HTML(
                    value=_error_banner("No run yet. Submit a query to see live run metrics.")
                )
                with gr.Tabs():
                    with gr.Tab("Assistant Response"):
                        response = gr.Markdown(value="Submit a query to start.", elem_classes=["result-text"])
                    with gr.Tab("Verified Response"):
                        annotated_response = gr.Markdown(value="No verified response yet.", elem_classes=["result-text"])
                    with gr.Tab("Run Summary"):
                        summary = gr.Markdown(value="No run yet.", elem_classes=["result-text"])
                    with gr.Tab("Metadata JSON"):
                        metadata = gr.Code(value="{}", language="json", label="Metadata")
                    with gr.Tab("Diagnostics"):
                        diagnostics = gr.Markdown(value="Diagnostics unavailable until first run.", elem_classes=["result-text"])

        submit.click(
            fn=run_query,
            inputs=[
                query,
                patient_id,
                session_id,
                ollama_model,
                ollama_base_url,
                num_ctx,
                max_tokens,
                temperature,
            ],
            outputs=[response, annotated_response, summary, metadata, diagnostics, status_cards],
        )
        use_template_btn.click(
            fn=_load_template,
            inputs=[template_selector],
            outputs=[query],
            queue=False,
        )
        preset_selector.change(
            fn=_apply_model_preset,
            inputs=[preset_selector],
            outputs=[ollama_model, num_ctx, max_tokens, temperature],
            queue=False,
        )
        clear.click(
            fn=lambda: (
                "",
                "No verified response yet.",
                "No run yet.",
                "{}",
                "Diagnostics unavailable until first run.",
                _error_banner("No run yet. Submit a query to see live run metrics."),
            ),
            outputs=[response, annotated_response, summary, metadata, diagnostics, status_cards],
            queue=False,
        )

    return demo


def main():
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
