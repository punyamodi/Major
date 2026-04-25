"""
MedAide+ Gradio Demo Application

Full web interface for the MedAide+ multi-agent medical framework.
Run with: python demo/app.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

# Lazy-loaded pipeline
_pipeline = None


def get_pipeline():
    """Lazy-load the MedAide+ pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            from medaide_plus.pipeline import MedAidePlusPipeline
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            _pipeline = MedAidePlusPipeline(
                config_path=str(config_path) if config_path.exists() else None,
                patient_id="demo_patient",
            )
        except Exception as e:
            print(f"Warning: Could not initialize pipeline: {e}")
            return None
    return _pipeline


def process_query(
    query: str,
    patient_id: str,
    session_id: str,
) -> Tuple[str, str, str, str, str, str]:
    """
    Process a medical query through the MedAide+ pipeline.

    Returns:
        Tuple of (response, annotated_response, metadata_json, agents_info, claims_info, patient_graph_info)
    """
    if not query.strip():
        return (
            "Please enter a medical question.",
            "",
            "{}",
            "No agents activated.",
            "No claims verified.",
            "No patient graph data.",
        )

    if not patient_id.strip():
        patient_id = "demo_patient"

    if not session_id.strip():
        session_id = patient_id

    pipeline = get_pipeline()
    if pipeline is None:
        mock_response = (
            f"**[Demo Mode — Pipeline not initialized]**\n\n"
            f"Your query: '{query}'\n\n"
            f"In production, this would be processed by the full MedAide+ pipeline with:\n"
            f"- M1 AMQU: Query decomposition\n"
            f"- M2 HDIO: Intent classification\n"
            f"- M3 DMACN: Parallel multi-agent execution\n"
            f"- M4 PLMM: Patient history injection\n"
            f"- M5 HDFG: Hallucination verification\n"
            f"- M6 AQCR: Complexity routing\n"
            f"- M7 MIET: Multi-turn tracking\n\n"
            f"Please ensure all dependencies are installed and OPENAI_API_KEY is set."
        )
        return (
            mock_response,
            mock_response,
            json.dumps({"status": "demo_mode"}, indent=2),
            "Pipeline not initialized.",
            "No verification performed.",
            "No patient data.",
        )

    try:
        # Run pipeline
        result = asyncio.run(
            pipeline.run(
                query=query,
                patient_id=patient_id,
                session_id=session_id,
            )
        )

        # Format main response
        main_response = result.final_response

        # Format annotated response
        annotated = result.annotated_response

        # Format metadata
        metadata = {
            "tier": result.tier,
            "n_agents": result.n_agents_used,
            "intents": result.intents,
            "top_category": result.top_category,
            "confidence": round(result.confidence, 3),
            "hallucination_rate": round(result.hallucination_rate, 3),
            "latency_ms": round(result.latency_ms, 1),
            "subqueries": result.subqueries,
            "modules": result.metadata,
        }
        metadata_json = json.dumps(metadata, indent=2)

        # Format agents info
        if result.agent_names:
            agents_info = (
                f"**Complexity Tier:** {result.tier}\n"
                f"**Agents Activated:** {result.n_agents_used}\n\n"
                + "\n".join(f"✓ {agent}" for agent in result.agent_names)
            )
        else:
            agents_info = f"**Tier:** {result.tier}\n**No agents completed successfully.**"

        # Format claims info
        if result.verified_claims:
            claims_lines = [
                f"**Hallucination Rate:** {result.hallucination_rate:.1%}\n"
            ]
            for claim in result.verified_claims:
                icon = "✅" if claim["supported"] else "⚠️"
                status = "SUPPORTED" if claim["supported"] else "FLAGGED"
                claims_lines.append(
                    f"{icon} [{status}] {claim['claim'][:120]}"
                )
            claims_info = "\n".join(claims_lines)
        else:
            claims_info = "No specific claims extracted for verification."

        # Format patient graph info
        try:
            summary = pipeline.m4_plmm.get_graph_summary(patient_id)
            graph_info = (
                f"**Patient ID:** {patient_id}\n"
                f"**Graph Nodes:** {summary['n_nodes']}\n"
                f"**Graph Edges:** {summary['n_edges']}\n"
                f"**Node Types:**\n"
                + "\n".join(
                    f"  • {t}: {c}" for t, c in summary.get("node_types", {}).items()
                )
            )
        except Exception:
            graph_info = f"Patient ID: {patient_id}"

        return main_response, annotated, metadata_json, agents_info, claims_info, graph_info

    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}\n\nPlease check logs for details."
        return error_msg, error_msg, "{}", "Error", "Error", "Error"


def create_interface() -> gr.Blocks:
    """Create the Gradio web interface."""

    with gr.Blocks(
        title="MedAide+ Medical AI Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
        ),
        css="""
        .main-header { text-align: center; padding: 20px; }
        .query-box textarea { font-size: 14px; }
        .response-box { background: #f8f9fa; border-radius: 8px; }
        """,
    ) as demo:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🏥 MedAide+ Medical AI Assistant</h1>
            <p style="color: #666;">
                Improved LLM-based multi-agent medical framework with 7 enhancement modules<br>
                <em>⚠️ For educational purposes only. Always consult a qualified healthcare professional.</em>
            </p>
        </div>
        """)

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Your Medical Question")
                query_input = gr.Textbox(
                    label="Medical Query",
                    placeholder="e.g., I've been having chest pain and shortness of breath for 2 days...",
                    lines=4,
                    elem_classes=["query-box"],
                )

                with gr.Row():
                    patient_id_input = gr.Textbox(
                        label="Patient ID",
                        placeholder="patient_001",
                        value="demo_patient",
                        scale=1,
                    )
                    session_id_input = gr.Textbox(
                        label="Session ID",
                        placeholder="session_001",
                        value="demo_session",
                        scale=1,
                    )

                submit_btn = gr.Button(
                    "🔍 Analyze Query",
                    variant="primary",
                    size="lg",
                )

                # Sample queries
                gr.Markdown("### 📋 Sample Queries")
                sample_queries = [
                    "I have severe chest pain radiating to my left arm and sweating profusely.",
                    "What is the correct dosage of metformin for type 2 diabetes?",
                    "I'm taking warfarin and my doctor prescribed fluconazole. Is this safe?",
                    "I had knee replacement surgery 6 weeks ago. When can I return to normal activities?",
                    "My HbA1c is 8.2% and fasting glucose is 160 mg/dL. What does this mean?",
                ]
                for sq in sample_queries:
                    gr.Button(f"📌 {sq[:60]}...", size="sm").click(
                        fn=lambda x=sq: x,
                        outputs=query_input,
                    )

            # Right column: Main response
            with gr.Column(scale=3):
                gr.Markdown("### 🤖 AI Response")
                response_output = gr.Markdown(
                    label="Response",
                    value="*Submit a query to see the response here.*",
                    elem_classes=["response-box"],
                )

        # Tabbed detail views
        gr.Markdown("---")
        gr.Markdown("### 📊 Analysis Details")

        with gr.Tabs():
            with gr.Tab("🔍 Verification"):
                gr.Markdown("Claims extracted and verified for factual support:")
                annotated_output = gr.Markdown(
                    value="*Submit a query to see verified claims.*"
                )

            with gr.Tab("🤖 Agent Activation"):
                gr.Markdown("Which agents were activated and at what complexity tier:")
                agents_output = gr.Markdown(
                    value="*Submit a query to see agent details.*"
                )

            with gr.Tab("🧠 Patient Graph"):
                gr.Markdown("Patient knowledge graph statistics (M4 PLMM):")
                graph_output = gr.Markdown(
                    value="*Submit a query to see patient graph.*"
                )

            with gr.Tab("⚙️ Pipeline Metadata"):
                gr.Markdown("Full pipeline metadata including intents, tier, and module outputs:")
                metadata_output = gr.Code(
                    value="{}",
                    language="json",
                    label="Pipeline Metadata (JSON)",
                )

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; color: #888; font-size: 12px;">
            <p>MedAide+ | arXiv 2410.12532 extension | 7 Enhancement Modules</p>
            <p>M1 AMQU | M2 HDIO | M3 DMACN | M4 PLMM | M5 HDFG | M6 AQCR | M7 MIET</p>
        </div>
        """)

        # Wire up the submit button
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, patient_id_input, session_id_input],
            outputs=[
                response_output,
                annotated_output,
                metadata_output,
                agents_output,
                graph_output,
                gr.Textbox(visible=False),  # placeholder for 6th output
            ],
        )

    return demo


def main():
    """Launch the Gradio demo."""
    print("Starting MedAide+ Demo...")
    print("Initializing pipeline (this may take a moment)...")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
