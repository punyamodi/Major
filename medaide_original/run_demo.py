"""CLI demo for the recreated original MedAide architecture."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from medaide_original.pipeline import MedAideOriginalPipeline


def main() -> None:
    pipeline = MedAideOriginalPipeline()
    result = pipeline.run_sync(
        query="I have fever, cough, and sore throat for 3 days. What could this be?",
        patient_id="original_demo_patient",
    )
    print("=== MedAide Original Demo ===")
    print(result.final_response)
    print("\n--- Metadata ---")
    print(json.dumps(
        {
            "tier": result.tier,
            "n_agents_used": result.n_agents_used,
            "intents": result.intents,
            "top_category": result.top_category,
            "confidence": round(result.confidence, 4),
            "latency_ms": round(result.latency_ms, 2),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
