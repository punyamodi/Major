"""
MedAide+ Phase 4: Composite-Intent Benchmark Generator

Generates a synthetic benchmark that is protocol-aligned with the original MedAide
paper's evaluation design:
  - 4 categories × 500 instances = 2,000 total instances
  - Each instance has COMPOSITE INTENTS (2–3 intents combined per query)
    matching the paper's 17-intent taxonomy
  - Queries are clinically realistic and span diverse medical scenarios
  - Reference answers generated via GPT-4o (if API key available) or
    template-based fallback

WHY NOT THE ORIGINAL MEDIAIDE DATASET?
  The original MedAide paper (arXiv:2410.12532v3, Section 4.1) built its own
  PRIVATE benchmark of composite-intent instances. It is not publicly released.
  MedQuAD is single-intent and not used in the paper. This generator is therefore
  a synthetic proxy that mirrors the paper's benchmark structure using the same
  intent taxonomy and composite query design.

COMPARISON PROTOCOL (matching paper Tables 1–4):
  The paper evaluates as a plug-and-play framework by comparing:
    (A) Vanilla LLM  — raw GPT-4o without any framework
    (B) LLM + MedAide (original 3-module: RIE + IPM + RAC)
    (C) LLM + MedAide+ (our 7-module extension)
  All conditions use the same base LLM (GPT-4o by default).

Usage:
    python -m evaluation.fetch_benchmark              # full 500/category
    python -m evaluation.fetch_benchmark --samples 20 # quick test
    python -m evaluation.fetch_benchmark --with-gpt-ref  # GPT-4o ground-truth
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("fetch_benchmark")

# ─── MedAide 17-intent taxonomy (from paper, Section 3.2) ─────────────────────
# MedAide+ adds Follow_up_Scheduling -> 18 intents total
INTENTS_BY_CATEGORY: Dict[str, List[str]] = {
    "Pre-Diagnosis": [
        "Symptom_Triage",
        "Risk_Assessment",
        "Department_Suggestion",
        "Health_Inquiry",
    ],
    "Diagnosis": [
        "Differential_Diagnosis",
        "Symptom_Analysis",
        "Test_Interpretation",
        "Etiology_Detection",
    ],
    "Medication": [
        "Drug_Interaction",
        "Dosage_Recommendation",
        "Contraindication_Check",
        "Drug_Counseling",
        "Prescription_Review",
    ],
    "Post-Diagnosis": [
        "Rehabilitation_Advice",
        "Lifestyle_Guidance",
        "Progress_Tracking",
        "Care_Support",
        "Follow_up_Scheduling",   # MedAide+ addition (18th intent)
    ],
}

# ─── Composite-intent query templates (2–3 intents per query) ─────────────────
# Each entry: (query_template, [intent1, intent2, ...])
# Templates use {age}, {condition}, {med}, {symptom}, {lab}, {value} placeholders.
# Following the paper: queries are clinically realistic composite queries.

PRE_DIAGNOSIS_TEMPLATES: List[Tuple[str, List[str]]] = [
    # Symptom_Triage + Risk_Assessment
    ("I am {age} years old and have been experiencing {symptom} for the past {days} days. "
     "My {relative} had {condition}. How serious is this and what is my risk of developing it?",
     ["Symptom_Triage", "Risk_Assessment"]),

    # Symptom_Triage + Department_Suggestion
    ("I've had {symptom} along with {symptom2} for {days} days. "
     "Which type of doctor should I see, and how urgently?",
     ["Symptom_Triage", "Department_Suggestion"]),

    # Risk_Assessment + Health_Inquiry
    ("I have {condition} and {condition2}. What is my risk for {condition3}, "
     "and what does elevated {lab} mean in my case?",
     ["Risk_Assessment", "Health_Inquiry"]),

    # Symptom_Triage + Risk_Assessment + Department_Suggestion
    ("My {age}-year-old {relative} suddenly developed {symptom}, {symptom2}, and {symptom3}. "
     "Given their history of {condition}, how serious is this and which emergency service should we contact?",
     ["Symptom_Triage", "Risk_Assessment", "Department_Suggestion"]),

    # Health_Inquiry + Department_Suggestion
    ("What does it mean when {lab} levels are {value}? Is this something I should see a "
     "{specialist} about or can my primary care doctor handle it?",
     ["Health_Inquiry", "Department_Suggestion"]),

    # Symptom_Triage + Health_Inquiry
    ("I've been having {symptom} that gets worse at night. Is this normal for someone with "
     "{condition}, or should I be concerned? What could be causing it?",
     ["Symptom_Triage", "Health_Inquiry"]),

    # Risk_Assessment + Department_Suggestion
    ("I'm {age} with a family history of {condition}. My BMI is {value} and I smoke. "
     "What's my cardiovascular risk and should I see a cardiologist proactively?",
     ["Risk_Assessment", "Department_Suggestion"]),

    # Symptom_Triage + Risk_Assessment + Health_Inquiry
    ("I woke up with severe {symptom} and {symptom2}. I have {condition} and take {med}. "
     "What could this indicate, and how dangerous is it in someone on {med}?",
     ["Symptom_Triage", "Risk_Assessment", "Health_Inquiry"]),
]

DIAGNOSIS_TEMPLATES: List[Tuple[str, List[str]]] = [
    # Differential_Diagnosis + Symptom_Analysis
    ("I have {symptom}, {symptom2}, and {symptom3} for {days} days. "
     "What conditions might explain all of these symptoms together?",
     ["Differential_Diagnosis", "Symptom_Analysis"]),

    # Test_Interpretation + Differential_Diagnosis
    ("My bloodwork shows {lab}: {value}, {lab2}: {value2}. "
     "What do these results indicate, and what conditions should be ruled out?",
     ["Test_Interpretation", "Differential_Diagnosis"]),

    # Etiology_Detection + Symptom_Analysis
    ("Why am I experiencing {symptom} even though my {condition} has been well-controlled? "
     "Could {condition2} or medications like {med} be contributing to this?",
     ["Etiology_Detection", "Symptom_Analysis"]),

    # Test_Interpretation + Etiology_Detection
    ("My {lab} came back at {value} — much higher than last time. I haven't changed my diet. "
     "What might be causing this sudden change, and is it related to my {condition}?",
     ["Test_Interpretation", "Etiology_Detection"]),

    # Differential_Diagnosis + Test_Interpretation + Etiology_Detection
    ("I have intermittent {symptom}, joint stiffness, and {lab}: {value}. "
     "What diseases present this way, and what is likely driving the elevated {lab}?",
     ["Differential_Diagnosis", "Test_Interpretation", "Etiology_Detection"]),

    # Symptom_Analysis + Test_Interpretation
    ("My doctor found {lab}: {value} in my recent test. I've also noticed {symptom} lately. "
     "Are these related, and what do they suggest about my {organ} function?",
     ["Symptom_Analysis", "Test_Interpretation"]),

    # Differential_Diagnosis + Symptom_Analysis + Etiology_Detection
    ("I'm {age} and have had {symptom}, {symptom2}, fatigue, and {symptom3} for {days} weeks. "
     "What are the top diagnoses and why would someone my age develop these together?",
     ["Differential_Diagnosis", "Symptom_Analysis", "Etiology_Detection"]),

    # Test_Interpretation + Differential_Diagnosis
    ("My MRI showed {finding}. I also have {symptom}. "
     "What does this finding mean, and which conditions is my neurologist likely considering?",
     ["Test_Interpretation", "Differential_Diagnosis"]),
]

MEDICATION_TEMPLATES: List[Tuple[str, List[str]]] = [
    # Drug_Interaction + Dosage_Recommendation
    ("I take {med} {dose}mg daily. My doctor added {med2} {dose2}mg. "
     "Are there interactions I should know about, and is this dose combination safe?",
     ["Drug_Interaction", "Dosage_Recommendation"]),

    # Contraindication_Check + Drug_Counseling
    ("I'm allergic to {allergen} and have {condition}. My doctor wants to prescribe {med}. "
     "Is this safe, and what side effects should I monitor?",
     ["Contraindication_Check", "Drug_Counseling"]),

    # Prescription_Review + Drug_Interaction
    ("I'm currently on {med}, {med2}, and {med3}. Can you review these for potential "
     "interactions, especially regarding {concern}?",
     ["Prescription_Review", "Drug_Interaction"]),

    # Dosage_Recommendation + Contraindication_Check
    ("What is the correct dose of {med} for a {age}-year-old with {condition}? "
     "Are there any kidney or liver concerns I should know about?",
     ["Dosage_Recommendation", "Contraindication_Check"]),

    # Drug_Interaction + Contraindication_Check + Dosage_Recommendation
    ("I take {med} for {condition} and my dentist prescribed {med2} for an infection. "
     "I'm allergic to {allergen}. Is {med2} safe and what dose is appropriate for me?",
     ["Drug_Interaction", "Contraindication_Check", "Dosage_Recommendation"]),

    # Drug_Counseling + Prescription_Review
    ("I just started {med}. What are the most important side effects to watch for, "
     "and does it interact with my current {med2} and {med3}?",
     ["Drug_Counseling", "Prescription_Review"]),

    # Dosage_Recommendation + Drug_Interaction
    ("My {age}-year-old child has {condition}. What dose of {med} is appropriate, "
     "and can it be given with {med2} that they already take?",
     ["Dosage_Recommendation", "Drug_Interaction"]),

    # Contraindication_Check + Drug_Counseling + Prescription_Review
    ("I have {condition}, {condition2}, and take {med} and {med2}. My doctor wants to add "
     "{med3}. Is there anything in my profile that makes this risky?",
     ["Contraindication_Check", "Drug_Counseling", "Prescription_Review"]),
]

POST_DIAGNOSIS_TEMPLATES: List[Tuple[str, List[str]]] = [
    # Rehabilitation_Advice + Follow_up_Scheduling
    ("I had {procedure} {weeks} weeks ago. When can I return to {activity}, "
     "and when should my next follow-up appointment be?",
     ["Rehabilitation_Advice", "Follow_up_Scheduling"]),

    # Lifestyle_Guidance + Progress_Tracking
    ("I was diagnosed with {condition} {months} months ago. What dietary and activity "
     "changes should I make, and how should I track my progress?",
     ["Lifestyle_Guidance", "Progress_Tracking"]),

    # Care_Support + Lifestyle_Guidance
    ("I was just told I have {condition}. I feel {emotion}. "
     "How do I cope emotionally, and what lifestyle adjustments should I prioritize first?",
     ["Care_Support", "Lifestyle_Guidance"]),

    # Rehabilitation_Advice + Lifestyle_Guidance + Follow_up_Scheduling
    ("After my {procedure}, when can I drive, work out, and eat normally again? "
     "Also, when should I schedule the next checkup with my {specialist}?",
     ["Rehabilitation_Advice", "Lifestyle_Guidance", "Follow_up_Scheduling"]),

    # Progress_Tracking + Follow_up_Scheduling
    ("I completed {treatment} {months} months ago. What symptoms should prompt me to "
     "call my doctor immediately, and how often should I get {lab} tests done?",
     ["Progress_Tracking", "Follow_up_Scheduling"]),

    # Care_Support + Progress_Tracking
    ("Living with {condition} is affecting my mental health. What support resources exist, "
     "and how do I know if my condition is getting better or worse?",
     ["Care_Support", "Progress_Tracking"]),

    # Rehabilitation_Advice + Care_Support
    ("My {relative} just had a {procedure} and is struggling emotionally with recovery. "
     "What exercises can they do, and how can I support them through this?",
     ["Rehabilitation_Advice", "Care_Support"]),

    # Lifestyle_Guidance + Progress_Tracking + Follow_up_Scheduling
    ("I'm managing {condition} with {med}. What lifestyle changes complement my medication, "
     "what metrics should I track at home, and when should I next see my doctor?",
     ["Lifestyle_Guidance", "Progress_Tracking", "Follow_up_Scheduling"]),
]

TEMPLATES_BY_CATEGORY: Dict[str, List[Tuple[str, List[str]]]] = {
    "Pre-Diagnosis": PRE_DIAGNOSIS_TEMPLATES,
    "Diagnosis": DIAGNOSIS_TEMPLATES,
    "Medication": MEDICATION_TEMPLATES,
    "Post-Diagnosis": POST_DIAGNOSIS_TEMPLATES,
}

# ─── Fill-in values for template placeholders ─────────────────────────────────
FILL_VALUES: Dict[str, List[str]] = {
    "age": ["28", "35", "42", "55", "63", "70", "22", "48", "51", "67"],
    "days": ["2", "3", "5", "7", "10", "14"],
    "weeks": ["2", "4", "6", "8"],
    "months": ["1", "2", "3", "6"],
    "relative": ["father", "mother", "brother", "sister", "grandfather", "grandmother"],
    "specialist": ["cardiologist", "neurologist", "gastroenterologist", "pulmonologist",
                   "endocrinologist", "rheumatologist", "nephrologist", "dermatologist"],
    "condition": [
        "type 2 diabetes", "hypertension", "asthma", "coronary artery disease",
        "hypothyroidism", "GERD", "chronic kidney disease", "heart failure",
        "atrial fibrillation", "osteoporosis", "rheumatoid arthritis", "COPD",
        "liver cirrhosis", "epilepsy", "Parkinson's disease",
    ],
    "condition2": [
        "hyperlipidemia", "obesity", "chronic anemia", "sleep apnea",
        "peripheral neuropathy", "metabolic syndrome", "vitamin D deficiency",
    ],
    "condition3": [
        "myocardial infarction", "stroke", "renal failure", "diabetic nephropathy",
        "heart failure", "pulmonary embolism",
    ],
    "symptom": [
        "chest pain", "shortness of breath", "severe headaches", "palpitations",
        "abdominal pain", "joint swelling", "extreme fatigue", "dizziness",
        "swollen ankles", "blurred vision", "nausea", "lower back pain",
    ],
    "symptom2": [
        "night sweats", "weight loss", "dry cough", "skin rash",
        "sensitivity to light", "tingling in hands", "morning stiffness",
    ],
    "symptom3": [
        "fever", "muscle weakness", "bruising easily", "hair loss",
        "irregular heartbeat", "loss of appetite",
    ],
    "lab": [
        "HbA1c", "serum creatinine", "TSH", "free T4", "LDL cholesterol",
        "CRP", "white blood cell count", "hemoglobin", "platelet count",
        "ALT", "AST", "troponin", "BNP", "INR", "potassium",
    ],
    "lab2": [
        "eGFR", "fasting glucose", "uric acid", "sodium", "albumin",
        "bilirubin", "PSA", "ferritin",
    ],
    "value": [
        "8.2%", "145 mg/dL", "9.1 mIU/L", "0.5 ng/dL", "185 mg/dL",
        "42 mg/L", "13,500 cells/μL", "9.2 g/dL", "72,000 /μL",
        "78 U/L", "3.2 mg/dL", "0.8 ng/mL", "1250 pg/mL", "3.4",
    ],
    "value2": [
        "28 mL/min", "195 mg/dL", "8.9 mg/dL", "128 mEq/L", "2.8 g/dL",
    ],
    "med": [
        "metformin", "lisinopril", "atorvastatin", "metoprolol", "warfarin",
        "levothyroxine", "omeprazole", "amlodipine", "losartan", "furosemide",
        "sertraline", "fluoxetine", "gabapentin", "prednisone", "allopurinol",
    ],
    "med2": [
        "aspirin", "ibuprofen", "amoxicillin", "ciprofloxacin", "fluconazole",
        "clopidogrel", "digoxin", "spironolactone", "carvedilol", "venlafaxine",
    ],
    "med3": [
        "colchicine", "hydroxychloroquine", "methotrexate", "adalimumab",
        "pantoprazole", "sitagliptin",
    ],
    "dose": ["10", "25", "50", "100", "500", "1000"],
    "dose2": ["5", "10", "20", "40", "200"],
    "allergen": [
        "penicillin", "sulfonamides", "NSAIDs", "codeine", "latex",
        "contrast dye", "cephalosporins",
    ],
    "concern": [
        "bleeding risk", "kidney function", "liver toxicity",
        "QT prolongation", "serotonin syndrome", "drug-induced hypertension",
    ],
    "procedure": [
        "bypass surgery", "knee replacement", "hip replacement",
        "appendectomy", "cholecystectomy", "coronary angioplasty",
        "spinal fusion", "mastectomy", "chemotherapy", "radiation therapy",
    ],
    "activity": [
        "driving", "going back to work", "exercising", "swimming",
        "playing sports", "traveling by air",
    ],
    "treatment": [
        "chemotherapy", "radiation therapy", "immunotherapy",
        "dialysis treatment", "antibiotic course",
    ],
    "emotion": [
        "overwhelmed", "scared", "depressed", "anxious", "hopeless",
        "confused about next steps",
    ],
    "finding": [
        "a 3mm white matter lesion", "disc herniation at L4-L5",
        "mild cortical atrophy", "a small meningioma",
        "enhancing lesion in the temporal lobe",
    ],
    "organ": [
        "kidney", "liver", "thyroid", "heart", "pancreas",
    ],
}


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a query template with random values from FILL_VALUES."""
    result = template
    # Iterate until all placeholders are filled
    max_iter = 20
    for _ in range(max_iter):
        import re
        placeholders = re.findall(r"\{(\w+)\}", result)
        if not placeholders:
            break
        for ph in placeholders:
            if ph in FILL_VALUES:
                result = result.replace("{" + ph + "}", rng.choice(FILL_VALUES[ph]), 1)
    return result


def _template_reference(query: str, intents: List[str], category: str) -> str:
    """Generate a plausible reference answer template (used when no API key)."""
    intent_phrases = {
        "Symptom_Triage": "Your described symptoms warrant medical evaluation. The pattern of symptoms suggests a potentially serious condition that should not be ignored.",
        "Risk_Assessment": "Based on your age, family history, and current conditions, your risk profile is moderate to high and warrants preventive monitoring.",
        "Department_Suggestion": "I recommend consulting a specialist in this area. Given your symptoms, an urgent evaluation within 24–48 hours is advisable.",
        "Health_Inquiry": "The value you mentioned falls outside the normal reference range and warrants medical attention and lifestyle adjustment.",
        "Differential_Diagnosis": "The combination of symptoms and findings is consistent with several conditions including inflammatory, metabolic, and infectious etiologies.",
        "Symptom_Analysis": "These symptoms, when considered together, suggest systemic involvement that requires comprehensive evaluation.",
        "Test_Interpretation": "The lab findings indicate dysfunction that requires correlation with clinical presentation and additional workup.",
        "Etiology_Detection": "The underlying cause likely involves a combination of metabolic, inflammatory, or medication-related factors that can be further investigated.",
        "Drug_Interaction": "This drug combination carries a clinically significant interaction risk and requires dose adjustment or alternative selection.",
        "Dosage_Recommendation": "The appropriate dose for your profile should account for renal/hepatic function, weight, and current medications.",
        "Contraindication_Check": "Based on your allergy history and comorbidities, a safer alternative would be preferable with careful monitoring if no alternatives exist.",
        "Drug_Counseling": "Key side effects to monitor include gastrointestinal upset, CNS changes, and potential organ toxicity with regular labs recommended.",
        "Prescription_Review": "Your current medication regimen has potential interactions and duplications that should be reviewed with your pharmacist.",
        "Rehabilitation_Advice": "A graduated return to activity is recommended, starting with low-impact exercises and progressively increasing intensity over 6–8 weeks.",
        "Lifestyle_Guidance": "Dietary modifications, regular aerobic exercise, stress management, and adequate sleep are foundational to managing your condition.",
        "Progress_Tracking": "Monitor key biomarkers monthly initially, then quarterly once stable. Report worsening symptoms, new onset issues, or significant lab changes promptly.",
        "Care_Support": "Emotional support through patient communities, therapy, and family education are important components of holistic management.",
        "Follow_up_Scheduling": "Schedule follow-up imaging or labs in 3 months, with a clinic visit in 4–6 weeks to assess treatment response and adjust the plan.",
    }
    parts = ["Based on the information provided, here is a comprehensive assessment:\n\n"]
    for intent in intents:
        phrase = intent_phrases.get(intent, "Medical assessment for this aspect of your query.")
        parts.append(f"**{intent.replace('_', ' ')}**: {phrase}\n\n")
    parts.append(
        "Please consult your healthcare provider before making any changes to your treatment. "
        "This information is for educational purposes only."
    )
    return "".join(parts)


def _gpt4o_reference(query: str, category: str) -> str:
    """Generate a reference answer using GPT-4o API."""
    try:
        import openai
        client = openai.OpenAI()
        prompt = (
            f"You are an expert medical AI assistant. Answer the following {category} "
            f"medical query comprehensively and accurately. Provide specific, actionable "
            f"information while noting any important safety considerations.\n\n"
            f"Query: {query}\n\n"
            f"Provide a detailed, medically accurate response (200–400 words)."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.debug(f"GPT-4o reference generation failed: {e}")
        return ""


def generate_composite_benchmark(
    n_per_category: int = 500,
    seed: int = 42,
    use_gpt_reference: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Generate composite-intent benchmark matching MedAide paper Section 4.1.

    Args:
        n_per_category: Target instances per category (paper uses 500).
        seed: Random seed for reproducibility.
        use_gpt_reference: Whether to call GPT-4o for reference answers.

    Returns:
        Dict mapping category -> list of benchmark instances.
    """
    rng = random.Random(seed)
    benchmark: Dict[str, List[Dict]] = {}

    for category, templates in TEMPLATES_BY_CATEGORY.items():
        instances = []
        prefix = "".join(c for c in category if c.isalpha())[:4].lower()
        idx = 0

        while len(instances) < n_per_category:
            # Cycle through templates and fill placeholders
            template_str, intents = templates[idx % len(templates)]
            query = _fill_template(template_str, rng)

            # Generate reference answer
            if use_gpt_reference and os.environ.get("OPENAI_API_KEY"):
                ref_answer = _gpt4o_reference(query, category)
                if not ref_answer:
                    ref_answer = _template_reference(query, intents, category)
            else:
                ref_answer = _template_reference(query, intents, category)

            instance = {
                "id": f"{prefix}_{len(instances) + 1:04d}",
                "query": query,
                "reference_answer": ref_answer,
                "expected_intents": intents,
                "n_intents": len(intents),
                "domain": category,
                "source": "composite_synthetic",
            }
            instances.append(instance)
            idx += 1

        benchmark[category] = instances
        logger.info(
            f"  {category}: {len(instances)} composite-intent instances "
            f"({len(templates)} templates × variations)"
        )

    return benchmark


def save_benchmark(benchmark: Dict[str, List[Dict]], output_dir: Path) -> Path:
    """Save benchmark to JSON and print stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "medaide_benchmark.json"

    # Intent distribution stats
    intent_counts: Dict[str, int] = {}
    multi_intent_count = 0
    for items in benchmark.values():
        for item in items:
            intents = item.get("expected_intents", [])
            if len(intents) > 1:
                multi_intent_count += 1
            for i in intents:
                intent_counts[i] = intent_counts.get(i, 0) + 1

    payload = {
        "description": (
            "MedAide+ Phase 4 Benchmark -- Composite-Intent Instances.\n"
            "Synthetic proxy generated to follow MedAide paper "
            "(arXiv:2410.12532v3) Section 4.1 protocol:\n"
            "4 categories × 500 instances with 2–3 combined medical intents per query.\n"
            "Matches 17-intent taxonomy from original paper (MedAide+ adds 18th: Follow_up_Scheduling)."
        ),
        "protocol": {
            "source_paper": "arXiv:2410.12532v3",
            "dataset_provenance": "synthetic_proxy_not_original_private_benchmark",
            "n_categories": 4,
            "n_per_category": next(iter([len(v) for v in benchmark.values()]), 0),
            "total_instances": sum(len(v) for v in benchmark.values()),
            "intent_type": "composite (2-3 intents per query)",
            "comparison_conditions": [
                "Vanilla GPT-4o (no framework)",
                "GPT-4o + Simulated MedAide (M1+M2+M3 only)",
                "GPT-4o + MedAide+ Full (M1-M7)",
            ],
        },
        "stats": {
            "total": sum(len(v) for v in benchmark.values()),
            "per_category": {cat: len(items) for cat, items in benchmark.items()},
            "multi_intent_instances": multi_intent_count,
            "intent_distribution": dict(sorted(intent_counts.items(), key=lambda x: -x[1])),
        },
        "categories": benchmark,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Benchmark saved -> {out_path}")
    logger.info(f"  Total: {payload['stats']['total']} instances")
    logger.info(f"  Multi-intent: {multi_intent_count} ({multi_intent_count*100//payload['stats']['total']}%)")

    return out_path


def load_benchmark(benchmark_path: Path) -> Dict[str, List[Dict]]:
    """Load benchmark from saved JSON file."""
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("categories", data)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MedAide+ composite-intent benchmark (matches arXiv:2410.12532v3 protocol)."
        )
    )
    parser.add_argument("--samples", type=int, default=500,
                        help="Instances per category (paper: 500). Use 20 for quick test.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-gpt-ref", action="store_true",
                        help="Use GPT-4o to generate reference answers (requires OPENAI_API_KEY).")
    args = parser.parse_args()

    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path(__file__).parent.parent / "data" / "benchmark")

    logger.info(f"Generating composite-intent benchmark ({args.samples} per category)...")
    benchmark = generate_composite_benchmark(
        n_per_category=args.samples,
        seed=args.seed,
        use_gpt_reference=args.with_gpt_ref,
    )

    out_path = save_benchmark(benchmark, output_dir)
    total = sum(len(v) for v in benchmark.values())
    print(f"\nBenchmark ready: {out_path}")
    print(f"   Total          : {total} composite-intent instances")
    for cat, items in benchmark.items():
        avg_intents = sum(len(i["expected_intents"]) for i in items) / len(items)
        print(f"   {cat:<20} {len(items)} instances  (avg {avg_intents:.1f} intents/query)")
    print(f"\n   Comparison protocol:")
    print(f"   (A) Vanilla GPT-4o                     (no framework)")
    print(f"   (B) GPT-4o + Simulated MedAide         (M1+M2+M3 only)")
    print(f"   (C) GPT-4o + Full MedAide+             (M1-M7, all 7 modules)")


if __name__ == "__main__":
    main()
