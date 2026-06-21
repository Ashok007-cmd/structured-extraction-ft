#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Structured JSON Extraction Fine-Tuning.

Task: Given unstructured text, extract entities, relationships, dates, and financials
into a consistent structured JSON schema. This is a task where prompting alone
often fails (inconsistent output schemas, hallucinated fields, missing entities).

Generates two datasets:
  1. SFT Dataset:  (text, structured_json) pairs for supervised fine-tuning
  2. DPO Dataset:  (text, chosen_json, rejected_json) triples for preference tuning
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

random.seed(42)

# ---------------------------------------------------------------------------
# Entity / Template Pools
# ---------------------------------------------------------------------------

PERSONS = [
    "Sarah Chen", "James Rodriguez", "Aisha Patel", "Marcus Johnson",
    "Elena Kowalski", "David Park", "Priya Sharma", "Thomas Mueller",
    "Yuki Tanaka", "Fatima Al-Rashid", "Robert O'Brien", "Mei-Lin Wu",
    "Carlos Santos", "Olga Petrov", "Hassan Ali", "Grace Kim",
    "Ahmed Hassan", "Isabella Conti", "Liam O'Connor", "Zara Nkosi",
    "Vikram Singh", "Sophia Andersson", "Hiroshi Yamamoto", "Maria Silva",
]

ORGANIZATIONS = [
    "NexGen Dynamics", "Quantum Labs", "Apex Innovations", "Crestview Partners",
    "Pinnacle Systems", "Meridian Health", "Vanguard AI", "Stellar Robotics",
    "Atlas Financial", "Cascade Energy", "Horizon BioTech", "Summit Security",
    "Titan Manufacturing", "Equinox Ventures", "Polaris Data", "Aurora Defense",
    "Zenith Education", "Legacy Media Group", "Phoenix Aerospace", "Sapphire Therapeutics",
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "London, UK", "Tokyo, Japan",
    "Berlin, Germany", "Singapore", "Dubai, UAE", "Toronto, Canada",
    "Sydney, Australia", "Bangalore, India", "Stockholm, Sweden", "Seoul, South Korea",
    "Austin, TX", "Boston, MA", "Zurich, Switzerland", "Tel Aviv, Israel",
]

# Financial amount patterns
AMOUNTS = [
    ("$5M", 5_000_000), ("$10M", 10_000_000), ("$25M", 25_000_000),
    ("$50M", 50_000_000), ("$100M", 100_000_000), ("$250M", 250_000_000),
    ("$500M", 500_000_000), ("$1B", 1_000_000_000), ("$2.5B", 2_500_000_000),
    ("$5B", 5_000_000_000), ("$10B", 10_000_000_000),
    ("€20M", 20_000_000), ("€75M", 75_000_000), ("€200M", 200_000_000),
    ("£15M", 15_000_000), ("£60M", 60_000_000),
    ("¥2B", 2_000_000_000), ("¥5B", 5_000_000_000),
]

DATES = [
    "January 15, 2024", "March 22, 2024", "April 8, 2024", "June 3, 2024",
    "July 19, 2024", "September 12, 2024", "November 5, 2024", "December 20, 2024",
    "February 28, 2025", "May 10, 2025", "August 14, 2025", "October 7, 2025",
]

NORMALIZED_DATES = [
    "2024-01-15", "2024-03-22", "2024-04-08", "2024-06-03",
    "2024-07-19", "2024-09-12", "2024-11-05", "2024-12-20",
    "2025-02-28", "2025-05-10", "2025-08-14", "2025-10-07",
]

PRODUCTS = [
    "QuantumSync Platform", "NeuroMesh AI", "CloudForge Suite", "OptimaFlow Engine",
    "BioSense Analyzer", "CyberShield Pro", "DataVault Enterprise", "AutoPilot 360",
    "EcoTrack System", "MediAssist AI", "SmartGrid Controller", "VisionCore SDK",
]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Each template produces a (unstructured_text, structured_json) pair.
# Templates cover: acquisitions, funding rounds, partnerships, executive hires,
# product launches, legal cases, quarterly results, and regulatory filings.

TEMPLATES = [
    # --- Template 0: Acquisition ---
    {
        "scenario": "acquisition",
        "text_template": (
            "{acquirer} announced today that it has completed the acquisition of {target} "
            "for {amount} in an all-cash transaction. The deal, which was first reported on "
            "{date_raw}, was led by {ceo1}, CEO of {acquirer}, and will see {target}'s CEO "
            "{ceo2} join the combined company's board. The acquisition was advised by {advisor}."
        ),
        "extract_fn": lambda v: {
            "event_type": "acquisition",
            "acquirer": v["acquirer"],
            "target": v["target"],
            "entities": [
                {"type": "organization", "name": v["acquirer"]},
                {"type": "organization", "name": v["target"]},
                {"type": "organization", "name": v["advisor"]},
                {"type": "person", "name": v["ceo1"]},
                {"type": "person", "name": v["ceo2"]},
            ],
            "financials": [{"type": "acquisition_value", "amount": v["amount_num"], "currency": v["currency"]}],
            "dates": [{"raw": v["date_raw"], "normalized": v["date_norm"], "context": "announcement"}],
            "relationships": [
                {"type": "employment", "subject": v["ceo1"], "object": v["acquirer"], "role": "CEO"},
                {"type": "employment", "subject": v["ceo2"], "object": v["target"], "role": "CEO"},
                {"type": "acquired", "subject": v["acquirer"], "object": v["target"], "value": v["amount"], "date": v["date_raw"]},
                {"type": "advisor", "subject": v["advisor"], "object": "acquisition"},
            ],
        },
        "required_vars": ["acquirer", "target", "ceo1", "ceo2", "advisor",
                          "amount", "amount_num", "currency", "date_raw", "date_norm"],
    },

    # --- Template 1: Funding Round ---
    {
        "scenario": "funding",
        "text_template": (
            "{company} has raised {amount} in Series {series} funding round led by "
            "{lead_investor} with participation from {participant}. The round, announced "
            "on {date_raw}, brings the company's total funding to {total_amount}. "
            "{ceo}, founder and CEO of {company}, said the capital will be used to expand "
            "the {product} team and accelerate go-to-market efforts."
        ),
        "extract_fn": lambda v: {
            "event_type": "funding_round",
            "company": v["company"],
            "entities": [
                {"type": "organization", "name": v["company"]},
                {"type": "organization", "name": v["lead_investor"]},
                {"type": "organization", "name": v["participant"]},
                {"type": "person", "name": v["ceo"]},
                {"type": "product", "name": v["product"]},
            ],
            "financials": [
                {"type": "funding_raised", "amount": v["amount_num"], "currency": v["currency"], "series": v["series"]},
                {"type": "total_funding", "amount": v["total_amount_num"], "currency": v["currency"]},
            ],
            "dates": [{"raw": v["date_raw"], "normalized": v["date_norm"], "context": "announcement"}],
            "relationships": [
                {"type": "employment", "subject": v["ceo"], "object": v["company"], "role": "CEO"},
                {"type": "lead_investor", "subject": v["lead_investor"], "object": v["company"]},
                {"type": "investor", "subject": v["participant"], "object": v["company"]},
            ],
        },
        "required_vars": ["company", "lead_investor", "participant", "ceo", "product",
                          "amount", "amount_num", "currency", "series", "total_amount", "total_amount_num",
                          "date_raw", "date_norm"],
    },

    # --- Template 2: Executive Hire ---
    {
        "scenario": "executive_hire",
        "text_template": (
            "{company} today announced the appointment of {person} as its new {role}, "
            "effective {date_raw}. {person} joins from {previous_company} where they served as "
            "{previous_role}. {ceo_comment_lead}, CEO of {company}, commented: "
            '"We are thrilled to welcome {person} to lead our {department} initiatives." '
            "This appointment is part of {company}'s strategy to strengthen its market position."
        ),
        "extract_fn": lambda v: {
            "event_type": "executive_hire",
            "company": v["company"],
            "entities": [
                {"type": "organization", "name": v["company"]},
                {"type": "organization", "name": v["previous_company"]},
                {"type": "person", "name": v["person"]},
                {"type": "person", "name": v["ceo_comment_lead"]},
            ],
            "dates": [{"raw": v["date_raw"], "normalized": v["date_norm"], "context": "effective_date"}],
            "relationships": [
                {"type": "employment", "subject": v["person"], "object": v["company"], "role": v["role"]},
                {"type": "employment", "subject": v["person"], "object": v["previous_company"], "role": v["previous_role"]},
                {"type": "employment", "subject": v["ceo_comment_lead"], "object": v["company"], "role": "CEO"},
            ],
        },
        "required_vars": ["company", "person", "role", "previous_company", "previous_role",
                          "ceo_comment_lead", "department", "date_raw", "date_norm"],
    },

    # --- Template 3: Partnership ---
    {
        "scenario": "partnership",
        "text_template": (
            "{org1} and {org2} have entered into a strategic partnership to jointly develop "
            "the {product}. The multi-year agreement, valued at {amount}, was signed on "
            "{date_raw} at {org1}'s headquarters in {location}. {exec1}, CTO of {org1}, stated: "
            '"Combining {org2}\'s expertise with our platform will revolutionize the industry." '
            "The partnership is expected to create {jobs} new jobs over the next two years."
        ),
        "extract_fn": lambda v: {
            "event_type": "partnership",
            "partners": [v["org1"], v["org2"]],
            "entities": [
                {"type": "organization", "name": v["org1"]},
                {"type": "organization", "name": v["org2"]},
                {"type": "person", "name": v["exec1"]},
                {"type": "product", "name": v["product"]},
                {"type": "location", "name": v["location"]},
            ],
            "financials": [{"type": "partnership_value", "amount": v["amount_num"], "currency": v["currency"]}],
            "dates": [{"raw": v["date_raw"], "normalized": v["date_norm"], "context": "signing_date"}],
            "relationships": [
                {"type": "partnership", "subject": v["org1"], "object": v["org2"], "focus": v["product"]},
                {"type": "employment", "subject": v["exec1"], "object": v["org1"], "role": "CTO"},
            ],
            "metrics": [{"name": "jobs_created", "value": v["jobs"]}],
        },
        "required_vars": ["org1", "org2", "exec1", "product", "location",
                          "amount", "amount_num", "currency", "jobs", "date_raw", "date_norm"],
    },

    # --- Template 4: Product Launch ---
    {
        "scenario": "product_launch",
        "text_template": (
            "{company} officially launched {product} on {date_raw}, marking the company's "
            "entry into the {industry} market. The product, priced at {amount} per license, "
            "was unveiled by {ceo}, CEO of {company}, at the {event} in {location}. "
            "Early adopters include {customer1} and {customer2}. {company} expects the "
            "product to generate {revenue} in its first year."
        ),
        "extract_fn": lambda v: {
            "event_type": "product_launch",
            "company": v["company"],
            "entities": [
                {"type": "organization", "name": v["company"]},
                {"type": "organization", "name": v["customer1"]},
                {"type": "organization", "name": v["customer2"]},
                {"type": "person", "name": v["ceo"]},
                {"type": "product", "name": v["product"]},
                {"type": "location", "name": v["location"]},
            ],
            "financials": [
                {"type": "unit_price", "amount": v["amount_num"], "currency": v["currency"]},
                {"type": "projected_revenue", "amount": v["revenue_num"], "currency": v["currency"]},
            ],
            "dates": [{"raw": v["date_raw"], "normalized": v["date_norm"], "context": "launch_date"}],
            "relationships": [
                {"type": "employment", "subject": v["ceo"], "object": v["company"], "role": "CEO"},
                {"type": "customer", "subject": v["customer1"], "object": v["company"]},
                {"type": "customer", "subject": v["customer2"], "object": v["company"]},
            ],
            "metrics": [{"name": "first_year_projection", "value": v["revenue"]}],
        },
        "required_vars": ["company", "ceo", "product", "industry", "event", "location",
                          "customer1", "customer2", "amount", "amount_num", "currency",
                          "revenue", "revenue_num", "date_raw", "date_norm"],
    },

    # --- Template 5: Quarterly Results ---
    {
        "scenario": "quarterly_results",
        "text_template": (
            "{company} reported its Q{quarter} {year} earnings on {date_raw}, posting revenue "
            "of {revenue} against expectations of {expected}. The company cited strong "
            "demand for {product} as a key driver. CFO {cfo} commented: "
            '"Our {segment} segment grew {growth} year-over-year." '
            "CEO {ceo} reaffirmed the company's full-year guidance of {guidance}."
        ),
        "extract_fn": lambda v: {
            "event_type": "earnings_report",
            "company": v["company"],
            "entities": [
                {"type": "organization", "name": v["company"]},
                {"type": "person", "name": v["ceo"]},
                {"type": "person", "name": v["cfo"]},
                {"type": "product", "name": v["product"]},
            ],
            "financials": [
                {"type": "revenue", "amount": v["revenue_num"], "currency": v["currency"], "period": f"Q{v['quarter']} {v['year']}"},
                {"type": "expected_revenue", "amount": v["expected_num"], "currency": v["currency"]},
                {"type": "guidance", "amount": v["guidance_num"], "currency": v["currency"]},
            ],
            "dates": [{"raw": v["date_raw"], "normalized": v["date_norm"], "context": "report_date"}],
            "relationships": [
                {"type": "employment", "subject": v["ceo"], "object": v["company"], "role": "CEO"},
                {"type": "employment", "subject": v["cfo"], "object": v["company"], "role": "CFO"},
            ],
            "metrics": [
                {"name": "segment_growth", "value": v["growth"]},
                {"name": "quarter", "value": f"Q{v['quarter']} {v['year']}"},
            ],
        },
        "required_vars": ["company", "ceo", "cfo", "product", "segment",
                          "revenue", "revenue_num", "expected", "expected_num",
                          "guidance", "guidance_num", "currency", "growth",
                          "quarter", "year", "date_raw", "date_norm"],
    },

    # --- Template 6: Legal/Regulatory ---
    {
        "scenario": "legal",
        "text_template": (
            "The {regulatory_body} has approved {company}'s {product} for {market} "
            "as of {date_raw}. The decision followed a {duration}-month review process led by "
            "{official}. {company} CEO {ceo} called it 'a landmark moment for the industry.' "
            "Shares of {company} rose {percent} on the news. The approval covers "
            "{indication} and positions {company} to capture an estimated {market_size} market."
        ),
        "extract_fn": lambda v: {
            "event_type": "regulatory_approval",
            "company": v["company"],
            "entities": [
                {"type": "organization", "name": v["company"]},
                {"type": "organization", "name": v["regulatory_body"]},
                {"type": "person", "name": v["ceo"]},
                {"type": "person", "name": v["official"]},
                {"type": "product", "name": v["product"]},
            ],
            "dates": [{"raw": v["date_raw"], "normalized": v["date_norm"], "context": "approval_date"}],
            "financials": [{"type": "market_size", "amount": v["market_size_num"], "currency": v["currency"]}],
            "relationships": [
                {"type": "employment", "subject": v["ceo"], "object": v["company"], "role": "CEO"},
                {"type": "regulatory_approved", "subject": v["regulatory_body"], "object": v["product"]},
            ],
            "metrics": [
                {"name": "review_duration_months", "value": v["duration"]},
                {"name": "stock_change_percent", "value": v["percent"]},
            ],
        },
        "required_vars": ["company", "regulatory_body", "ceo", "official", "product",
                          "market", "indication", "duration", "percent",
                          "market_size", "market_size_num", "currency", "date_raw", "date_norm"],
    },
]


# ---------------------------------------------------------------------------
# Sampling Helpers
# ---------------------------------------------------------------------------

def pick(items, used=None):
    """Pick a random item, avoiding `used` if possible."""
    pool = [i for i in items if used is None or i not in used]
    if not pool:
        pool = items
    return random.choice(pool)


def sample_amount() -> Tuple[str, float, str]:
    """Returns (formatted_str, numeric_value, currency_symbol)."""
    fmt_str, val = random.choice(AMOUNTS)
    currency = fmt_str[0]  # $, €, £, ¥
    return fmt_str, val, currency


def sample_date_pair() -> Tuple[str, str]:
    """Returns (raw_date_string, normalized_iso_date)."""
    idx = random.randrange(len(DATES))
    return DATES[idx], NORMALIZED_DATES[idx]


# ---------------------------------------------------------------------------
# Dataset Generators
# ---------------------------------------------------------------------------

def generate_sft_example(template: dict, used_names: set) -> Optional[Dict]:
    """Generate one SFT example from a template. Returns None if template can't be filled."""
    try:
        v = {}
        required = template["required_vars"]

        # Fill variables based on type hints in var names
        var_pool = {
            "acquirer": lambda: pick(ORGANIZATIONS),
            "target": lambda: pick(ORGANIZATIONS, used=[v.get("acquirer")]),
            "company": lambda: pick(ORGANIZATIONS),
            "lead_investor": lambda: pick(ORGANIZATIONS, used=[v.get("company")]),
            "participant": lambda: pick(ORGANIZATIONS, used=[v.get("company"), v.get("lead_investor")]),
            "previous_company": lambda: pick(ORGANIZATIONS, used=[v.get("company")]),
            "advisor": lambda: pick(ORGANIZATIONS),
            "org1": lambda: pick(ORGANIZATIONS),
            "org2": lambda: pick(ORGANIZATIONS, used=[v.get("org1")]),
            "customer1": lambda: pick(ORGANIZATIONS, used=[v.get("company")]),
            "customer2": lambda: pick(ORGANIZATIONS, used=[v.get("company"), v.get("customer1")]),
            "regulatory_body": lambda: random.choice(["FDA", "EMA", "FCC", "SEC", "FTC", "MHRA"]),

            "ceo": lambda: pick(PERSONS),
            "ceo1": lambda: pick(PERSONS),
            "ceo2": lambda: pick(PERSONS, used=[v.get("ceo1")]),
            "ceo_comment_lead": lambda: pick(PERSONS),
            "cfo": lambda: pick(PERSONS, used=[v.get("ceo")]),
            "person": lambda: pick(PERSONS),
            "exec1": lambda: pick(PERSONS),
            "official": lambda: pick(PERSONS, used=[v.get("ceo")]),

            "product": lambda: pick(PRODUCTS),
            "location": lambda: pick(LOCATIONS),
            "role": lambda: random.choice(["Chief Technology Officer", "VP of Engineering",
                                            "Chief Product Officer", "Head of AI Research",
                                            "Chief Data Officer", "SVP of Sales"]),
            "previous_role": lambda: random.choice(["CTO", "VP Engineering", "Chief Scientist",
                                                      "Head of Product", "Managing Director"]),
            "department": lambda: random.choice(["engineering", "research", "product",
                                                   "sales", "marketing", "AI"]),
            "industry": lambda: random.choice(["AI", "biotech", "fintech", "clean energy",
                                                 "cybersecurity", "healthcare", "defense"]),
            "event": lambda: random.choice(["World Economic Forum", "CES", "TechCrunch Disrupt",
                                              "Slush", "Web Summit", "NeurIPS"]),
            "segment": lambda: random.choice(["enterprise", "consumer", "cloud", "healthcare", "defense"]),
            "market": lambda: random.choice(["US", "EU", "global", "Asia-Pacific", "North America"]),
            "indication": lambda: random.choice(["oncology", "cardiology", "neurological disorders",
                                                   "autoimmune diseases", "rare diseases"]),
            "growth": lambda: f"{random.randint(10, 150)}%",
            "series": lambda: random.choice(["A", "B", "C", "D", "E"]),
            "jobs": lambda: random.randint(50, 2000),
            "quarter": lambda: random.randint(1, 4),
            "year": lambda: random.choice(["2024", "2025"]),
            "duration": lambda: random.randint(3, 24),
            "percent": lambda: f"{random.randint(2, 25)}%",
        }

        # Fill financial variables
        amount_str, amount_val, currency_sym = sample_amount()
        v["amount"] = amount_str
        v["amount_num"] = amount_val
        v["currency"] = currency_sym

        # Handle second amount if needed
        if "revenue" in required or "total_amount" in required or "guidance" in required:
            amt2_str, amt2_val, _ = sample_amount()
            if "revenue" in required:
                v["revenue"] = amt2_str
                v["revenue_num"] = amt2_val
            if "total_amount" in required:
                v["total_amount"] = amt2_str
                v["total_amount_num"] = amt2_val
            if "guidance" in required:
                v["guidance"] = amt2_str
                v["guidance_num"] = amt2_val

        if "expected" in required or "market_size" in required:
            amt3_str, amt3_val, _ = sample_amount()
            if "expected" in required:
                v["expected"] = amt3_str
                v["expected_num"] = amt3_val
            if "market_size" in required:
                v["market_size"] = amt3_str
                v["market_size_num"] = amt3_val

        # Fill dates
        d_raw, d_norm = sample_date_pair()
        v["date_raw"] = d_raw
        v["date_norm"] = d_norm

        # Fill all other variables
        for var_name in required:
            if var_name not in v and var_name in var_pool:
                v[var_name] = var_pool[var_name]()

        # Fill any remaining missing vars with placeholders (shouldn't happen with well-formed templates)
        for var_name in required:
            if var_name not in v:
                v[var_name] = "Unknown"

        # Build text
        text = template["text_template"].format(**v)

        # Build structured JSON output
        output_data = template["extract_fn"](v)

        # Normalize output: sort keys, ensure clean serialization (minified JSON)
        output_json = json.dumps(output_data, ensure_ascii=False)

        # Track used names for diversity
        for ent in output_data.get("entities", []):
            used_names.add(ent["name"])

        return {
            "text": text,
            "structured_json": output_data,
            "json_string": output_json,
            "scenario": template["scenario"],
            "num_entities": len(output_data.get("entities", [])),
        }

    except Exception as e:
        print(f"  [WARN] Failed to generate example: {e}")
        return None


def generate_sft_dataset(num_examples: int = 5000,
                         output_dir: str = "./data/sft_dataset") -> Tuple[List[Dict], List[Dict]]:
    """
    Generate SFT dataset with train/eval split.

    Returns (train_data, eval_data) where each item has 'prompt' and 'completion' keys
    formatted for chat-template-based training.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    used_names: set = set()
    examples = []
    attempts = 0
    max_attempts = num_examples * 3

    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1
        template = random.choice(TEMPLATES)
        ex = generate_sft_example(template, used_names)
        if ex is None:
            continue

        # Format as chat messages for instruct model
        system_prompt = (
            "You are a structured data extraction assistant. "
            "Extract all entities, relationships, dates, financials, and metrics from the text "
            "and output them as a structured JSON object. Follow the JSON schema precisely. "
            "Do NOT include any text outside the JSON."
        )

        # The full chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract structured data from this text:\n\n{ex['text']}"},
            {"role": "assistant", "content": ex['json_string']},
        ]

        examples.append({
            "messages": messages,
            "text": ex['text'],
            "output_json": ex['structured_json'],
            "scenario": ex['scenario'],
        })

    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    eval_data = examples[split_idx:]

    # Save
    def save_split(data, filename):
        path = Path(output_dir) / filename
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps({"messages": item["messages"]}, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data)} examples to {path}")
        return path

    print(f"Generating {num_examples} SFT examples...")
    save_split(train, "train.jsonl")
    save_split(eval_data, "eval.jsonl")

    # Save a readable sample
    sample_path = Path(output_dir) / "sample.json"
    with open(sample_path, "w") as f:
        json.dump(examples[0], f, indent=2, ensure_ascii=False)
    print(f"  Sample saved to {sample_path}")

    # Print stats
    scenarios = {}
    for ex in examples:
        s = ex["scenario"]
        scenarios[s] = scenarios.get(s, 0) + 1
    print("\nDataset composition by scenario:")
    for scenario, count in sorted(scenarios.items()):
        print(f"  {scenario}: {count} ({count/len(examples)*100:.1f}%)")

    return train, eval_data


# ---------------------------------------------------------------------------
# DPO Dataset Generation
# ---------------------------------------------------------------------------

def generate_dpo_example(sft_example: Dict) -> Optional[Dict]:
    """
    Given an SFT example, produce a DPO triple (prompt, chosen, rejected).

    "Chosen" = the correct structured JSON.
    "Rejected" = a deliberately corrupted version (missing fields, wrong structure, etc.)
    """
    text = sft_example["text"]
    correct_json = sft_example["output_json"]
    correct_str = json.dumps(correct_json, ensure_ascii=False)

    system_prompt = (
        "You are a structured data extraction assistant. "
        "Extract all entities, relationships, dates, financials, and metrics from the text "
        "and output them as a structured JSON object. Follow the JSON schema precisely. "
        "Do NOT include any text outside the JSON."
    )

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract structured data from this text:\n\n{text}"},
    ]

    # Generate a "rejected" version with deliberate errors
    rejected = corrupt_json(correct_json)
    chosen_str = correct_str
    rejected_str = json.dumps(rejected, ensure_ascii=False)

    # Occasionally swap chosen/rejected to avoid bias (5% of the time)
    if random.random() < 0.05:
        chosen_str, rejected_str = rejected_str, correct_str

    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen_str}],
        "rejected": [{"role": "assistant", "content": rejected_str}],
    }


def corrupt_json(original: dict) -> dict:
    """
    Introduce structured errors into a JSON output. Types of corruption:
    - Drop entities (missing extraction)
    - Add hallucinated entities
    - Wrong field types (string instead of number)
    - Missing required fields
    - Nested structure errors
    - Incorrect normalization
    """
    corrupted = json.loads(json.dumps(original))  # deep copy

    corruption_type = random.choice([
        "drop_entity",
        "hallucinate_entity",
        "wrong_field_type",
        "missing_field",
        "flat_instead_of_nested",
        "bad_normalization",
        "multiple_errors",
    ])

    if corruption_type == "drop_entity":
        # Remove a random entity
        if "entities" in corrupted and len(corrupted["entities"]) > 1:
            idx = random.randrange(len(corrupted["entities"]))
            corrupted["entities"].pop(idx)

    elif corruption_type == "hallucinate_entity":
        # Add a fake entity
        if "entities" in corrupted:
            fake = random.choice([
                {"type": "person", "name": "John Doe"},
                {"type": "organization", "name": "FakeCorp Inc."},
                {"type": "product", "name": "FakeProduct 9000"},
                {"type": "location", "name": "Atlantis"},
            ])
            corrupted["entities"].append(fake)

    elif corruption_type == "wrong_field_type":
        # Change amount to string or remove numeric value
        if "financials" in corrupted and corrupted["financials"]:
            fi = random.choice(corrupted["financials"])
            if "amount" in fi and isinstance(fi["amount"], (int, float)):
                fi["amount"] = f"approximately {fi['amount']}"

    elif corruption_type == "missing_field":
        # Remove an entire section
        sections = ["entities", "relationships", "financials", "dates", "metrics"]
        present = [s for s in sections if s in corrupted and corrupted[s]]
        if present:
            corrupted.pop(random.choice(present))

    elif corruption_type == "flat_instead_of_nested":
        # Flatten nested structures
        if "entities" in corrupted:
            corrupted["extracted_entities"] = [e["name"] for e in corrupted["entities"]]
            del corrupted["entities"]

    elif corruption_type == "bad_normalization":
        # Wrong date format
        if "dates" in corrupted and corrupted["dates"]:
            d = random.choice(corrupted["dates"])
            if "normalized" in d:
                d["normalized"] = d["raw"]  # not normalized at all

    elif corruption_type == "multiple_errors":
        # 2-3 errors combined
        if "entities" in corrupted and len(corrupted["entities"]) > 1:
            corrupted["entities"].pop(0)
        if "financials" in corrupted and corrupted["financials"]:
            fi = corrupted["financials"][0]
            if "amount" in fi:
                fi.pop("amount")
        corrupted["_note"] = "This might have errors"  # extra unofficial field

    return corrupted


def generate_dpo_dataset(sft_data: List[Dict],
                         num_examples: int = 2000,
                         output_dir: str = "./data/dpo_dataset") -> Tuple[List[Dict], List[Dict]]:
    """
    Generate DPO dataset from SFT examples.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dpo_examples = []
    for sft_ex in sft_data:
        ex = generate_dpo_example(sft_ex)
        if ex is not None:
            dpo_examples.append(ex)
        if len(dpo_examples) >= num_examples:
            break

    random.shuffle(dpo_examples)
    split_idx = int(len(dpo_examples) * 0.9)
    train = dpo_examples[:split_idx]
    eval_data = dpo_examples[split_idx:]

    def save_dpo_split(data, filename):
        path = Path(output_dir) / filename
        with open(path, "w") as f:
            for item in data:
                record = {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data)} DPO examples to {path}")

    print(f"\nGenerating {num_examples} DPO examples...")
    save_dpo_split(train, "train.jsonl")
    save_dpo_split(eval_data, "eval.jsonl")

    return train, eval_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate fine-tuning datasets")
    parser.add_argument("--sft-size", type=int, default=5000,
                        help="Number of SFT examples (default: 5000)")
    parser.add_argument("--dpo-size", type=int, default=2000,
                        help="Number of DPO examples (default: 2000)")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory (default: ./data)")
    args = parser.parse_args()

    print("=" * 60)
    print("STRUCTURED JSON EXTRACTION DATASET GENERATOR")
    print("=" * 60)

    # Generate SFT dataset
    sft_train, sft_eval = generate_sft_dataset(
        num_examples=args.sft_size,
        output_dir=f"{args.output_dir}/sft_dataset",
    )

    # Generate DPO dataset from SFT examples
    dpo_train, dpo_eval = generate_dpo_dataset(
        sft_data=sft_train + sft_eval,
        num_examples=args.dpo_size,
        output_dir=f"{args.output_dir}/dpo_dataset",
    )

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print(f"  SFT: {len(sft_train)} train + {len(sft_eval)} eval = {len(sft_train) + len(sft_eval)} total")
    print(f"  DPO: {len(dpo_train)} train + {len(dpo_eval)} eval = {len(dpo_train) + len(dpo_eval)} total")
    print("=" * 60)
