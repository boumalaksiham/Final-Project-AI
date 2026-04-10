"""
Qualitative Evaluation Script
Runs the pipeline on unseen papers and scores the generated reports.

The evaluation is designed around the real use case of this system:
a researcher needs to decide whether to read a paper BEFORE reading it.
The system generates a report, and the evaluator judges whether that
report gives enough information to make that decision.

Scoring dimensions:
  - Accuracy:     Is the extracted information factually correct?
  - Completeness: Does the report cover all important aspects?
  - Triage Value: Based on this report alone, could you decide whether to read this paper?

Usage:
    # Step 1 — run pipeline on 5 new papers:
    python qualitative_eval.py --arxiv 2106.09685 2203.02155 2302.13971 2210.11610 2005.11401

    # Step 2 — score each report interactively:
    python qualitative_eval.py --score

    # Print saved scores:
    python qualitative_eval.py --summary
"""

import argparse
import json
from pathlib import Path
from main import run_pipeline
from paper_loader import load_arxiv

OUTPUT_DIR = "outputs"
SCORES_FILE = "outputs/qualitative_eval.json"

EVAL_PAPERS = {
    "2106.09685": "LoRA: Low-Rank Adaptation of Large Language Models",
    "2203.02155": "Chain-of-Thought Prompting Elicits Reasoning in LLMs",
    "2302.13971": "LLaMA: Open and Efficient Foundation Language Models",
    "2210.11610": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
}


def run_on_papers(arxiv_ids: list):
    """Run the pipeline on each paper and save reports."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    results = {}

    for arxiv_id in arxiv_ids:
        title = EVAL_PAPERS.get(arxiv_id, arxiv_id)
        print(f"\n{'='*60}")
        print(f"  Processing: {title}")
        print(f"{'='*60}")

        try:
            text, _ = load_arxiv(arxiv_id)
            run_pipeline(text, source=f"arxiv:{arxiv_id}", user_topic="")

            report_path = Path(OUTPUT_DIR) / "analysis_report.md"
            raw_path    = Path(OUTPUT_DIR) / "raw_outputs.json"

            report_text = report_path.read_text() if report_path.exists() else ""
            raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}

            paper_report_path = Path(OUTPUT_DIR) / f"qual_report_{arxiv_id.replace('.','_')}.md"
            paper_report_path.write_text(report_text)

            results[arxiv_id] = {
                "title": title,
                "summary":      raw.get("summary",      {}).get("summary", ""),
                "tldr":         raw.get("relevance",     {}).get("tldr", ""),
                "num_refs":     raw.get("citations",     {}).get("num_references", 0),
                "datasets":     raw.get("methodology",   {}).get("datasets_identified", []),
                "metrics":      raw.get("methodology",   {}).get("evaluation_metrics", []),
                "models":       raw.get("methodology",   {}).get("model_architectures", []),
                "limitations":  raw.get("critical",      {}).get("limitations", []),
                "report_path":  str(paper_report_path),
                "scores": {"accuracy": None, "completeness": None, "triage_value": None},
            }
            print(f"  Report saved to {paper_report_path}")

        except Exception as e:
            print(f"  ERROR on {arxiv_id}: {e}")
            results[arxiv_id] = {"title": title, "error": str(e),
                                  "scores": {"accuracy": None, "completeness": None,
                                             "triage_value": None}}

    with open(SCORES_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll reports saved. Now run: python qualitative_eval.py --score")


def score_interactively():
    if not Path(SCORES_FILE).exists():
        print("No results file found. Run with --arxiv first.")
        return

    with open(SCORES_FILE) as f:
        results = json.load(f)

    print("\n" + "="*65)
    print("  Qualitative Evaluation — Manual Scoring")
    print("="*65)
    print("""
Goal of this system: Help a researcher decide whether to read a paper
BEFORE spending time on it. The system reads the paper and gives them
a structured report.

This evaluation has TWO different perspectives:

  [DEVELOPER perspective — Questions 1 and 2]
  Accuracy and Completeness are scored by the system developer (the author).
  They measure whether the agents are working correctly — did the system
  find the right datasets, metrics, models, and limitations?
  These require domain knowledge of the papers being evaluated.

  [USER perspective — Question 3]
  Triage Value is the most important question and can be scored by anyone.
  It measures whether a researcher who has NEVER read this paper
  would get enough information from the report to decide whether to read it.
  No domain knowledge needed — just read the report and ask yourself:
  do I now know what this paper is about?
""")

    for arxiv_id, data in results.items():
        if "error" in data and not data.get("summary"):
            print(f"\n  Skipping {data.get('title', arxiv_id)} — pipeline error.")
            continue

        print(f"\n{'='*65}")
        print(f"  Paper: {data.get('title', arxiv_id)}")
        print(f"  arXiv: {arxiv_id}")
        print(f"\n  ── What the system produced ──────────────────────────")
        print(f"  TL;DR:     {data.get('tldr', 'Not generated')[:120]}")
        print(f"  Summary:   {data.get('summary', 'Empty')[:150]}")
        print(f"  Datasets:  {data.get('datasets', [])}")
        print(f"  Metrics:   {data.get('metrics', [])}")
        print(f"  Models:    {data.get('models', [])}")
        print(f"  Limits:    {len(data.get('limitations', []))} found")
        print()

        GUIDES = {
            "accuracy": """
  ── QUESTION 1: ACCURACY [DEVELOPER EVALUATION] ───────────────
  Scored by the system developer who knows the paper content.
  Did the system get the FACTS right about this paper?
  Look at the datasets, metrics, and models listed above.
  Compare them to what you know this paper actually uses.

    1 = Everything is wrong, or the output is completely empty
    2 = Mostly wrong — datasets/models listed don't match this paper
    3 = About half correct — some things right, some clearly wrong
    4 = Mostly correct — maybe 1 small error but generally right
    5 = Everything listed is factually correct for this paper""",

            "completeness": """
  ── QUESTION 2: COMPLETENESS [DEVELOPER EVALUATION] ──────────
  Scored by the system developer who knows the paper content.
  Did the system find EVERYTHING important about this paper?
  A good report should have: a summary, the datasets used,
  the metrics, the model names, and at least one limitation.

    1 = Almost everything missing (empty summary, nothing found)
    2 = Only 1-2 things found, most sections are empty
    3 = Summary present but several gaps (missing key dataset etc.)
    4 = Most things covered, maybe 1 small gap
    5 = Everything present: summary, datasets, metrics, models, limits""",

            "triage_value": """
  ── QUESTION 3: TRIAGE VALUE [USER EVALUATION] ────────────────
  This is the end-user question — no domain knowledge needed.
  Imagine you have NEVER read this paper and you receive this report.
  Based ONLY on what the system produced above —
  would you have enough information to decide whether to read it?

    1 = No — I still have no idea what this paper is about
    2 = Barely — I know the general topic but not enough to decide
    3 = Somewhat — I could make a rough guess but I'm not confident
    4 = Mostly yes — I have enough to make a reasonable decision
    5 = Yes, completely — I know what this paper is about and
        can decide whether it's relevant to my work""",
        }

        for dim in ["accuracy", "completeness", "triage_value"]:
            print(GUIDES[dim])
            while True:
                try:
                    score = int(input(f"\n  Your score (1-5): "))
                    if 1 <= score <= 5:
                        results[arxiv_id]["scores"][dim] = score
                        break
                    else:
                        print("  Please enter a number between 1 and 5.")
                except ValueError:
                    print("  Please enter a number between 1 and 5.")

    with open(SCORES_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)


def print_summary(results: dict):
    print("\n" + "="*72)
    print("  Qualitative Evaluation Results")
    print("="*72)
    print(f"{'Paper':<44} {'Acc':>5} {'Comp':>6} {'Triage':>7} {'Avg':>6}")
    print("-"*72)

    all_scores = []
    for arxiv_id, data in results.items():
        scores = data.get("scores", {})
        acc    = scores.get("accuracy")
        comp   = scores.get("completeness")
        triage = scores.get("triage_value")
        if None in (acc, comp, triage):
            continue
        avg = round((acc + comp + triage) / 3, 2)
        all_scores.append((acc, comp, triage, avg))
        title = data.get("title", arxiv_id)[:43]
        print(f"{title:<44} {acc:>5} {comp:>6} {triage:>7} {avg:>6}")

    if all_scores:
        print("-"*72)
        print(f"{'AVERAGE':<44} "
              f"{round(sum(s[0] for s in all_scores)/len(all_scores),2):>5} "
              f"{round(sum(s[1] for s in all_scores)/len(all_scores),2):>6} "
              f"{round(sum(s[2] for s in all_scores)/len(all_scores),2):>7} "
              f"{round(sum(s[3] for s in all_scores)/len(all_scores),2):>6}")

    with open(SCORES_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to {SCORES_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualitative evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arxiv",   nargs="+", help="arXiv IDs to run pipeline on")
    group.add_argument("--score",   action="store_true", help="Score saved reports")
    group.add_argument("--summary", action="store_true", help="Print saved scores")
    args = parser.parse_args()

    if args.arxiv:
        run_on_papers(args.arxiv)
    elif args.score:
        score_interactively()
    elif args.summary:
        with open(SCORES_FILE) as f:
            results = json.load(f)
        print_summary(results)