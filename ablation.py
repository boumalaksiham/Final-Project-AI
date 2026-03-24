"""
Ablation Study
Runs the pipeline with each agent disabled one at a time.
Compares full system vs. each ablated version using evaluation metrics.
"""

import json
import copy
from pathlib import Path
from paper_loader import load_arxiv
from agents import (
    SummarizationAgent,
    CitationAnalysisAgent,
    MethodologyExtractorAgent,
    CriticalAnalysisAgent,
    CoordinatorAgent,
)
from evaluate import evaluate_summarization, evaluate_extraction

OUTPUT_DIR = "outputs/ablation"

# Ground truth per paper (same as evaluate.py)
GROUND_TRUTH = {
    "1706.03762": {
        "title": "Attention Is All You Need",
        "reference_summary": (
            "We propose the Transformer, a model based solely on attention mechanisms, "
            "achieving state-of-the-art BLEU scores on WMT 2014 English-German and "
            "English-French translation tasks, trained significantly faster than "
            "recurrent or convolutional architectures."
        ),
        "datasets": ["wmt", "wmt 2014"],
        "metrics": ["bleu", "ppl", "accuracy"],
        "models": ["transformer", "attention", "encoder", "decoder"],
    },
    "2005.14165": {
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "reference_summary": (
            "GPT-3 is an autoregressive language model with 175 billion parameters "
            "that achieves strong few-shot performance on NLP tasks including translation, "
            "question answering, and cloze tasks without any gradient updates or fine-tuning."
        ),
        "datasets": ["common crawl", "arxiv", "glue", "superglue"],
        "metrics": ["em", "ppl", "roc", "wer"],
        "models": ["gpt", "transformer"],
    },
}

BLANK_SUMMARY   = {"summary": "", "chunk_summaries": []}
BLANK_CITATION  = {"paper_title": "Unknown", "num_references": 0, "references": [],
                   "graph_nodes": 0, "graph_edges": 0, "graph_image": ""}
BLANK_METHOD    = {"datasets_identified": [], "evaluation_metrics": [],
                   "model_architectures": [], "ner_entities": {}, "methodology_excerpt": ""}
BLANK_CRITICAL  = {"limitations": [], "future_directions": [],
                   "num_limitations_found": 0, "num_future_directions_found": 0}


def run_full_pipeline(text):
    summarizer    = SummarizationAgent()
    citation      = CitationAnalysisAgent()
    methodology   = MethodologyExtractorAgent()
    critical      = CriticalAnalysisAgent()

    s = summarizer.run(text)
    c = citation.run(text, output_dir=OUTPUT_DIR)
    m = methodology.run(text)
    cr = critical.run(text)
    return s, c, m, cr


def score_outputs(s, c, m, cr, gt):
    rouge = evaluate_summarization(s.get("summary", ""), gt["reference_summary"])
    ds    = evaluate_extraction(m.get("datasets_identified", []),   gt["datasets"])
    ms    = evaluate_extraction(m.get("evaluation_metrics", []),    gt["metrics"])
    mo    = evaluate_extraction(m.get("model_architectures", []),   gt["models"])
    return {
        "rouge1":         rouge["rouge1"],
        "rouge2":         rouge["rouge2"],
        "rougeL":         rouge["rougeL"],
        "dataset_f1":     ds["f1"],
        "metric_f1":      ms["f1"],
        "model_f1":       mo["f1"],
        "num_references": c.get("num_references", 0),
        "num_limitations": cr.get("num_limitations_found", 0),
    }


def run_ablation(paper_id: str):
    print(f"\n{'='*60}")
    print(f"  Ablation Study — arXiv:{paper_id}")
    print(f"{'='*60}\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    gt = GROUND_TRUTH[paper_id]
    text, _ = load_arxiv(paper_id)

    print("Running full pipeline...")
    s, c, m, cr = run_full_pipeline(text)
    full_scores = score_outputs(s, c, m, cr, gt)
    print(f"  Full system scores: ROUGE-1={full_scores['rouge1']}, "
          f"Dataset-F1={full_scores['dataset_f1']}, "
          f"Model-F1={full_scores['model_f1']}, "
          f"Refs={full_scores['num_references']}\n")

    ablations = {
        "no_summarization":  (BLANK_SUMMARY,  c,            m,          cr),
        "no_citation":       (s,              BLANK_CITATION, m,         cr),
        "no_methodology":    (s,              c,            BLANK_METHOD, cr),
        "no_critical":       (s,              c,            m,          BLANK_CRITICAL),
    }

    results = {"full_system": full_scores}

    for name, (as_, ac, am, acr) in ablations.items():
        print(f"Running ablation: {name}...")
        scores = score_outputs(as_, ac, am, acr, gt)
        results[name] = scores

        # Show delta vs full system
        rouge_drop = round(full_scores["rouge1"] - scores["rouge1"], 4)
        ds_drop    = round(full_scores["dataset_f1"] - scores["dataset_f1"], 4)
        ref_drop   = full_scores["num_references"] - scores["num_references"]
        print(f"  ROUGE-1 drop: {rouge_drop:+.4f} | "
              f"Dataset-F1 drop: {ds_drop:+.4f} | "
              f"Refs drop: {ref_drop:+d}\n")

    # Save results
    out_path = f"outputs/ablation_{paper_id.replace('.', '_')}.json"
    with open(out_path, "w") as f:
        json.dump({"paper_id": paper_id, "ground_truth": gt["title"],
                   "ablation_results": results}, f, indent=2)

    print(f"\nAblation results saved to {out_path}")
    print_table(results, paper_id)
    return results


def print_table(results, paper_id):
    print(f"\n{'='*60}")
    print(f"  Summary Table — arXiv:{paper_id}")
    print(f"{'='*60}")
    print(f"{'Config':<22} {'ROUGE-1':>8} {'ROUGE-2':>8} {'DS-F1':>7} "
          f"{'MT-F1':>7} {'MO-F1':>7} {'Refs':>6} {'Lims':>5}")
    print("-" * 70)
    for name, s in results.items():
        print(f"{name:<22} {s['rouge1']:>8.4f} {s['rouge2']:>8.4f} "
              f"{s['dataset_f1']:>7.4f} {s['metric_f1']:>7.4f} "
              f"{s['model_f1']:>7.4f} {s['num_references']:>6} "
              f"{s['num_limitations']:>5}")
    print()


if __name__ == "__main__":
    import sys
    paper_id = sys.argv[1] if len(sys.argv) > 1 else "1706.03762"
    run_ablation(paper_id)