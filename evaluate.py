"""
Evaluation Script
Computes ROUGE scores for summarization and precision/recall for extraction agents.
Supports multiple papers with annotated ground truth.
"""

from rouge_score import rouge_scorer
import json
from pathlib import Path


def evaluate_summarization(generated: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L between generated and reference summaries."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def evaluate_extraction(predicted: list, ground_truth: list) -> dict:
    """Compute precision, recall, F1 for extraction tasks."""
    pred_set = set(p.lower() for p in predicted)
    truth_set = set(t.lower() for t in ground_truth)
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


# -----------------------------------------------------------------------
# Annotated ground truth test set
# Each entry corresponds to a paper we have run the pipeline on.
# Ground truth is manually verified from the actual paper content.
# -----------------------------------------------------------------------
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


def run_evaluation(raw_outputs_path: str = "outputs/raw_outputs.json",
                   paper_id: str = None):
    """
    Run evaluation on saved agent outputs against annotated ground truth.
    paper_id: arXiv ID to select the right ground truth (auto-detected if None).
    """
    print("\n=== Evaluation ===\n")

    with open(raw_outputs_path) as f:
        outputs = json.load(f)

    # Auto-detect paper from citation output title if paper_id not given
    if paper_id is None:
        detected_title = outputs.get("citations", {}).get("paper_title", "")
        for pid, gt in GROUND_TRUTH.items():
            if gt["title"].lower().split()[0] in detected_title.lower():
                paper_id = pid
                break

    gt = GROUND_TRUTH.get(paper_id)
    if gt is None:
        print(f"No ground truth found for paper_id={paper_id}. Using first available.")
        gt = list(GROUND_TRUTH.values())[0]

    print(f"Evaluating against: {gt['title']}\n")

    # --- Summarization ROUGE ---
    generated_summary = outputs["summary"].get("summary", "")
    rouge_scores = evaluate_summarization(generated_summary, gt["reference_summary"])
    print("Summarization ROUGE Scores:")
    for k, v in rouge_scores.items():
        print(f"  {k}: {v}")

    # --- Methodology Extraction ---
    pred_datasets = outputs["methodology"].get("datasets_identified", [])
    pred_metrics = outputs["methodology"].get("evaluation_metrics", [])
    pred_models = outputs["methodology"].get("model_architectures", [])

    print(f"\nDataset Extraction:")
    print(f"  Predicted:    {pred_datasets}")
    print(f"  Ground truth: {gt['datasets']}")
    dataset_scores = evaluate_extraction(pred_datasets, gt["datasets"])
    for k, v in dataset_scores.items():
        print(f"  {k}: {v}")

    print(f"\nMetric Extraction:")
    print(f"  Predicted:    {pred_metrics}")
    print(f"  Ground truth: {gt['metrics']}")
    metric_scores = evaluate_extraction(pred_metrics, gt["metrics"])
    for k, v in metric_scores.items():
        print(f"  {k}: {v}")

    print(f"\nModel Architecture Extraction:")
    print(f"  Predicted:    {pred_models}")
    print(f"  Ground truth: {gt['models']}")
    model_scores = evaluate_extraction(pred_models, gt["models"])
    for k, v in model_scores.items():
        print(f"  {k}: {v}")

    # --- Citation completeness ---
    num_refs = outputs.get("citations", {}).get("num_references", 0)
    print(f"\nCitation Graph:")
    print(f"  References extracted: {num_refs}")
    print(f"  Graph nodes: {outputs.get('citations', {}).get('graph_nodes', 0)}")
    print(f"  Graph edges: {outputs.get('citations', {}).get('graph_edges', 0)}")

    # Save results
    eval_results = {
        "paper": gt["title"],
        "summarization": rouge_scores,
        "dataset_extraction": dataset_scores,
        "metric_extraction": metric_scores,
        "model_extraction": model_scores,
        "citation_count": num_refs,
    }
    eval_path = "outputs/evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to {eval_path}")
    return eval_results


if __name__ == "__main__":
    import sys
    paper_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(paper_id=paper_id)