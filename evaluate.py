"""
Evaluation Script
Computes ROUGE scores for summarization and precision/recall for extraction agents.
Compares multi-agent system vs. single-model baseline.
"""

from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
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


def evaluate_extraction(predicted: list[str], ground_truth: list[str]) -> dict:
    """
    Compute precision, recall, F1 for extraction tasks (datasets, metrics, models).
    Treats each as a set membership problem.
    """
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


def run_evaluation(raw_outputs_path: str = "outputs/raw_outputs.json"):
    """
    Run evaluation on saved agent outputs.
    """
    print("\n=== Evaluation ===\n")

    # Load outputs
    with open(raw_outputs_path) as f:
        outputs = json.load(f)

    # --- Summarization ROUGE ---
    # Reference summary (I will replace this later with ground truth abstracts)
    reference_summary = (
        "We use the Transformer, a model based solely on attention mechanisms, "
        "achieving state-of-the-art BLEU scores on WMT 2014 translation tasks."
    )
    generated_summary = outputs["summary"].get("summary", "")
    rouge_scores = evaluate_summarization(generated_summary, reference_summary)
    print("Summarization ROUGE Scores:")
    for k, v in rouge_scores.items():
        print(f"  {k}: {v}")

    # --- Methodology Extraction P/R/F1 ---
    gt_datasets = ["wmt", "cifar", "mnist"]
    gt_metrics = ["bleu", "accuracy", "f1"]
    gt_models = ["transformer", "bert", "attention"]

    pred_datasets = outputs["methodology"].get("datasets_identified", [])
    pred_metrics = outputs["methodology"].get("evaluation_metrics", [])
    pred_models = outputs["methodology"].get("model_architectures", [])

    print("\nDataset Extraction:")
    print(f"  Predicted: {pred_datasets}")
    print(f"  Ground truth: {gt_datasets}")
    dataset_scores = evaluate_extraction(pred_datasets, gt_datasets)
    for k, v in dataset_scores.items():
        print(f"  {k}: {v}")

    print("\nMetric Extraction:")
    metric_scores = evaluate_extraction(pred_metrics, gt_metrics)
    for k, v in metric_scores.items():
        print(f"  {k}: {v}")

    print("\nModel Architecture Extraction:")
    model_scores = evaluate_extraction(pred_models, gt_models)
    for k, v in model_scores.items():
        print(f"  {k}: {v}")

    # Evaluation results
    eval_results = {
        "summarization": rouge_scores,
        "dataset_extraction": dataset_scores,
        "metric_extraction": metric_scores,
        "model_extraction": model_scores,
    }
    eval_path = "outputs/evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to {eval_path}")
    return eval_results


if __name__ == "__main__":
    run_evaluation()