"""
Single-Model Baseline
Instead of 5 specialized agents, uses one BART model to extract
all information from the paper in a single pass.
Results are compared against the multi-agent system.
"""

import json
import re
from pathlib import Path
from transformers import pipeline
from paper_loader import load_arxiv
from evaluate import evaluate_summarization, evaluate_extraction

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

KNOWN_DATASETS = [
    "wmt", "wmt 2014", "imagenet", "cifar", "mnist", "coco", "squad",
    "glue", "superglue", "arxiv", "pubmed", "common crawl", "bookcorpus",
    "openwebtext", "penn treebank", "conll", "ms marco",
]
KNOWN_METRICS = [
    "bleu", "rouge", "accuracy", "f1", "precision", "recall",
    "perplexity", "ppl", "em", "exact match", "auc", "map",
    "roc", "wer", "cer", "mse", "mae",
]
KNOWN_MODELS = [
    "transformer", "bert", "gpt", "t5", "bart", "roberta", "lstm", "gru",
    "cnn", "resnet", "attention", "encoder", "decoder", "seq2seq",
]


class SingleModelBaseline:
    """
    Baseline: one BART model handles everything.
    - Summarization: BART on paper text
    - Extraction: simple keyword scan on raw text (no NER, no agents)
    - Citations: simple regex on raw text (no section isolation)
    - Critical analysis: keyword scan on full text (no section targeting)
    """

    def __init__(self):
        print("[Baseline] Loading BART model (single model for all tasks)...")
        self.model = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    def summarize(self, text: str) -> str:
        """Single-pass summarization — just take first 700 words."""
        words = text.split()[:700]
        combined = " ".join(words)
        if len(combined.split()) < 60:
            return ""
        try:
            result = self.model(combined, max_length=200, min_length=50,
                                do_sample=False, truncation=True)
            return result[0]["summary_text"]
        except Exception as e:
            print(f"  [Baseline] Summarization failed: {e}")
            return ""

    def extract_datasets(self, text: str) -> list:
        """Keyword scan on full text — no section isolation."""
        lower = text.lower()
        return [kw for kw in KNOWN_DATASETS if kw in lower]

    def extract_metrics(self, text: str) -> list:
        """Keyword scan on full text — no section isolation."""
        lower = text.lower()
        return [kw for kw in KNOWN_METRICS if kw in lower]

    def extract_models(self, text: str) -> list:
        """Keyword scan on full text — no section isolation."""
        lower = text.lower()
        return [kw for kw in KNOWN_MODELS if kw in lower]

    def extract_references(self, text: str) -> list:
        """Simple regex on full text — no section isolation."""
        matches = re.findall(r"(?:\[\d+\]|\d+\.)\s+([A-Z][^\n]{20,})", text)
        return matches[:30]

    def extract_limitations(self, text: str) -> list:
        """Keyword scan on full text without targeting sections."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        found = []
        for s in sentences:
            if any(kw in s.lower() for kw in ["limitation", "drawback", "however", "cannot", "unable"]):
                if 40 < len(s) < 300:
                    found.append(s.strip())
        return found[:5]

    def run(self, text: str) -> dict:
        print("[Baseline] Running single-model analysis...")
        summary = self.summarize(text)
        datasets = self.extract_datasets(text)
        metrics = self.extract_metrics(text)
        models = self.extract_models(text)
        references = self.extract_references(text)
        limitations = self.extract_limitations(text)
        print(f"  Summary: {len(summary)} chars")
        print(f"  Datasets: {datasets}")
        print(f"  Refs: {len(references)}")
        return {
            "summary": summary,
            "datasets": datasets,
            "metrics": metrics,
            "models": models,
            "references": references,
            "limitations": limitations,
        }


def compare(paper_id: str, multi_agent_outputs_path: str = None):
    print(f"\n{'='*60}")
    print(f"  Baseline vs Multi-Agent — arXiv:{paper_id}")
    print(f"{'='*60}\n")

    gt = GROUND_TRUTH[paper_id]
    text, _ = load_arxiv(paper_id)

    # Run baseline
    baseline = SingleModelBaseline()
    b = baseline.run(text)

    # Score baseline
    b_rouge  = evaluate_summarization(b["summary"], gt["reference_summary"])
    b_ds     = evaluate_extraction(b["datasets"], gt["datasets"])
    b_mt     = evaluate_extraction(b["metrics"],  gt["metrics"])
    b_mo     = evaluate_extraction(b["models"],   gt["models"])

    # Load multi-agent outputs if available
    ma_path = multi_agent_outputs_path or "outputs/raw_outputs.json"
    try:
        with open(ma_path) as f:
            ma = json.load(f)
        ma_rouge = evaluate_summarization(
            ma["summary"].get("summary", ""), gt["reference_summary"])
        ma_ds = evaluate_extraction(
            ma["methodology"].get("datasets_identified", []), gt["datasets"])
        ma_mt = evaluate_extraction(
            ma["methodology"].get("evaluation_metrics", []), gt["metrics"])
        ma_mo = evaluate_extraction(
            ma["methodology"].get("model_architectures", []), gt["models"])
        ma_refs = ma["citations"].get("num_references", 0)
    except Exception as e:
        print(f"  Could not load multi-agent outputs: {e}")
        ma_rouge = ma_ds = ma_mt = ma_mo = {"rouge1":0,"rouge2":0,"rougeL":0,"f1":0}
        ma_refs = 0

    # Print comparison table
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Multi-Agent':>12} {'Delta':>10}")
    print("-" * 62)
    metrics_to_compare = [
        ("ROUGE-1",      b_rouge["rouge1"],  ma_rouge["rouge1"]),
        ("ROUGE-2",      b_rouge["rouge2"],  ma_rouge["rouge2"]),
        ("ROUGE-L",      b_rouge["rougeL"],  ma_rouge["rougeL"]),
        ("Dataset F1",   b_ds["f1"],         ma_ds["f1"]),
        ("Metric F1",    b_mt["f1"],         ma_mt["f1"]),
        ("Model F1",     b_mo["f1"],         ma_mo["f1"]),
        ("# References", len(b["references"]), ma_refs),
        ("# Limitations",len(b["limitations"]), ma.get("critical",{}).get("num_limitations_found",0) if "ma" in dir() else 0),
    ]
    for name, bval, maval in metrics_to_compare:
        if isinstance(bval, float):
            delta = maval - bval
            print(f"{name:<25} {bval:>12.4f} {maval:>12.4f} {delta:>+10.4f}")
        else:
            delta = maval - bval
            print(f"{name:<25} {bval:>12} {maval:>12} {delta:>+10}")

    # Save results
    results = {
        "paper_id": paper_id,
        "paper": gt["title"],
        "baseline": {
            "rouge": b_rouge, "dataset_f1": b_ds["f1"],
            "metric_f1": b_mt["f1"], "model_f1": b_mo["f1"],
            "num_references": len(b["references"]),
        },
        "multi_agent": {
            "rouge": ma_rouge, "dataset_f1": ma_ds["f1"],
            "metric_f1": ma_mt["f1"], "model_f1": ma_mo["f1"],
            "num_references": ma_refs,
        },
    }
    out_path = f"outputs/baseline_comparison_{paper_id.replace('.','_')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    import sys
    paper_id = sys.argv[1] if len(sys.argv) > 1 else "1706.03762"
    # First run the multi-agent pipeline to get fresh outputs
    from main import run_pipeline
    from paper_loader import load_arxiv
    text, _ = load_arxiv(paper_id)
    run_pipeline(text, source=f"arxiv:{paper_id}")
    compare(paper_id)