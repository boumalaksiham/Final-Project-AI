"""
Methodology Extractor Agent
Uses spaCy NER + pattern matching to identify datasets, metrics, and model architectures.
"""

import re
import spacy
from collections import defaultdict


class MethodologyExtractorAgent:
    # Curated keyword lists for scientific ML papers
    DATASET_KEYWORDS = [
        "imagenet", "cifar", "mnist", "coco", "squad", "glue", "superglue",
        "wikitext", "penn treebank", "conll", "arxiv", "pubmed", "common crawl",
        "openwebtext", "bookcorpus", "ms marco", "natural questions", "trivia",
        "wmt", "wmt14", "wmt 2014", "multi30k", "voc", "ade20k", "cityscapes",
        "english-german", "english-french", "wsj", "wall street journal",
        "newstest", "europarl", "ptb", "ontonotes", "snli", "multinli",
        "sst", "imdb", "yelp", "amazon", "ag news", "dbpedia",
    ]
    METRIC_KEYWORDS = [
        "accuracy", "f1", "f1-score", "precision", "recall", "rouge",
        "bleu", "perplexity", "auc", "roc", "map", "ndcg", "mrr",
        "exact match", "em", "cer", "wer", "mse", "mae", "rmse",
        "top-1", "top-5", "ppl", "bits per character",
    ]
    MODEL_KEYWORDS = [
        "bert", "gpt", "t5", "bart", "roberta", "xlnet", "albert", "electra",
        "transformer", "lstm", "gru", "cnn", "resnet", "vgg", "efficientnet",
        "vit", "clip", "diffusion", "gan", "vae", "llama", "mistral",
        "attention", "encoder", "decoder", "seq2seq", "fine-tun",
    ]

    def __init__(self):
        print("[MethodologyExtractorAgent] Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("  spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def _keyword_search(self, text: str, keywords: list[str]) -> list[str]:
        """Find keywords in text (case-insensitive), return unique matches."""
        found = set()
        lower = text.lower()
        for kw in keywords:
            if kw in lower:
                found.add(kw)
        return sorted(found)

    def _extract_with_ner(self, text: str) -> dict:
        """Use spaCy NER to extract organizations, products, and named entities."""
        if not self.nlp:
            return {}
        doc = self.nlp(text[:50000])  # spaCy has limits
        entities = defaultdict(set)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "GPE"):
                entities[ent.label_].add(ent.text.strip())
        return {k: list(v) for k, v in entities.items()}

    def _extract_methodology_section(self, text: str) -> str:
        """Try to isolate the methodology/experiments section."""
        markers = ["method", "approach", "experiment", "model architecture", "proposed"]
        lower = text.lower()
        best_idx = -1
        for marker in markers:
            idx = lower.find(marker)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
        if best_idx != -1:
            return text[best_idx : best_idx + 8000]
        return text[:8000]

    def run(self, text: str) -> dict:
        print("[MethodologyExtractorAgent] Extracting methodology components...")
        method_section = self._extract_methodology_section(text)

        datasets = self._keyword_search(method_section, self.DATASET_KEYWORDS)
        metrics = self._keyword_search(method_section, self.METRIC_KEYWORDS)
        models = self._keyword_search(method_section, self.MODEL_KEYWORDS)
        ner_entities = self._extract_with_ner(method_section)

        return {
            "datasets_identified": datasets,
            "evaluation_metrics": metrics,
            "model_architectures": models,
            "ner_entities": ner_entities,
            "methodology_excerpt": method_section[:500] + "...",
        }