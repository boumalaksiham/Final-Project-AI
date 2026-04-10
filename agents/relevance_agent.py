"""
Relevance Agent
Given a user's research interest (e.g. "reinforcement learning"),
scores how relevant the paper is and generates a one-sentence TL;DR.

This directly serves the paper triage use case: a researcher needs to
quickly decide whether a paper is worth reading before spending time on it.
"""

import re


class RelevanceAgent:
    def __init__(self):
        print("[RelevanceAgent] Initialized.")

    def _generate_tldr(self, summary: str, methodology: dict, citations: dict) -> str:
        """
        Generate a single-sentence TL;DR from available agent outputs.
        Format: "This paper proposes [contribution] using [method] on [datasets],
                 evaluated with [metrics]."
        """
        if not summary:
            return "No summary available to generate TL;DR."

        # Extract key pieces
        models   = methodology.get("model_architectures", [])
        datasets = methodology.get("datasets_identified", [])
        metrics  = methodology.get("evaluation_metrics", [])
        title    = citations.get("paper_title", "This paper")

        # Build TL;DR from summary first sentence + key facts
        first_sentence = re.split(r"(?<=[.!?])\s+", summary.strip())[0]

        extras = []
        if models:
            extras.append(f"using {', '.join(models[:2])}")
        if datasets:
            extras.append(f"evaluated on {', '.join(datasets[:2])}")
        if metrics:
            extras.append(f"measured by {', '.join(metrics[:2])}")

        if extras:
            tldr = first_sentence.rstrip(".") + " — " + ", ".join(extras) + "."
        else:
            tldr = first_sentence

        return tldr

    def _compute_relevance_score(self, text: str, user_topic: str) -> dict:
        """
        Score how relevant the paper is to the user's research topic.

        Strategy:
        - Split user topic into keywords
        - Count how many appear in the paper text (title, abstract, intro)
        - Normalize to a 0-100 score
        - Provide a human-readable verdict
        """
        if not user_topic.strip():
            return {
                "score": None,
                "verdict": "No research topic provided.",
                "matched_keywords": [],
                "user_topic": user_topic,
            }

        # Focus on the first 3000 chars (title + abstract + intro)
        focus_text = text[:3000].lower()

        # Split topic into meaningful keywords (ignore short words)
        stop = {"a", "an", "the", "and", "or", "for", "of", "in", "on",
                "to", "is", "are", "with", "that", "this", "how", "using"}
        keywords = [
            w.lower().strip(".,;:")
            for w in user_topic.split()
            if len(w) > 3 and w.lower() not in stop
        ]

        if not keywords:
            return {
                "score": 0,
                "verdict": "Could not extract keywords from topic.",
                "matched_keywords": [],
                "user_topic": user_topic,
            }

        # Count matches
        matched = [kw for kw in keywords if kw in focus_text]
        score = round((len(matched) / len(keywords)) * 100)

        # Verdict
        if score >= 70:
            verdict = "Highly relevant — strongly recommend reading."
        elif score >= 40:
            verdict = "Possibly relevant — skim the abstract and introduction."
        elif score >= 15:
            verdict = "Loosely related — may contain useful references."
        else:
            verdict = "Likely not relevant to your research topic."

        return {
            "score": score,
            "verdict": verdict,
            "matched_keywords": matched,
            "total_keywords": len(keywords),
            "user_topic": user_topic,
        }

    def run(self, text: str, summary: str, methodology: dict,
            citations: dict, user_topic: str = "") -> dict:
        """
        Run the relevance agent.

        Args:
            text:        Full paper text
            summary:     Generated summary from SummarizationAgent
            methodology: Output from MethodologyExtractorAgent
            citations:   Output from CitationAnalysisAgent
            user_topic:  User's research interest (e.g. "transformer NLP")

        Returns dict with tldr, relevance_score, verdict, matched_keywords
        """
        print("[RelevanceAgent] Computing relevance and TL;DR...")

        tldr = self._generate_tldr(summary, methodology, citations)
        relevance = self._compute_relevance_score(text, user_topic)

        print(f"  TL;DR: {tldr[:100]}...")
        if relevance["score"] is not None:
            print(f"  Relevance score: {relevance['score']}/100 — {relevance['verdict']}")

        return {
            "tldr": tldr,
            "relevance_score": relevance["score"],
            "relevance_verdict": relevance["verdict"],
            "matched_keywords": relevance["matched_keywords"],
            "user_topic": user_topic,
        }