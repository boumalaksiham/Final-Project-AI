"""
Critical Analysis Agent
Identifies limitations, future work directions, and research gaps
by analyzing conclusion and discussion sections.
"""

import re


class CriticalAnalysisAgent:
    LIMITATION_MARKERS = [
        "limitation", "drawback", "shortcoming", "weakness", "constraint",
        "however", "despite", "unfortunately", "one issue", "a challenge",
        "not address", "fail to", "unable to", "cannot", "does not handle",
    ]
    FUTURE_WORK_MARKERS = [
        "future work", "future research", "future direction", "in the future",
        "remains to be", "an open question", "promising direction", "can be extended",
        "could be explored", "plan to", "intend to", "next step",
        "further investigation", "leave for future",
    ]

    def __init__(self):
        print("[CriticalAnalysisAgent] Initialized.")

    def _extract_section(self, text: str, section_names: list[str]) -> str:
        """Extract a named section from the paper."""
        lower = text.lower()
        for name in section_names:
            idx = lower.rfind(name)
            if idx != -1:
                return text[idx : idx + 5000]
        return text[-4000:]  # fallback: end of paper

    def _extract_sentences_with_markers(self, text: str, markers: list[str]) -> list[str]:
        """Return sentences containing any of the marker phrases."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        found = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(marker in lower for marker in markers):
                clean = sentence.strip()
                if 30 < len(clean) < 400:
                    found.append(clean)
        return found[:8]  # top 8

    def run(self, text: str) -> dict:
        print("[CriticalAnalysisAgent] Identifying limitations and future work...")

        # Search conclusion + discussion sections
        discussion = self._extract_section(
            text, ["conclusion", "discussion", "limitation", "future work"]
        )

        limitations = self._extract_sentences_with_markers(discussion, self.LIMITATION_MARKERS)
        future_directions = self._extract_sentences_with_markers(discussion, self.FUTURE_WORK_MARKERS)

        # If none found, note that explicitly
        if not limitations:
            limitations = ["No explicit limitations section detected in this paper."]
        if not future_directions:
            future_directions = ["No explicit future work section detected in this paper."]

        return {
            "limitations": limitations,
            "future_directions": future_directions,
            "num_limitations_found": len(limitations),
            "num_future_directions_found": len(future_directions),
        }