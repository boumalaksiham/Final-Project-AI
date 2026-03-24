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
        "restricted to", "is limited", "only works", "does not scale",
        "computationally expensive", "requires large", "suffers from",
        "one disadvantage", "a downside", "not yet", "remains challenging",
        "quadratic", "memory", "inefficient",
    ]
    FUTURE_WORK_MARKERS = [
        "future work", "future research", "future direction", "in the future",
        "remains to be", "an open question", "promising direction", "can be extended",
        "could be explored", "plan to", "intend to", "next step",
        "further investigation", "leave for future",
    ]

    def __init__(self):
        print("[CriticalAnalysisAgent] Initialized.")

    def _extract_section(self, text: str, section_names: list) -> str:
        """Extract a named section from the paper, stopping before references."""
        lower = text.lower()

        # Find where references section starts so we can cut it off
        ref_idx = lower.rfind("references")
        if ref_idx == -1:
            ref_idx = len(text)

        body = text[:ref_idx]
        lower_body = body.lower()

        for name in section_names:
            start = 0
            while True:
                idx = lower_body.find(name, start)
                if idx == -1:
                    break
                snippet = body[idx:idx + 100].strip()
                lines = snippet.split("\n")
                second_line = lines[1].strip() if len(lines) > 1 else ""
                # Skip TOC entries — second line is just a page number
                if second_line.isdigit() or (len(second_line) < 5 and second_line.replace(".", "").isdigit()):
                    start = idx + len(name)
                    continue
                return body[idx : idx + 5000]

        return body[-4000:]

    def _extract_sentences_with_markers(self, text: str, markers: list) -> list:
        """Return sentences containing any of the marker phrases."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        found = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(marker in lower for marker in markers):
                clean = sentence.strip()
                if not (30 < len(clean) < 500):
                    continue
                if re.search(r"\b(19|20)\d{2}\b", clean) and len(clean) < 100:
                    continue
                if "et al" in lower and len(clean) < 120:
                    continue
                if not re.search(r"\b(is|are|was|were|has|have|can|cannot|will|would|require|suffer|limit|restrict|fail|does|do|need)\b", lower):
                    continue
                found.append(clean)
        return found[:8]

    def run(self, text: str) -> dict:
        print("[CriticalAnalysisAgent] Identifying limitations and future work...")

        discussion = self._extract_section(
            text, ["limitation", "limitations", "broader impact",
                   "conclusion", "discussion", "future work", "future directions"]
        )

        limitations = self._extract_sentences_with_markers(discussion, self.LIMITATION_MARKERS)
        future_directions = self._extract_sentences_with_markers(discussion, self.FUTURE_WORK_MARKERS)

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