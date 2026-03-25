"""
Summarization Agent
Uses facebook/bart-large-cnn to generate concise summaries of scientific paper text.
"""

import re
from transformers import pipeline


class SummarizationAgent:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        print("[SummarizationAgent] Loading model...")
        self.summarizer = pipeline("summarization", model=model_name, device=-1)

    def _clean_text(self, text: str) -> str:
        """Remove PDF artifacts: hyphenated line breaks, ligatures, extra whitespace."""
        text = re.sub(r'-\n\s*', '', text)        # join hyphenated words
        text = re.sub(r'\n+', ' ', text)           # flatten newlines
        text = re.sub(r'\s+', ' ', text).strip()   # normalize spaces
        text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')  # ligatures
        return text

    def _extract_key_sections(self, text: str) -> str:
        """Extract abstract + intro + conclusion and combine into one passage."""
        # Strip ARXIV_TITLE prefix if present
        if text.startswith("ARXIV_TITLE:"):
            text = "\n".join(text.split("\n")[2:])

        lower = text.lower()
        abstract_idx   = lower.find("abstract")
        intro_idx      = lower.find("introduction")
        conclusion_idx = lower.rfind("conclusion")

        chunks = []

        if abstract_idx != -1:
            end = intro_idx if (intro_idx != -1 and intro_idx > abstract_idx) else abstract_idx + 2000
            chunk = text[abstract_idx:end].strip()
            if len(chunk.split()) > 30:
                chunks.append(chunk)

        if intro_idx != -1:
            chunk = text[intro_idx:intro_idx + 2000].strip()
            if len(chunk.split()) > 30:
                chunks.append(chunk)

        if conclusion_idx != -1:
            chunk = text[conclusion_idx:conclusion_idx + 1500].strip()
            if len(chunk.split()) > 30:
                chunks.append(chunk)

        combined = " ".join(chunks)

        # Fallback to first 700 words if sections not found or too short
        if len(combined.split()) < 100:
            combined = " ".join(text.split()[:700])

        # Cap at 700 words
        words = combined.split()
        if len(words) > 700:
            combined = " ".join(words[:700])

        return combined

    def run(self, text: str) -> dict:
        """Summarize the paper text."""
        print("[SummarizationAgent] Summarizing paper...")

        combined = self._extract_key_sections(text)
        combined = self._clean_text(combined)
        word_count = len(combined.split())

        full_summary = ""
        chunk_summaries = []

        if word_count < 30:
            print("  [SummarizationAgent] Not enough text to summarize.")
            return {"summary": "", "chunk_summaries": []}

        max_len = min(200, max(60, word_count // 3))
        min_len = min(40, max(20, word_count // 8))

        if min_len >= max_len:
            min_len = max(10, max_len // 2)

        try:
            result = self.summarizer(
                combined,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            full_summary = result[0]["summary_text"]
            chunk_summaries = [full_summary]
        except Exception as e:
            print(f"  [SummarizationAgent] Summarization failed: {e}")

        return {
            "summary": full_summary,
            "chunk_summaries": chunk_summaries,
        }