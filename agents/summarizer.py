"""
Summarization Agent
Uses facebook/bart-large-cnn to generate concise summaries of scientific paper text.
"""

from transformers import pipeline


class SummarizationAgent:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        print("[SummarizationAgent] Loading model...")
        self.summarizer = pipeline("summarization", model=model_name, device=-1)
        self.max_chunk_tokens = 1024

    def _chunk_text(self, text: str, max_chars: int = 3000) -> list[str]:
        """Split long text into overlapping chunks for summarization."""
        # Try to extract abstract first — best dense summary source
        lower = text.lower()
        abstract_idx = lower.find("abstract")
        intro_idx = lower.find("introduction")
        conclusion_idx = lower.rfind("conclusion")

        chunks = []

        # Chunk 1: abstract if present
        if abstract_idx != -1:
            end = intro_idx if intro_idx > abstract_idx else abstract_idx + 2000
            abstract = text[abstract_idx:end].strip()
            if len(abstract.split()) > 50:
                chunks.append(abstract)

        # Chunk 2: introduction
        if intro_idx != -1:
            intro = text[intro_idx:intro_idx + 3000].strip()
            if len(intro.split()) > 50:
                chunks.append(intro)

        # Chunk 3: conclusion
        if conclusion_idx != -1:
            conclusion = text[conclusion_idx:conclusion_idx + 2000].strip()
            if len(conclusion.split()) > 50:
                chunks.append(conclusion)

        # Fallback: split full text if nothing found
        if not chunks:
            words = text.split()
            chunk_size = max_chars // 5
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i : i + chunk_size]))

        return chunks

    def run(self, text: str) -> dict:
        """
        Summarize the full paper text.
        Extracts abstract + intro + conclusion, combines them, and summarizes once.
        Returns a dict with 'summary' and 'chunk_summaries'.
        """
        print("[SummarizationAgent] Summarizing paper...")
        chunks = self._chunk_text(text)

        # Combine all chunks into one passage for a richer summary
        combined = " ".join(chunks)
        words = combined.split()

        # BART handles ~700 words comfortably on CPU; truncate if longer
        if len(words) > 700:
            combined = " ".join(words[:700])

        if len(combined.split()) < 60:
            # Not enough text — fall back to first 700 words of full text
            combined = " ".join(text.split()[:700])

        chunk_summaries = []
        full_summary = ""

        try:
            word_count = len(combined.split())
            max_len = min(250, max(80, word_count // 3))
            min_len = min(60, max(30, word_count // 8))
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