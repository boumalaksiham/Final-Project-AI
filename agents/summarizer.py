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
        words = text.split()
        chunks = []
        chunk_size = max_chars // 5  # approx words
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

    def run(self, text: str) -> dict:
        """
        Summarize the full paper text.
        Returns a dict with 'summary' and 'chunk_summaries'.
        """
        print("[SummarizationAgent] Summarizing paper...")
        chunks = self._chunk_text(text)
        chunk_summaries = []

        for i, chunk in enumerate(chunks[:4]):  # cap at 4 chunks (for speed)
            if len(chunk.strip()) < 50:
                continue
            word_count = len(chunk.split())
            max_len = min(150, max(30, word_count // 2))
            min_len = min(20, max(10, word_count // 4))
            try:
                result = self.summarizer(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True,
                )
                chunk_summaries.append(result[0]["summary_text"])
            except Exception as e:
                print(f"  [SummarizationAgent] Skipping chunk {i}: {e}")
                continue

        full_summary = " ".join(chunk_summaries)

        # Summarize the combined chunk summaries for a final concise output
        if len(chunk_summaries) > 1:
            word_count = len(full_summary.split())
            max_len = min(200, max(40, word_count // 2))
            min_len = min(30, max(10, word_count // 4))
            try:
                final = self.summarizer(
                    full_summary,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True,
                )
                full_summary = final[0]["summary_text"]
            except Exception as e:
                print(f"  [SummarizationAgent] Final summary failed: {e}")

        return {
            "summary": full_summary,
            "chunk_summaries": chunk_summaries,
        }