"""
Multi-Agent Scientific Paper Analysis System
Main pipeline — runs all 6 agents on a given paper.

Usage:
    # Analyze a local PDF:
    python main.py --pdf path/to/paper.pdf

    # Analyze an arXiv paper by ID:
    python main.py --arxiv 1706.03762

    # Analyze with a research topic for relevance scoring:
    python main.py --arxiv 1706.03762 --topic "attention mechanisms NLP"

    # Analyze the built-in demo text:
    python main.py --demo

Demo mode:
    The --demo flag runs the pipeline on a built-in excerpt from:
    Vaswani et al. (2017), "Attention Is All You Need", NeurIPS 2017.
    arXiv: https://arxiv.org/abs/1706.03762

    This paper introduced the Transformer architecture and is used here
    as a well-known benchmark text to verify the pipeline works correctly
    without needing to download or provide a PDF.
"""

import argparse
from pathlib import Path

from agents import (
    SummarizationAgent,
    CitationAnalysisAgent,
    MethodologyExtractorAgent,
    CriticalAnalysisAgent,
    CoordinatorAgent,
    RelevanceAgent,
)
from paper_loader import load_pdf, load_arxiv, load_text, load_pubmed

OUTPUT_DIR = "outputs"


DEMO_TEXT = """
Attention Is All You Need

Abstract: The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best performing
models also connect the encoder and decoder through an attention mechanism. We propose
a new simple network architecture, the Transformer, based solely on attention mechanisms,
dispensing with recurrence and convolutions entirely.

Introduction: Recurrent neural networks, long short-term memory and gated recurrent neural
networks in particular, have been firmly established as state of the art approaches in
sequence modeling. Numerous efforts have since continued to push the boundaries of recurrent
language models and encoder-decoder architectures.

Methodology: We trained on the standard WMT 2014 English-German dataset consisting of
about 4.5 million sentence pairs. We used the Adam optimizer with beta1=0.9, beta2=0.98.
We used multi-head attention with 8 heads. The model was evaluated using BLEU score.
The Transformer uses BERT-style embeddings and ResNet skip connections for stability.
The dataset splits followed standard CIFAR and MNIST benchmarks for ablation studies.
Evaluation metrics include BLEU, accuracy, and F1-score.

Results: On the WMT 2014 English-to-German translation task, the big transformer model
outperforms the best previously reported models including ensembles by more than 2.0 BLEU,
establishing a new state-of-the-art BLEU score of 28.4.

Limitations: The model requires significantly more memory than recurrent architectures.
However, training time is substantially reduced. One limitation is the quadratic complexity
of self-attention with respect to sequence length. Despite strong results, the model
cannot handle extremely long sequences efficiently. Unfortunately, we do not address
streaming inference in this work.

Future Work: Future research could explore linear attention mechanisms to address the
quadratic complexity limitation. In the future, we plan to extend the Transformer to
other modalities including images and audio. This direction remains to be explored
fully. Further investigation is needed for low-resource settings.

Conclusion: We presented the Transformer, the first sequence transduction model based
entirely on attention. We are excited about the future of attention-based models and
plan to apply them to other problems.

References:
[1] Bahdanau, D., Cho, K., Bengio, Y. Neural machine translation by jointly learning to align and translate. ICLR 2015.
[2] Hochreiter, S., Schmidhuber, J. Long short-term memory. Neural Computation, 1997.
[3] Sutskever, I., Vinyals, O., Le, Q. Sequence to sequence learning with neural networks. NeurIPS 2014.
[4] Cho, K., et al. Learning phrase representations using RNN encoder-decoder. EMNLP 2014.
[5] Devlin, J., Chang, M., Lee, K., Toutanova, K. BERT: Pre-training of deep bidirectional transformers. NAACL 2019.
"""


def run_pipeline(text: str, source: str = "Unknown", user_topic: str = ""):
    """Run all 6 agents on the provided text."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  Multi-Agent Scientific Paper Analysis System")
    print("=" * 60 + "\n")

    # Agent 1: Summarization
    summarizer = SummarizationAgent()
    summary_output = summarizer.run(text)
    print(f"  ✓ Summary generated ({len(summary_output['summary'])} chars)\n")

    # Agent 2: Citation Analysis
    citation_agent = CitationAnalysisAgent()
    citation_output = citation_agent.run(text, output_dir=OUTPUT_DIR)
    print(f"  ✓ {citation_output['num_references']} references extracted\n")

    # Agent 3: Methodology Extraction
    methodology_agent = MethodologyExtractorAgent()
    methodology_output = methodology_agent.run(text)
    print(f"  ✓ Methodology extracted\n")

    # Agent 4: Critical Analysis
    critical_agent = CriticalAnalysisAgent()
    critical_output = critical_agent.run(text)
    print(f"  ✓ {critical_output['num_limitations_found']} limitations found\n")

    # Agent 5: Relevance Scoring + TL;DR
    relevance_agent = RelevanceAgent()
    relevance_output = relevance_agent.run(
        text=text,
        summary=summary_output["summary"],
        methodology=methodology_output,
        citations=citation_output,
        user_topic=user_topic,
    )
    print(f"  ✓ TL;DR and relevance score generated\n")

    # Agent 6: Coordinator
    coordinator = CoordinatorAgent()
    final_output = coordinator.run(
        summary_output,
        citation_output,
        methodology_output,
        critical_output,
        relevance_output=relevance_output,
        paper_source=source,
        output_dir=OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print(f"  ✅ Analysis complete!")
    print(f"  📄 Report: {final_output['report_path']}")
    print(f"  📊 Graph:  {citation_output.get('graph_image', 'N/A')}")
    print(f"  🗂  JSON:   {final_output['json_path']}")
    print("=" * 60 + "\n")

    print(final_output["report"])
    return final_output


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Scientific Paper Analysis System"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",   type=str, help="Path to a local PDF file")
    group.add_argument("--arxiv", type=str, help="arXiv paper ID (e.g. 1706.03762)")
    group.add_argument("--txt",    type=str, help="Path to a plain text file")
    group.add_argument("--pubmed", type=str, help="PubMed ID (e.g. 33278872)")
    group.add_argument("--demo",   action="store_true", help="Run on built-in demo paper")
    parser.add_argument(
        "--topic", type=str, default="",
        help="Your research interest for relevance scoring (e.g. 'transformer NLP')"
    )
    args = parser.parse_args()

    if args.demo:
        print("[Main] Running in demo mode (Attention Is All You Need excerpt)...")
        run_pipeline(DEMO_TEXT, source="demo/attention_is_all_you_need",
                     user_topic=args.topic)

    elif args.pdf:
        print(f"[Main] Loading PDF: {args.pdf}")
        text = load_pdf(args.pdf)
        run_pipeline(text, source=args.pdf, user_topic=args.topic)

    elif args.arxiv:
        print(f"[Main] Fetching arXiv: {args.arxiv}")
        text, path = load_arxiv(args.arxiv)
        run_pipeline(text, source=f"arxiv:{args.arxiv}", user_topic=args.topic)

    elif args.txt:
        print(f"[Main] Loading text file: {args.txt}")
        text = load_text(args.txt)
        run_pipeline(text, source=args.txt, user_topic=args.topic)

    elif args.pubmed:
        print(f"[Main] Fetching PubMed: {args.pubmed}")
        text, title = load_pubmed(args.pubmed)
        run_pipeline(text, source=f"pubmed:{args.pubmed}", user_topic=args.topic)


if __name__ == "__main__":
    main()