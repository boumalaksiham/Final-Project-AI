# Multi-Agent Scientific Paper Analysis System

A multi-agent AI system that automatically analyzes scientific papers and generates comprehensive structured reports. Built for EECE 5644 — Machine Learning.

## Architecture

Five specialized agents, each handling a distinct analysis task:

| Agent | Task | Technique |
|-------|------|-----------|
| **Summarization Agent** | Generates concise paper summaries | BART (facebook/bart-large-cnn) |
| **Citation Analysis Agent** | Extracts references + builds knowledge graphs | Regex + NetworkX |
| **Methodology Extractor Agent** | Identifies datasets, metrics, architectures | NER (spaCy) + keyword matching |
| **Critical Analysis Agent** | Identifies limitations and future directions | Pattern matching on discussion sections |
| **Coordinator Agent** | Synthesizes all outputs into a unified report | Multi-document synthesis |

## Setup

```bash
# Clone the repo and install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Run on built-in demo paper (no internet needed)
python main.py --demo

# Analyze a local PDF
python main.py --pdf path/to/paper.pdf

# Analyze any arXiv paper by ID
python main.py --arxiv 1706.03762

# Run evaluation metrics
python evaluate.py
```

## Output

Running the pipeline produces:
- `outputs/analysis_report.md` — full structured report
- `outputs/citation_graph.png` — knowledge graph visualization
- `outputs/raw_outputs.json` — raw JSON from all agents
- `outputs/evaluation_results.json` — ROUGE + P/R/F1 scores

## Evaluation

- **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L vs reference abstracts
- **Extraction**: Precision, Recall, F1 for datasets/metrics/architectures
- **Ablation study**: Compare full system vs. removing individual agents

## Project Structure

```
paper_analysis/
├── main.py                  # Pipeline entry point
├── evaluate.py              # Evaluation metrics
├── paper_loader.py          # PDF + arXiv loading utilities
├── requirements.txt
├── agents/
│   ├── __init__.py
│   ├── summarizer.py        # Agent 1: Summarization
│   ├── citation_agent.py    # Agent 2: Citation Analysis
│   ├── methodology_agent.py # Agent 3: Methodology Extraction
│   ├── critical_agent.py    # Agent 4: Critical Analysis
│   └── coordinator.py       # Agent 5: Coordinator
├── data/                    # Downloaded/cached PDFs
└── outputs/                 # Generated reports and graphs
```