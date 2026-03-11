# Multi-Agent Scientific Paper Analysis System

A multi-agent AI system that automatically analyzes scientific papers and generates comprehensive structured reports. Each agent specializes in a distinct analysis task — summarization, citation extraction, methodology identification, critical analysis, and report synthesis — and they work together in a pipeline to process any research paper.

Built for CS 5100 – Foundations of Artificial Intelligence.

---

## What It Does

Given a scientific paper (PDF, arXiv ID, or plain text), the system produces:
- A concise **summary** of the paper
- A **citation knowledge graph** showing relationships between references
- Extracted **datasets, evaluation metrics, and model architectures**
- Identified **limitations and future research directions**
- A unified **structured report** combining all of the above

---

## Agent Architecture

| Agent | Responsibility | Technique |
|-------|---------------|-----------|
| Summarization Agent | Generates concise paper summaries | BART (`facebook/bart-large-cnn`) |
| Citation Analysis Agent | Extracts references + builds knowledge graph | Regex + NetworkX |
| Methodology Extractor Agent | Identifies datasets, metrics, architectures | spaCy NER + keyword matching |
| Critical Analysis Agent | Finds limitations and future directions | Pattern matching on discussion sections |
| Coordinator Agent | Synthesizes all outputs into a final report | Multi-document synthesis |

---

## Prerequisites

- Python 3.9+
- pip
- Git

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/paper-analysis.git
cd paper-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy language model
python -m spacy download en_core_web_sm
```

---

## Usage

### Run on the built-in demo
```bash
python main.py --demo
```
This runs the pipeline on an excerpt from *"Attention Is All You Need"* (Vaswani et al., 2017, NeurIPS) — the paper that introduced the Transformer architecture. It is included as a built-in demo to verify the pipeline works correctly without requiring any external files.

### Analyze a local PDF
```bash
python main.py --pdf path/to/paper.pdf
```

### Analyze any arXiv paper by ID
```bash
python main.py --arxiv 1706.03762
```

### Run evaluation metrics
```bash
python evaluate.py
```

---

## Output

All outputs are saved to the `outputs/` folder:

| File | Description |
|------|-------------|
| `analysis_report.md` | Full structured analysis report |
| `citation_graph.png` | Knowledge graph of paper references |
| `raw_outputs.json` | Raw JSON from all agents |
| `evaluation_results.json` | ROUGE + precision/recall/F1 scores |

---

## Project Structure

```
paper_analysis/
├── main.py                   # Pipeline entry point
├── evaluate.py               # Evaluation metrics (ROUGE, P/R/F1)
├── paper_loader.py           # PDF and arXiv loading utilities
├── requirements.txt          # Python dependencies
├── README.md
├── agents/
│   ├── __init__.py
│   ├── summarizer.py         # Agent 1: Summarization (BART)
│   ├── citation_agent.py     # Agent 2: Citation Analysis
│   ├── methodology_agent.py  # Agent 3: Methodology Extraction
│   ├── critical_agent.py     # Agent 4: Critical Analysis
│   └── coordinator.py        # Agent 5: Coordinator (in progress)
├── data/                     # Cached downloaded PDFs
└── outputs/                  # Generated reports and graphs
```

---

## Known Issues

- BART summarization returns empty output on Apple Silicon when using the MPS backend. Fixed by forcing CPU inference (`device=-1`). 
- Very long papers (50+ pages) may require additional chunking tuning in `summarizer.py`.

---

## License

This project is for academic purposes only (CS 5100 coursework). Not licensed for commercial use.

---


Siham Boumalak — Northeastern University, MS Artificial Intelligence  
