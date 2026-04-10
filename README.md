# Multi-Agent Scientific Paper Analysis System

A multi-agent AI system that automatically analyzes scientific papers and generates comprehensive structured reports.

---

## Overview

Reading and comprehending academic papers is time-intensive. This system addresses that challenge by deploying five specialized AI agents that each handle a distinct analysis task — summarization, citation extraction, methodology identification, critical analysis, and report synthesis — and combine their outputs into a unified structured report.

The system handles three input modes:
- A built-in demo excerpt from *Attention Is All You Need* (Vaswani et al., 2017)
- A local PDF file
- Any arXiv paper fetched automatically by ID

---

## Architecture

| Agent | Task | Technique |
|-------|------|-----------|
| **Summarization Agent** | Generates concise paper summaries | BART (`facebook/bart-large-cnn`), section-aware chunking, PDF artifact cleaning |
| **Citation Analysis Agent** | Extracts references and builds knowledge graphs | Regex (3 citation formats), NetworkX, arXiv API for title fetching |
| **Methodology Extractor Agent** | Identifies datasets, metrics, and architectures | spaCy NER + curated keyword matching |
| **Critical Analysis Agent** | Extracts limitations and future work | Pattern matching on isolated sections, TOC-skip logic |
| **Coordinator Agent** | Synthesizes all outputs into a unified report | Multi-document synthesis, Markdown + JSON output |

---

## Project Structure

```
paper_analysis/
├── main.py                  # Pipeline entry point
├── evaluate.py              # ROUGE + precision/recall evaluation
├── ablation.py              # Ablation study (remove one agent at a time)
├── baseline.py              # Single-model baseline comparison
├── paper_loader.py          # PDF loading + arXiv fetching utilities
├── requirements.txt
├── agents/
│   ├── __init__.py
│   ├── summarizer.py        # Agent 1: Summarization (BART)
│   ├── citation_agent.py    # Agent 2: Citation Analysis
│   ├── methodology_agent.py # Agent 3: Methodology Extraction
│   ├── critical_agent.py    # Agent 4: Critical Analysis
│   └── coordinator.py       # Agent 5: Coordinator
├── data/                    # Downloaded/cached PDFs (gitignored)
└── outputs/                 # Generated reports and graphs (gitignored)
```

---

## Setup

### Requirements
- Python 3.10+
- macOS, Linux, or Windows

### Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Known issue — Apple Silicon (MPS backend)
BART produces empty output on Apple Silicon's MPS backend due to a known PyTorch/Transformers bug. This is already handled in the code by forcing CPU inference (`device=-1`). No action needed.

---

## Usage

### Run on built-in demo (no download needed)
```bash
python main.py --demo
```

### Analyze any arXiv paper by ID
```bash
python main.py --arxiv 1706.03762    # Attention Is All You Need
python main.py --arxiv 2005.14165    # GPT-3
python main.py --arxiv 1810.04805    # BERT
python main.py --arxiv 1512.03385    # ResNet
```

### Analyze a local PDF
```bash
python main.py --pdf path/to/paper.pdf
```

### Run evaluation (ROUGE + P/R/F1)
```bash
python evaluate.py 1706.03762
python evaluate.py 2005.14165
```

### Run ablation study
```bash
python ablation.py 1706.03762
python ablation.py 2005.14165
```

### Run single-model baseline comparison
```bash
python baseline.py 1706.03762
python baseline.py 2005.14165
```

---

## Output

Each run produces the following in `outputs/`:

| File | Description |
|------|-------------|
| `analysis_report.md` | Full structured analysis report |
| `citation_graph.png` | Directed knowledge graph visualization |
| `raw_outputs.json` | Raw JSON from all agents |
| `evaluation_results.json` | ROUGE + P/R/F1 scores |
| `ablation_<id>.json` | Ablation study results |
| `baseline_comparison_<id>.json` | Baseline vs. multi-agent comparison |

---

## Evaluation Results

Evaluated on four papers: *Attention Is All You Need*, *GPT-3*, *BERT*, and *Deep Residual Learning for Image Recognition*.

### Multi-Agent System

| Paper | ROUGE-1 | ROUGE-2 | Dataset F1 | Metric F1 | Model F1 | Citations |
|-------|---------|---------|------------|-----------|----------|-----------|
| Attention Is All You Need | 0.458 | 0.192 | 0.800 | 0.444 | **1.000** | 30 |
| GPT-3 | **0.546** | **0.419** | **1.000** | **1.000** | **1.000** | 30 |
| BERT | 0.374 | — | 0.800 | 0.444 | 0.875 | 30 |
| ResNet | 0.347 | — | 0.800 | 0.500 | 0.333 | 30 |

### Multi-Agent vs. Single-Model Baseline (averaged across papers)

| Metric | Baseline | Multi-Agent | Improvement |
|--------|----------|-------------|-------------|
| ROUGE-1 | 0.19 | 0.50 | **+0.31** |
| ROUGE-2 | 0.11 | 0.31 | **+0.20** |
| Dataset F1 | 0.70 | 0.90 | **+0.20** |
| Metric F1 | 0.44 | 0.72 | **+0.28** |
| Model F1 | 0.55 | 1.00 | **+0.45** |

### Ablation Study

Removing any single agent causes complete failure in its dimension:

| Agent Removed | Impact |
|---------------|--------|
| Summarization | ROUGE drops to 0.000 |
| Citation | References drop from 30 to 0 |
| Methodology | All extraction F1 drops to 0.000 |
| Critical | Limitations drop to 0 |

---

## Tech Stack

- **Python 3.12**
- **HuggingFace Transformers** — BART summarization
- **PyTorch** — model backend
- **spaCy** — named entity recognition
- **NetworkX + Matplotlib** — citation graph construction and visualization
- **PyMuPDF (fitz)** — PDF text extraction
- **rouge-score** — ROUGE evaluation
- **scikit-learn** — precision/recall/F1
- **arXiv API** — paper fetching and metadata

---

## Course

CS 5600 — Foundation of Artificial Intelligence 
Khoury College of Computer Sciences, Northeastern University  
Siham Boumalak — Spring 2026