# Multi-Agent Scientific Paper Triage System

A modular NLP system for **scientific paper triage**: helping researchers quickly understand a paper's contribution, methodology, limitations, references, and relevance before deciding whether to read it in depth.

The system combines six specialized components for summarization, citation analysis, methodology extraction, critical analysis, relevance scoring, and report synthesis. Rather than relying on a single model for every task, each component uses the technique best suited to its objective.

---

## Overview

Given a scientific paper as a **PDF, arXiv ID, or PubMed ID**, the system generates a structured research-triage report containing:

* **TL;DR** — a concise statement of the paper's main contribution
* **Relevance score** — a 0–100 lexical match against a user-specified research topic
* **Abstractive summary** — generated with BART
* **Methodology extraction** — detected datasets, evaluation metrics, and model architectures
* **Limitations and future work** — extracted from relevant sections of the paper
* **Citation graph** — a directed graph connecting the paper to extracted references
* **Agent summary** — status and primary output of each pipeline component
* **Raw JSON output** — structured outputs for downstream analysis

The project also includes **quantitative evaluation, baseline comparison, ablation analysis, cross-paper citation analysis, and qualitative evaluation on unseen papers**.

---

## Why a Multi-Agent Architecture?

Scientific paper analysis involves several different NLP and information-extraction problems.

A summarization model is useful for generating an overview, but it is not necessarily the best tool for extracting citation structure, identifying datasets, or determining whether a paper contains an explicit limitations section.

This system therefore decomposes paper analysis into specialized components:

1. each component performs one clearly defined task;
2. outputs can be evaluated independently;
3. failures can be traced to a specific stage of the pipeline; and
4. individual components can be replaced or improved without redesigning the entire system.

The architecture also makes it possible to perform **component-level ablation studies** and compare the complete system with a simpler monolithic baseline.

---

## System Architecture

```text
INPUT
PDF / arXiv ID / PubMed ID
        │
        ▼
┌───────────────────────────────┐
│         Paper Loader          │
│ PyMuPDF + arXiv + NCBI APIs   │
└──────────────┬────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│                                             │
│  1. Summarization Agent                    │
│     BART + section-aware preprocessing      │
│                                             │
│  2. Citation Analysis Agent                │
│     Regex + NetworkX                        │
│                                             │
│  3. Methodology Extraction Agent           │
│     spaCy + curated terminology             │
│                                             │
│  4. Critical Analysis Agent                │
│     Section detection + pattern matching    │
│                                             │
│  5. Relevance & TL;DR Agent                │
│     Lexical relevance scoring               │
│                                             │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
             6. Coordinator
          Output synthesis layer
                      │
                      ▼
      ┌───────────────────────────────┐
      │ Markdown Report               │
      │ Citation Graph                │
      │ Structured JSON               │
      └───────────────────────────────┘
```

---

## Components

| Component                  | Responsibility                                                        | Approach                                                                        |
| -------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Summarization**          | Generate an abstractive paper summary                                 | BART (`facebook/bart-large-cnn`), section-aware chunking, PDF artifact cleaning |
| **Citation Analysis**      | Extract references and construct citation graphs                      | Regex-based reference parsing, NetworkX, arXiv metadata                         |
| **Methodology Extraction** | Identify datasets, metrics, and model architectures                   | spaCy NER + curated scientific terminology                                      |
| **Critical Analysis**      | Identify author-stated limitations and future work                    | Section isolation + pattern matching                                            |
| **Relevance & TL;DR**      | Estimate topic relevance and produce a concise contribution statement | Keyword overlap + structured sentence synthesis                                 |
| **Coordinator**            | Merge outputs into a unified research report                          | Structured aggregation, Markdown generation, JSON serialization                 |

---

## Project Structure

```text
paper_analysis/
│
├── main.py
│   └── Main pipeline and CLI
│
├── paper_loader.py
│   └── PDF extraction + arXiv + PubMed retrieval
│
├── agents/
│   ├── summarizer.py
│   ├── citation_agent.py
│   ├── methodology_agent.py
│   ├── critical_agent.py
│   ├── relevance_agent.py
│   └── coordinator.py
│
├── evaluate.py
│   └── ROUGE + precision/recall/F1 evaluation
│
├── baseline.py
│   └── Simpler monolithic pipeline for comparison
│
├── ablation.py
│   └── Component-level ablation experiments
│
├── cross_paper_graph.py
│   └── Shared-reference analysis across papers
│
├── qualitative_eval.py
│   └── Human-guided evaluation on unseen papers
│
├── generate_figures.py
│   └── Evaluation visualization utilities
│
├── requirements.txt
├── data/
│   └── Cached input papers (gitignored)
│
└── outputs/
    └── Generated reports and evaluation artifacts (gitignored)
```

---

## Installation

### Requirements

* Python 3.10+
* macOS, Linux, or Windows

Clone the repository and install the dependencies:

```bash
git clone https://github.com/boumalaksiham/Final-Project-AI.git
cd Final-Project-AI

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Apple Silicon

BART inference is configured to use the CPU rather than the Apple MPS backend because some PyTorch/Transformers configurations can produce unreliable generation behavior on MPS.

The project handles this internally using:

```python
device=-1
```

No additional configuration is required.

---

## Usage

### Run the built-in synthetic demo

```bash
python main.py --demo
```

The demo is intended as a lightweight pipeline test and should use **synthetic test content**, rather than presenting generated test sentences as verbatim text from a published paper.

---

### Analyze an arXiv Paper

```bash
python main.py --arxiv 1706.03762
```

Examples:

```bash
# Attention Is All You Need
python main.py --arxiv 1706.03762

# Language Models are Few-Shot Learners (GPT-3)
python main.py --arxiv 2005.14165

# BERT
python main.py --arxiv 1810.04805

# Deep Residual Learning for Image Recognition
python main.py --arxiv 1512.03385
```

---

### Analyze a Local PDF

```bash
python main.py --pdf path/to/paper.pdf
```

---

### Analyze a PubMed Paper

```bash
python main.py --pubmed 33278872
```

The PubMed integration uses NCBI E-utilities to retrieve available article metadata and content.

---

## Topic-Relevance Scoring

A research topic can be supplied using the `--topic` argument:

```bash
python main.py \
  --arxiv 1706.03762 \
  --topic "attention mechanisms transformer NLP"
```

The relevance component computes a **lexical relevance score from 0–100** by comparing terminology in the paper with the supplied topic.

The score is intended as a triage heuristic rather than a semantic judgment of scientific importance.

Current interpretation:

|  Score | Interpretation                                 |
| -----: | ---------------------------------------------- |
| 70–100 | Strong lexical overlap with the supplied topic |
|  40–69 | Moderate overlap                               |
|  15–39 | Limited overlap                                |
|   0–14 | Little detected overlap                        |

---

# Evaluation

Evaluation was designed to test the system from multiple perspectives rather than relying on a single metric.

The project includes:

* quantitative evaluation against manually defined ground truth;
* comparison with a simpler baseline;
* component-level ablation;
* cross-paper citation analysis; and
* qualitative evaluation on previously unseen papers.

---

## 1. Quantitative Evaluation

Two papers currently have reproducible manually defined ground-truth annotations in the evaluation pipeline:

* *Attention Is All You Need*
* *Language Models are Few-Shot Learners (GPT-3)*

Evaluation measures include:

* ROUGE-1
* ROUGE-2
* dataset precision / recall / F1
* metric precision / recall / F1
* model extraction precision / recall / F1
* number of extracted references

### Results

| Paper                     |   ROUGE-1 |   ROUGE-2 | Dataset F1 | Metric F1 |  Model F1 | References |
| ------------------------- | --------: | --------: | ---------: | --------: | --------: | ---------: |
| Attention Is All You Need |     0.458 |     0.192 |      0.800 |     0.444 | **1.000** |         30 |
| GPT-3                     | **0.546** | **0.419** |  **1.000** | **1.000** | **1.000** |         30 |
| **Average**               | **0.502** | **0.306** |  **0.900** | **0.722** | **1.000** |         30 |

Run the evaluation with:

```bash
python evaluate.py 1706.03762
python evaluate.py 2005.14165
```

These results show that structured methodology extraction performs strongly on the annotated papers, while summary quality varies more substantially between papers.

---

## 2. Multi-Agent vs. Monolithic Baseline

The full pipeline was compared with a simpler baseline that combines BART summarization with basic keyword- and pattern-based extraction.

Because the baseline itself combines multiple deterministic and model-based operations, it is described here as a **monolithic baseline** rather than a "single-model" system.

| Metric     | Baseline | Multi-Agent Pipeline | Difference |
| ---------- | -------: | -------------------: | ---------: |
| ROUGE-1    |    0.185 |            **0.502** |     +0.317 |
| ROUGE-2    |    0.114 |            **0.306** |     +0.192 |
| Dataset F1 |    0.697 |            **0.900** |     +0.203 |
| Metric F1  |    0.444 |            **0.722** |     +0.278 |
| Model F1   |    0.554 |            **1.000** |     +0.446 |

Run the comparison:

```bash
python baseline.py 1706.03762
python baseline.py 2005.14165
```

The comparison suggests that separating scientific-paper analysis into specialized components improves extraction quality on the annotated evaluation set.

The experiment is small, so these results should be interpreted as **project-level evidence rather than a general benchmark claim**.

---

## 3. Ablation Study

The ablation experiment removes one specialized component at a time to determine which outputs depend on that component.

Example results on the GPT-3 paper:

| Removed Component      | Observed Effect                         |
| ---------------------- | --------------------------------------- |
| Summarization          | ROUGE-1 and ROUGE-2 fall to 0           |
| Citation analysis      | Extracted-reference count falls to 0    |
| Methodology extraction | Dataset, metric, and model F1 fall to 0 |
| Critical analysis      | Extracted limitation count falls to 0   |

Run:

```bash
python ablation.py 2005.14165
```

The experiment confirms that the pipeline is modular: each specialized component is responsible for a distinct dimension of the final report.

---

## 4. Cross-Paper Citation Analysis

The project can also compare references across multiple papers and identify shared citations.

Example:

```bash
python cross_paper_graph.py \
  --arxiv \
  1706.03762 \
  2005.14165 \
  1810.04805 \
  1512.03385
```

Using:

* *Attention Is All You Need*
* GPT-3
* BERT
* ResNet

the system identified:

* **15 shared references** across the four papers
* **4 direct citation relationships** among the analyzed papers
* multiple foundational works cited across more than one paper

This experiment demonstrates how the citation component can move beyond single-paper analysis toward lightweight literature-network exploration.

---

## 5. Generalization Check on Unseen Papers

To evaluate behavior beyond the papers used for quantitative development, the pipeline was also tested on five previously unseen papers:

* LoRA
* InstructGPT
* LLaMA
* LLMs Can Self-Improve
* Retrieval-Augmented Generation (RAG)

Reports were evaluated from 1–5 on:

* **Accuracy**
* **Completeness**
* **Triage Value**

### Results

| Paper                 | Accuracy | Completeness | Triage Value |  Average |
| --------------------- | -------: | -----------: | -----------: | -------: |
| LoRA                  |        3 |            2 |            2 |     2.33 |
| InstructGPT           |        2 |            2 |            2 |     2.00 |
| LLaMA                 |        3 |            3 |            3 |     3.00 |
| LLMs Can Self-Improve |        2 |            2 |            2 |     2.00 |
| RAG                   |        4 |            3 |            4 | **3.67** |
| **Average**           | **2.80** |     **2.40** |     **2.60** | **2.60** |

These results expose an important limitation of the current system rather than showing uniformly strong performance.

**The primary bottleneck was long-document summarization.**

The RAG paper received the highest triage-value score and was also the case where BART generated the most complete summary. When summarization failed or omitted important sections, the usefulness of the complete report decreased substantially.

This suggests that improving long-context summarization would likely produce the largest overall improvement to the system.

---

# Current Limitations

This project is a research prototype rather than a production literature-review system.

Important limitations include:

### Long-document summarization

`facebook/bart-large-cnn` was not designed for full-length scientific documents. Long papers must therefore be cleaned and divided into sections or chunks, and important context can be lost during generation.

### Heuristic methodology extraction

Dataset, metric, and architecture extraction combines spaCy with curated terminology.

This works well for recognized terminology but may miss:

* newly introduced datasets;
* uncommon evaluation metrics;
* domain-specific architectures;
* terminology not present in the curated vocabulary.

### Citation parsing

Reference extraction currently relies on patterns for common citation formats.

Unusual bibliography formatting may reduce extraction accuracy.

### Lexical relevance scoring

The relevance score measures keyword overlap rather than deep semantic similarity.

Two conceptually related papers using substantially different terminology may therefore receive a lower score than expected.

### Limited evaluation set

The quantitative ground-truth evaluation currently covers two papers.

The results demonstrate the behavior of this implementation but should not be interpreted as a broad benchmark of scientific-document analysis systems.

### No factuality verification

The summarization component does not independently verify whether every generated statement is supported by the source document.

Generated summaries should therefore be treated as research-triage aids rather than authoritative representations of a paper.

---

# Future Improvements

The experiments point toward several natural extensions:

* replace BART with a model designed for longer scientific documents;
* add semantic embeddings to improve relevance scoring;
* evaluate methodology extraction on a larger manually annotated corpus;
* introduce confidence scores for extracted entities;
* improve citation parsing across additional bibliography formats;
* detect relationships between papers beyond direct citation overlap;
* add factual-consistency evaluation for generated summaries;
* expand quantitative evaluation to additional research domains;
* expose the pipeline through a lightweight web interface or API.

---

# Output Files

Each execution can generate artifacts inside `outputs/`:

| File                             | Description                           |
| -------------------------------- | ------------------------------------- |
| `analysis_report.md`             | Structured research-triage report     |
| `citation_graph.png`             | Paper-to-reference citation graph     |
| `raw_outputs.json`               | Structured output from all components |
| `evaluation_results.json`        | Quantitative evaluation metrics       |
| `ablation_<id>.json`             | Ablation results                      |
| `baseline_comparison_<id>.json`  | Baseline comparison results           |
| `qualitative_eval.json`          | Saved qualitative evaluation          |
| `cross_paper_citation_graph.png` | Multi-paper citation visualization    |
| `cross_paper_graph_data.json`    | Shared-reference graph data           |

---

# Tech Stack

| Area                     | Technologies                    |
| ------------------------ | ------------------------------- |
| NLP / Summarization      | Hugging Face Transformers, BART |
| Named Entity Recognition | spaCy                           |
| Machine Learning Backend | PyTorch                         |
| Graph Analysis           | NetworkX                        |
| Visualization            | Matplotlib                      |
| PDF Processing           | PyMuPDF                         |
| Research APIs            | arXiv API, NCBI E-utilities     |
| Evaluation               | rouge-score, scikit-learn       |
| Language                 | Python                          |

---

# Selected Libraries and References

* Lewis et al. — **BART: Denoising Sequence-to-Sequence Pre-training**, ACL 2020
* Wolf et al. — **Transformers: State-of-the-Art Natural Language Processing**, EMNLP 2020
* Paszke et al. — **PyTorch: An Imperative Style, High-Performance Deep Learning Library**, NeurIPS 2019
* Hagberg et al. — **Exploring Network Structure, Dynamics, and Function using NetworkX**, SciPy 2008
* spaCy — Industrial-strength natural language processing in Python
* PyMuPDF — PDF parsing and text extraction
* arXiv API
* NCBI E-utilities

---

## Author

**Siham Boumalak**

M.S. Artificial Intelligence
Khoury College of Computer Sciences
Northeastern University

---

## Project Status

This repository represents a completed research-oriented NLP project and an experimental platform for studying **modular scientific-document analysis, information extraction, and research-paper triage**.
