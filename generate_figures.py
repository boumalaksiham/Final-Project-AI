"""
Figure Generator for Final Paper
Generates all matplotlib figures needed for the LaTeX paper.
Run this after generate_all_outputs.sh

Usage: python generate_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = "outputs"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ─── Figure 1: Quantitative Evaluation Bar Chart ────────────────────────────
def fig_quantitative_results():
    papers = ["Attention\nIs All You Need", "GPT-3", "BERT", "ResNet"]
    rouge1  = [0.458, 0.546, 0.374, 0.347]
    rouge2  = [0.192, 0.419, 0.211, 0.183]
    ds_f1   = [0.800, 1.000, 0.800, 0.800]
    mt_f1   = [0.444, 1.000, 0.444, 0.500]
    mo_f1   = [1.000, 1.000, 0.875, 0.333]

    x = np.arange(len(papers))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 5))
    bars1 = ax.bar(x - 2*width, rouge1, width, label='ROUGE-1',     color='#4A90D9', alpha=0.9)
    bars2 = ax.bar(x - 1*width, rouge2, width, label='ROUGE-2',     color='#7BC8F6', alpha=0.9)
    bars3 = ax.bar(x,           ds_f1,  width, label='Dataset F1',  color='#F4A460', alpha=0.9)
    bars4 = ax.bar(x + 1*width, mt_f1,  width, label='Metric F1',   color='#E8845A', alpha=0.9)
    bars5 = ax.bar(x + 2*width, mo_f1,  width, label='Model F1',    color='#A8D5A2', alpha=0.9)

    ax.set_xlabel('Paper', fontsize=12)
    ax.set_ylabel('Score (0–1)', fontsize=12)
    ax.set_title('Multi-Agent System: Quantitative Evaluation Results', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(papers, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_quantitative_results.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 2: Ablation Study Bar Chart ─────────────────────────────────────
def fig_ablation():
    configs = ['Full\nSystem', 'No\nSummarization', 'No\nCitation', 'No\nMethodology', 'No\nCritical']
    rouge1 = [0.546, 0.000, 0.546, 0.546, 0.546]
    ds_f1  = [1.000, 1.000, 1.000, 0.000, 1.000]
    refs   = [30,    30,    0,     30,    30]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width, rouge1, width, label='ROUGE-1',    color='#4A90D9', alpha=0.9)
    bars2 = ax1.bar(x,         ds_f1,  width, label='Dataset F1', color='#F4A460', alpha=0.9)
    bars3 = ax2.bar(x + width, refs,   width, label='References', color='#A8D5A2', alpha=0.9)

    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Score (0–1)', fontsize=12, color='black')
    ax2.set_ylabel('References Extracted', fontsize=12, color='#2E7D32')
    ax1.set_title('Ablation Study — GPT-3 Paper (arXiv: 2005.14165)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=10)
    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0, 45)
    ax2.tick_params(axis='y', labelcolor='#2E7D32')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_ablation_study.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 3: Baseline Comparison Bar Chart ────────────────────────────────
def fig_baseline():
    metrics = ['ROUGE-1', 'ROUGE-2', 'Dataset F1', 'Metric F1', 'Model F1']
    baseline    = [0.185, 0.114, 0.697, 0.444, 0.554]
    multiagent  = [0.502, 0.306, 0.900, 0.722, 1.000]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, baseline,   width, label='Single-Model Baseline', color='#E8845A', alpha=0.9)
    bars2 = ax.bar(x + width/2, multiagent, width, label='Multi-Agent System',    color='#4A90D9', alpha=0.9)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score (0–1)', fontsize=12)
    ax.set_title('Multi-Agent System vs. Single-Model BART Baseline', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

    # Draw delta arrows
    for i, (b, m) in enumerate(zip(baseline, multiagent)):
        delta = m - b
        ax.annotate(f'+{delta:.3f}',
            xy=(i + width/2, m + 0.03),
            ha='center', fontsize=8, color='#1A5276', fontweight='bold')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_baseline_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 4: Qualitative Evaluation Bar Chart ─────────────────────────────
def fig_qualitative():
    papers = ['LoRA', 'Chain-of-\nThought', 'LLaMA', 'ReAct', 'RAG']
    accuracy    = [3, 2, 3, 2, 4]
    completeness= [2, 2, 3, 2, 3]
    triage      = [2, 2, 3, 2, 4]

    x = np.arange(len(papers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width, accuracy,     width, label='Accuracy (Developer)',     color='#4A90D9', alpha=0.9)
    bars2 = ax.bar(x,         completeness, width, label='Completeness (Developer)', color='#F4A460', alpha=0.9)
    bars3 = ax.bar(x + width, triage,       width, label='Triage Value (User)',      color='#A8D5A2', alpha=0.9)

    ax.set_xlabel('Paper', fontsize=12)
    ax.set_ylabel('Score (1–5)', fontsize=12)
    ax.set_title('Qualitative Evaluation Results — Five Unseen Papers', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(papers, fontsize=11)
    ax.set_ylim(0, 6)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Midpoint (3)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{int(h)}',
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_qualitative_eval.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─── Figure 5: System Architecture Diagram ──────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    # Box style
    box_style = dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB', edgecolor='#2E86C1', linewidth=2)
    agent_style = dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5', edgecolor='#1E8449', linewidth=1.5)
    output_style = dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', edgecolor='#B7950B', linewidth=1.5)

    # Input
    ax.text(0.5, 0.93, 'INPUT: PDF / arXiv ID / PubMed ID', ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=box_style, transform=ax.transAxes)

    # Pre-processing
    ax.text(0.5, 0.80, 'paper_loader.py\n(PyMuPDF + arXiv API + NCBI API)',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F4ECF7', edgecolor='#7D3C98', linewidth=1.5),
            transform=ax.transAxes)

    # Agents
    agents = [
        (0.12, 0.60, 'Agent 1\nSummarization\n(BART)'),
        (0.30, 0.60, 'Agent 2\nCitation\n(NetworkX)'),
        (0.50, 0.60, 'Agent 3\nMethodology\n(spaCy NER)'),
        (0.70, 0.60, 'Agent 4\nCritical\n(Patterns)'),
        (0.88, 0.60, 'Agent 5\nRelevance\n(TL;DR)'),
    ]
    for x_pos, y_pos, label in agents:
        ax.text(x_pos, y_pos, label, ha='center', va='center', fontsize=9,
                bbox=agent_style, transform=ax.transAxes)

    # Coordinator
    ax.text(0.5, 0.38, 'Agent 6: Coordinator\n(Synthesizes all outputs)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEDEC', edgecolor='#C0392B', linewidth=2),
            transform=ax.transAxes)

    # Outputs
    outputs = [
        (0.20, 0.18, 'Markdown\nReport'),
        (0.40, 0.18, 'Citation\nGraph (PNG)'),
        (0.60, 0.18, 'Raw JSON\nOutputs'),
        (0.80, 0.18, 'TL;DR +\nRelevance Score'),
    ]
    for x_pos, y_pos, label in outputs:
        ax.text(x_pos, y_pos, label, ha='center', va='center', fontsize=9,
                bbox=output_style, transform=ax.transAxes)

    # Arrows (approximate)
    arrow_props = dict(arrowstyle='->', color='#566573', lw=1.5)
    ax.annotate('', xy=(0.5, 0.84), xytext=(0.5, 0.88),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=arrow_props)
    for x_pos, _, _ in agents:
        ax.annotate('', xy=(x_pos, 0.67), xytext=(0.5, 0.76),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#566573', lw=1))
    for x_pos, _, _ in agents:
        ax.annotate('', xy=(0.5, 0.44), xytext=(x_pos, 0.53),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#566573', lw=1))
    for x_pos, _, _ in outputs:
        ax.annotate('', xy=(x_pos, 0.24), xytext=(0.5, 0.33),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#566573', lw=1))

    ax.set_title('Multi-Agent Scientific Paper Analysis System — Architecture', 
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_architecture.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating all paper figures...\n")
    fig_quantitative_results()
    fig_ablation()
    fig_baseline()
    fig_qualitative()
    fig_architecture()
    print("\nAll figures saved to outputs/")
    print("\nNow run generate_all_outputs.sh to generate the citation graphs for each paper.")