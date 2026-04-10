"""
Cross-Paper Citation Graph
Builds a knowledge graph showing relationships BETWEEN multiple papers
based on shared references and direct citation relationships.

Usage:
    python cross_paper_graph.py --arxiv 1706.03762 2005.14165 1810.04805 1512.03385
    python cross_paper_graph.py --pdf paper1.pdf paper2.pdf
"""

import argparse
import re
import json
from pathlib import Path
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from paper_loader import load_arxiv, load_pdf
from agents.citation_agent import CitationAnalysisAgent


OUTPUT_DIR = "outputs"

# Common words to exclude from author fingerprints
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was",
    "has", "have", "not", "but", "they", "their", "which", "also",
    "neural", "learning", "deep", "language", "model", "models",
    "network", "networks", "using", "based", "large", "new", "via",
    "machine", "natural", "processing", "training", "pre", "on", "in",
    "of", "a", "an", "to", "is", "it", "its", "at", "by", "as",
}


def extract_author_fingerprint(ref: str) -> frozenset:
    """
    Extract a set of likely author last names from a reference string.
    Two references are considered the same if they share 2+ author tokens.
    This works across citation formats (numbered, [ADG+16], author-year).
    """
    # Remove the citation key prefix if present (e.g. [ADG+16])
    ref = re.sub(r"^\[[A-Z][A-Za-z+\d]{2,8}\]\s*", "", ref)

    # Extract capitalized words that look like last names
    # (capital letter, followed by lowercase, length 3-20)
    tokens = re.findall(r"\b[A-Z][a-z]{2,19}\b", ref)
    tokens = [t.lower() for t in tokens if t.lower() not in STOP_WORDS]

    # Also extract year if present
    year_match = re.search(r"\b(19|20)\d{2}\b", ref)
    if year_match:
        tokens.append(year_match.group())

    return frozenset(tokens)


def refs_match(fp1: frozenset, fp2: frozenset) -> bool:
    """
    Two references match if they share at least 2 author-like tokens.
    This is robust to formatting differences.
    """
    if not fp1 or not fp2:
        return False
    shared = fp1 & fp2
    # Need at least 2 shared tokens to avoid false positives
    return len(shared) >= 2


def load_paper(source: str, is_arxiv: bool) -> tuple:
    """Load a paper and return (text, title)."""
    if is_arxiv:
        text, _ = load_arxiv(source)
        for line in text.split("\n")[:3]:
            if line.startswith("ARXIV_TITLE:"):
                title = line.replace("ARXIV_TITLE:", "").strip()
                return text, title
        return text, source
    else:
        text = load_pdf(source)
        return text, Path(source).stem


def build_cross_paper_graph(papers: list) -> tuple:
    """
    Build a cross-paper citation graph using author fingerprint matching.

    Returns: (G, shared_refs_info)
    """
    agent = CitationAnalysisAgent()
    G = nx.DiGraph()

    # Extract refs for each paper
    paper_data = []
    for text, title in papers:
        refs = agent._extract_references(text)
        fingerprints = [(r, extract_author_fingerprint(r)) for r in refs]
        paper_data.append((title, refs, fingerprints))
        G.add_node(title, node_type="main_paper")
        print(f"  [{title[:50]}] — {len(refs)} references")

    # Find shared references across papers
    # For each pair of papers, find refs that match
    shared_refs_info = []  # list of {display, papers_citing}

    n = len(paper_data)
    for i in range(n):
        title_i, refs_i, fps_i = paper_data[i]
        for j in range(i + 1, n):
            title_j, refs_j, fps_j = paper_data[j]
            # Find matching refs between paper i and paper j
            for ref_i, fp_i in fps_i:
                for ref_j, fp_j in fps_j:
                    if refs_match(fp_i, fp_j):
                        # Use the longer ref string as display
                        display = ref_i if len(ref_i) >= len(ref_j) else ref_j
                        display = display[:80]
                        shared_refs_info.append({
                            "display": display,
                            "paper_a": title_i,
                            "paper_b": title_j,
                        })
                        break  # one match per ref_i is enough

    # Deduplicate shared refs
    # Group by display text
    seen_displays = {}
    for item in shared_refs_info:
        key = item["display"][:50]
        if key not in seen_displays:
            seen_displays[key] = {"display": item["display"], "papers": set()}
        seen_displays[key]["papers"].add(item["paper_a"])
        seen_displays[key]["papers"].add(item["paper_b"])

    # Add shared ref nodes and edges to graph
    for key, info in seen_displays.items():
        display = info["display"]
        G.add_node(display, node_type="shared_ref", cite_count=len(info["papers"]))
        for paper_title in info["papers"]:
            G.add_edge(paper_title, display)

    # Check for direct paper-to-paper citations
    for i in range(n):
        title_i, _, _ = paper_data[i]
        text_i = papers[i][0].lower()
        for j in range(n):
            if i == j:
                continue
            title_j = paper_data[j][0]
            # Check if 3+ words from title_j appear in text_i
            words = [w for w in title_j.lower().split() if len(w) > 4][:5]
            if sum(1 for w in words if w in text_i) >= 3:
                G.add_edge(title_i, title_j, edge_type="direct_citation")
                print(f"  Direct citation detected: [{title_i[:35]}] → [{title_j[:35]}]")

    return G, seen_displays


def visualize(G: nx.DiGraph, output_path: str):
    """Visualize the cross-paper graph."""
    main_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "main_paper"]
    ref_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "shared_ref"]

    print(f"\n  [CrossPaperGraph] {len(main_nodes)} papers, "
          f"{len(ref_nodes)} shared references, {G.number_of_edges()} edges")

    fig, ax = plt.subplots(figsize=(20, 14))

    if G.number_of_nodes() == 0 or len(ref_nodes) == 0:
        ax.text(0.5, 0.5, "No shared references found between papers.",
                ha="center", va="center", fontsize=14)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Draw shared ref nodes
    nx.draw_networkx_nodes(G, pos, nodelist=ref_nodes,
                           node_color="#F4A460", node_size=150, alpha=0.75, ax=ax)
    # Draw main paper nodes
    nx.draw_networkx_nodes(G, pos, nodelist=main_nodes,
                           node_color="#4A90D9", node_size=1500, alpha=0.95, ax=ax)
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=True,
                           arrowsize=8, edge_color="#888888", ax=ax)
    # Labels for main papers
    main_labels = {}
    for n in main_nodes:
        words = n.split()
        label = " ".join(words[:4])
        if len(words) > 4:
            label += "\n" + " ".join(words[4:7])
        main_labels[n] = label
    nx.draw_networkx_labels(G, pos, labels=main_labels,
                            font_size=6, font_weight="bold",
                            font_color="white", ax=ax)
    # Short labels for shared refs
    ref_labels = {n: n[:30] for n in ref_nodes}
    nx.draw_networkx_labels(G, pos, labels=ref_labels,
                            font_size=4, font_color="#333333", ax=ax)

    legend_elements = [
        mpatches.Patch(color="#4A90D9", label="Analyzed Paper"),
        mpatches.Patch(color="#F4A460", label="Shared Reference (cited by 2+ papers)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title(
        f"Cross-Paper Citation Knowledge Graph\n"
        f"{len(main_nodes)} papers | {len(ref_nodes)} shared references",
        fontsize=13
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [CrossPaperGraph] Graph saved to {output_path}")


def run(sources: list, is_arxiv: bool):
    print("\n" + "=" * 60)
    print("  Cross-Paper Citation Graph Builder")
    print("=" * 60 + "\n")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    papers = []
    for source in sources:
        print(f"Loading: {source}")
        text, title = load_paper(source, is_arxiv)
        papers.append((text, title))

    print(f"\nBuilding cross-paper graph for {len(papers)} papers...\n")
    G, seen_displays = build_cross_paper_graph(papers)

    output_path = str(Path(OUTPUT_DIR) / "cross_paper_citation_graph.png")
    visualize(G, output_path)

    # Summary
    shared_count = len(seen_displays)
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Papers analyzed:   {len(papers)}")
    print(f"  Shared references: {shared_count}")
    if seen_displays:
        print(f"\n  Sample shared references:")
        for i, (key, info) in enumerate(list(seen_displays.items())[:8]):
            print(f"    [{len(info['papers'])} papers] {info['display'][:70]}")

    # Save JSON
    out = {
        "papers_analyzed": [t for _, t in papers],
        "shared_references_count": shared_count,
        "shared_references": [
            {"reference": info["display"], "cited_by": list(info["papers"])}
            for info in list(seen_displays.values())[:30]
        ],
    }
    json_path = str(Path(OUTPUT_DIR) / "cross_paper_graph_data.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Data saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-paper citation graph")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arxiv", nargs="+", help="arXiv IDs")
    group.add_argument("--pdf",   nargs="+", help="PDF file paths")
    args = parser.parse_args()
    run(args.arxiv if args.arxiv else args.pdf, is_arxiv=bool(args.arxiv))