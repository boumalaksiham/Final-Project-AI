"""
Citation Analysis Agent
Extracts references from paper text and builds a knowledge graph.
"""

import re
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


class CitationAnalysisAgent:
    def __init__(self):
        print("[CitationAnalysisAgent] Initialized.")

    def _extract_references(self, text: str) -> list[str]:
        """
        Extract references section and parse individual citations.
        Handles common formats: [1] Author..., numbered lists, etc.
        """
        # Try to isolate the references section
        ref_section = ""
        markers = ["references", "bibliography", "works cited"]
        lower = text.lower()
        for marker in markers:
            idx = lower.rfind(marker)
            if idx != -1:
                ref_section = text[idx:]
                break

        if not ref_section:
            ref_section = text[-3000:]  # fallback: use last portion

        # Extract numbered references [1] or 1.
        pattern = r"(?:\[\d+\]|\d+\.)\s+([A-Z][^\n]{20,})"
        matches = re.findall(pattern, ref_section)

        # Fallback: grab lines that look like citations (author-year style)
        if len(matches) < 3:
            lines = ref_section.split("\n")
            matches = [
                line.strip()
                for line in lines
                if len(line.strip()) > 40 and any(c.isupper() for c in line[:20])
            ]

        return matches[:30]  # cap at 30 refs

    def _extract_paper_title(self, text: str) -> str:
        """Attempt to extract the paper title from the first few lines."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines[:10]:
            if 10 < len(line) < 150:
                return line
        return "Current Paper"

    def build_graph(self, paper_title: str, references: list[str]) -> nx.DiGraph:
        """Build a directed citation graph."""
        G = nx.DiGraph()
        G.add_node(paper_title, node_type="main")
        for ref in references:
            short_ref = ref[:80]
            G.add_node(short_ref, node_type="reference")
            G.add_edge(paper_title, short_ref)
        return G

    def save_graph_image(self, G: nx.DiGraph, output_path: str):
        """Save a visualization of the citation graph."""
        plt.figure(figsize=(14, 8))
        pos = nx.spring_layout(G, seed=42, k=2)
        main_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "main"]
        ref_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "reference"]

        nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, node_color="#4A90D9", node_size=800)
        nx.draw_networkx_nodes(G, pos, nodelist=ref_nodes, node_color="#A8D5A2", node_size=300)
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)

        # Only label the main node
        main_labels = {n: n[:40] for n in main_nodes}
        nx.draw_networkx_labels(G, pos, labels=main_labels, font_size=9, font_weight="bold")

        plt.title("Citation Knowledge Graph", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[CitationAnalysisAgent] Graph saved to {output_path}")

    def run(self, text: str, output_dir: str = "outputs") -> dict:
        print("[CitationAnalysisAgent] Extracting citations...")
        references = self._extract_references(text)
        paper_title = self._extract_paper_title(text)
        G = self.build_graph(paper_title, references)

        graph_path = str(Path(output_dir) / "citation_graph.png")
        self.save_graph_image(G, graph_path)

        return {
            "paper_title": paper_title,
            "num_references": len(references),
            "references": references,
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
            "graph_image": graph_path,
        }