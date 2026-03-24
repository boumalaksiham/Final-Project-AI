"""
Citation Analysis Agent
Extracts references from paper text and builds a knowledge graph.
"""

import re
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


class CitationAnalysisAgent:
    def __init__(self):
        print("[CitationAnalysisAgent] Initialized.")

    def _find_ref_section(self, text):
        """Find the actual references section, skipping TOC entries."""
        lower = text.lower()
        for m in re.finditer(r"references\s*\n", lower):
            snippet = text[m.start():m.start() + 50]
            lines = snippet.split("\n")
            second = lines[1].strip() if len(lines) > 1 else ""
            if second.isdigit():
                continue
            return text[m.start():]
        idx = lower.find("references")
        if idx != -1:
            return text[idx:]
        return text[-5000:]

    def _extract_references(self, text):
        ref_section = self._find_ref_section(text)

        # Pattern 1: [1] numbered
        numbered = re.findall(r"(?:\[\d+\]|\d+\.)\s+([A-Z][^\n]{20,})", ref_section)
        if len(numbered) >= 10:
            return numbered[:30]

        # Pattern 2: [ADG+16] abbreviated keys
        entries = re.split(r"(?=\[[A-Z][A-Za-z+\d]{2,8}\])", ref_section)
        abbrev = []
        for entry in entries:
            entry = entry.strip()
            if not re.match(r"\[[A-Z][A-Za-z+\d]{2,8}\]", entry):
                continue
            if len(entry) < 30:
                continue
            clean = re.sub(r"\n\s*", " ", entry)
            clean = re.sub(r"^\[[A-Z][A-Za-z+\d]{2,8}\]\s*", "", clean).strip()
            if len(clean) > 20:
                abbrev.append(clean)
        if len(abbrev) >= 3:
            return abbrev[:30]

        # Pattern 3: Author (year) style
        matches = []
        for line in ref_section.split("\n"):
            line = line.strip()
            if len(line) < 40:
                continue
            if re.match(r"[A-Z][a-z]+", line) and re.search(r"\b(19|20)\d{2}\b", line):
                matches.append(line)
        if matches:
            return matches[:30]

        # Fallback
        fallback = []
        for line in ref_section.split("\n"):
            line = line.strip()
            if len(line) > 40 and any(c.isupper() for c in line[:20]):
                fallback.append(line)
        return fallback[:30]

    def _extract_paper_title(self, text):
        for line in text.split("\n")[:3]:
            if line.startswith("ARXIV_TITLE:"):
                return line.replace("ARXIV_TITLE:", "").strip()
        SKIP = [
            "permission", "copyright", "license", "rights reserved",
            "preprint", "doi", "http", "www", "©", "under review",
            "submitted", "proceedings", "workshop", "grant",
            "attribution", "hereby", "journal", "scholarly"
        ]
        for line in [l.strip() for l in text.split("\n") if l.strip()][:20]:
            lower = line.lower()
            if not (10 < len(line) < 150):
                continue
            if any(s in lower for s in SKIP):
                continue
            if sum(c.isalpha() or c.isspace() for c in line) / len(line) > 0.7:
                return line
        return "Current Paper"

    def build_graph(self, paper_title, references):
        G = nx.DiGraph()
        G.add_node(paper_title, node_type="main")
        for ref in references:
            short_ref = ref[:80]
            G.add_node(short_ref, node_type="reference")
            G.add_edge(paper_title, short_ref)
        return G

    def save_graph_image(self, G, output_path):
        plt.figure(figsize=(14, 8))
        pos = nx.spring_layout(G, seed=42, k=2)
        main_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "main"]
        ref_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "reference"]
        nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, node_color="#4A90D9", node_size=800)
        nx.draw_networkx_nodes(G, pos, nodelist=ref_nodes, node_color="#A8D5A2", node_size=300)
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
        main_labels = {n: n[:40] for n in main_nodes}
        nx.draw_networkx_labels(G, pos, labels=main_labels, font_size=9, font_weight="bold")
        plt.title("Citation Knowledge Graph", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[CitationAnalysisAgent] Graph saved to {output_path}")

    def run(self, text, output_dir="outputs"):
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