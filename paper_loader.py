"""
Paper Loader Utility
Handles PDF extraction and arXiv paper fetching.
"""

import re
import urllib.request
from pathlib import Path


def load_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")


def load_arxiv(arxiv_id: str, save_dir: str = "data") -> tuple[str, str]:
    """
    Download and extract text from an arXiv paper.
    arxiv_id: e.g. '1706.03762' or 'https://arxiv.org/abs/1706.03762'
    Returns: (text, pdf_path)
    """
    # Normalize ID
    arxiv_id = arxiv_id.strip()
    arxiv_id = re.sub(r"https?://arxiv\.org/(abs|pdf)/", "", arxiv_id)
    arxiv_id = arxiv_id.replace(".pdf", "")

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    save_path = str(Path(save_dir) / f"{arxiv_id.replace('/', '_')}.pdf")

    Path(save_dir).mkdir(exist_ok=True)

    if not Path(save_path).exists():
        print(f"[Loader] Downloading arXiv:{arxiv_id}...")
        urllib.request.urlretrieve(pdf_url, save_path)
        print(f"[Loader] Saved to {save_path}")
    else:
        print(f"[Loader] Using cached file: {save_path}")

    text = load_pdf(save_path)
    return text, save_path


def load_text(text_path: str) -> str:
    """Load plain text file."""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()