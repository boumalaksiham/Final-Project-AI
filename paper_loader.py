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
    Returns: (text, pdf_path) where text is prepended with ARXIV_TITLE: line
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

    # Fetch title from arXiv API and prepend it so agents can find it reliably
    try:
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        with urllib.request.urlopen(api_url, timeout=5) as resp:
            api_text = resp.read().decode("utf-8")
        # The real title is inside <entry><title>, skip the top-level feed <title>
        entry_match = re.search(r"<entry>.*?<title>([^<]+)</title>", api_text, re.DOTALL)
        if entry_match:
            title = entry_match.group(1).strip().replace("\n", " ")
            text = f"ARXIV_TITLE: {title}\n\n" + text
            print(f"[Loader] Title: {title}")
    except Exception:
        pass  # silently skip if API call fails

    return text, save_path


def load_text(text_path: str) -> str:
    """Load plain text file."""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()