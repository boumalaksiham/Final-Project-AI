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


def load_pubmed(pmcid: str, save_dir: str = "data") -> tuple:
    """
    Download and extract text from a PubMed Central open-access paper.
    pmcid: PubMed Central ID, e.g. 'PMC7010990' or just '7010990'
    Returns: (text, pdf_path)

    Note: Only Open Access papers available via PMC are supported.
    PMC open access subset: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
    """
    # Normalize ID
    pmcid = pmcid.strip().upper()
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid

    # Fetch paper metadata from NCBI E-utilities API
    meta_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pmc&term={pmcid}&retmode=json"
    )

    Path(save_dir).mkdir(exist_ok=True)
    save_path = str(Path(save_dir) / f"{pmcid}.pdf")

    # Try to download the PDF via PMC FTP
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

    if not Path(save_path).exists():
        print(f"[Loader] Downloading PubMed Central: {pmcid}...")
        try:
            urllib.request.urlretrieve(pdf_url, save_path)
            print(f"[Loader] Saved to {save_path}")
        except Exception as e:
            raise RuntimeError(
                f"Could not download {pmcid} from PubMed Central. "
                f"Only open-access papers are available. Error: {e}"
            )
    else:
        print(f"[Loader] Using cached file: {save_path}")

    # Fetch title from NCBI API
    try:
        title_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pmc&id={pmcid.replace('PMC','')}&retmode=json"
        )
        with urllib.request.urlopen(title_url, timeout=5) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
        uid = list(meta.get("result", {}).keys() - {"uids"})[0]
        title = meta["result"][uid].get("title", "")
        if title:
            print(f"[Loader] Title: {title}")
    except Exception:
        title = pmcid

    text = load_pdf(save_path)
    if title:
        text = f"ARXIV_TITLE: {title}\n\n" + text

    return text, save_path


def load_pubmed(pmid: str, save_dir: str = "data") -> tuple[str, str]:
    """
    Fetch a PubMed article by its PubMed ID (PMID) using the NCBI E-utilities API.
    Returns the abstract text and the paper title.

    Note: PubMed articles are biomedical papers. Unlike arXiv, PubMed does not
    provide free full-text PDFs for all articles. This function fetches the
    abstract and metadata, which is sufficient for summarization and methodology
    extraction on most papers.

    pmid: PubMed ID, e.g. '33278872'
    Returns: (text, title)
    """
    import urllib.request
    import json as _json

    pmid = pmid.strip()

    # Step 1: Fetch metadata (title + abstract) via NCBI E-utilities
    fetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={pmid}&rettype=abstract&retmode=json"
    )

    try:
        with urllib.request.urlopen(fetch_url, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch PubMed article {pmid}: {e}")

    # Step 2: Parse title and abstract from response
    # NCBI returns structured XML-like text in abstract mode
    title = f"PubMed:{pmid}"
    abstract = ""

    # Try to extract title and abstract with simple pattern matching
    title_match = re.search(r'"Title":\s*"([^"]+)"', raw)
    if title_match:
        title = title_match.group(1).strip()

    abstract_match = re.search(r'"AbstractText":\s*"([^"]+)"', raw)
    if abstract_match:
        abstract = abstract_match.group(1).strip()

    # If JSON parsing failed, try the text mode
    if not abstract:
        text_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
        )
        try:
            with urllib.request.urlopen(text_url, timeout=10) as resp:
                raw_text = resp.read().decode("utf-8")
            # The text format starts with title, then authors, then abstract
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            if lines:
                title = lines[0]
            # Abstract typically starts after "Abstract" header
            abstract_idx = next(
                (i for i, l in enumerate(lines) if l.lower().startswith("abstract")),
                None
            )
            if abstract_idx is not None:
                abstract = " ".join(lines[abstract_idx + 1:abstract_idx + 20])
            elif len(lines) > 3:
                abstract = " ".join(lines[3:15])
        except Exception as e:
            raise RuntimeError(f"Failed to fetch PubMed text for {pmid}: {e}")

    if not abstract:
        raise RuntimeError(
            f"Could not extract abstract for PubMed:{pmid}. "
            "The article may require institutional access."
        )

    # Prepend title tag so downstream agents can find it
    text = f"ARXIV_TITLE: {title}\n\nAbstract\n{abstract}"
    print(f"[Loader] PubMed title: {title}")
    print(f"[Loader] Abstract length: {len(abstract)} chars")

    return text, title