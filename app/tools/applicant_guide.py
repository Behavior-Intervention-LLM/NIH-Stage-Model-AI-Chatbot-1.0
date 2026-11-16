"""Versioned, page-citable local applicant guide with automatic ingestion.

Only the configured reference PDF is indexed. Content-addressed generations
are published atomically; a failed extraction cannot publish a partial index.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import re
import tempfile
from threading import Lock

import pdfplumber

from app.config import settings
from app.core.types import Citation, ToolResult
from app.tools.base import BaseTool
from app.tools.document_loader import DocumentChunk
from app.tools.vector_store import SimpleVectorStore

INDEX_VERSION = 2
TOOL_NAME = "applicant_guide_tool"


def _page_chunks(text: str, source: str):
    """Overlap complete lines so excerpts never start halfway through a word."""
    buffer = []
    index = 0
    for line in text.splitlines():
        if buffer and len("\n".join(buffer + [line])) > 1800:
            yield DocumentChunk("\n".join(buffer), source, index)
            index += 1
            overlap = []
            for previous in reversed(buffer):
                if len("\n".join([previous] + overlap)) > 200:
                    break
                overlap.insert(0, previous)
            buffer = overlap
        buffer.append(line)
    if "\n".join(buffer).strip():
        yield DocumentChunk("\n".join(buffer), source, index)


def _program_sections(query: str):
    """Retrieval hints only, from G.130; never an eligibility determination.

    Unknown or ambiguous codes retain all programs. K12 is institutional
    training, not an individual career award. Comparisons retain both groups.
    """
    patterns = {
        "research": r"\b(?:R01|R03|R15|R21|R34|U01|research grant)\b",
        "career": r"\b(?:K01|K02|K05|K07|K08|K18|K22|K23|K24|K25|K26|K43|K76|K99|career development award)\b",
        "training": r"\b(?:T15|T32|T34|T35|T36|T37|T90|D43|D71|K12|U2R|institutional training)\b",
        "fellowship": r"\b(?:F30|F31|F32|F33|fellowship)\b",
        "business": r"\b(?:R41|R42|R43|R44|SBIR|STTR)\b",
    }
    codes = re.findall(r"\b[A-Z]\d{2}\b|\bU2R\b", query, re.I)
    if any(not any(re.fullmatch(pattern, code, re.I) for pattern in patterns.values()) for code in codes):
        return None
    groups = [name for name, pattern in patterns.items() if re.search(pattern, query, re.I)]
    if len(groups) != 1:
        return None
    return {"research": {"G.400"}, "career": {"G.410"},
            "training": {"G.330", "G.340", "G.420"}, "fellowship": {"G.430"},
            "business": {"G.400", "G.440"}}[groups[0]]


def extract_guide(path: Path):
    """Keep PDF page numbers and enclosing G.xxx sections out of TF-IDF text."""
    chunks = []
    section = "Front matter"
    with pdfplumber.open(path) as pdf:
        cover = pdf.pages[0].extract_text(x_tolerance=1) or ""
        version = re.search(r"FORMS\s+VERSION\s+(\w+)\s+SERIES", cover, re.I)
        release = re.search(r"Released:\s*([^\n]+)", cover, re.I)
        edition = {
            "forms_version": version.group(1) if version else "unknown",
            "released": release.group(1).strip() if release else "unknown",
            "page_count": len(pdf.pages),
        }
        for page_number, page in enumerate(pdf.pages, 1):
            # The default tolerance joins words in this PDF ("SpecificAims").
            text = page.extract_text(x_tolerance=1) or ""
            heading = re.match(r"(G\.\d{3}\s*[-–].+)", text)
            if heading:
                section = heading.group(1).strip()
            for chunk in _page_chunks(text, path.name):
                chunk.metadata.update(edition, page=page_number, section=section,
                                      reference_type="applicant_guide")
                chunks.append(chunk)
            page.flush_cache()
    if not chunks:
        raise ValueError("Applicant guide contains no extractable text; OCR is required.")
    return chunks, edition


def ensure_guide_index(pdf_path: str | Path, index_root: str | Path) -> Path:
    """Reuse an unchanged PDF; replace the active generation when bytes change."""
    source = Path(pdf_path).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    root = Path(index_root)
    root.mkdir(parents=True, exist_ok=True)
    generation = root / f"v{INDEX_VERSION}-{digest}"
    # Both API and Streamlit can start on the same machine at the same time.
    with (root / ".ingest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (generation / "manifest.json").exists():
            return generation
        with tempfile.TemporaryDirectory(prefix=".building-", dir=root) as temp:
            staging = Path(temp) / "index"
            chunks, edition = extract_guide(source)
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Applicant guide changed during extraction; retry ingestion.")
            store = SimpleVectorStore(str(staging))
            store.add_chunks(chunks, replace=True)
            manifest = {"sha256": digest, "index_version": INDEX_VERSION,
                        "source": source.name, "chunk_count": len(store.chunks), **edition}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
            staging.rename(generation)
    return generation


class ApplicantGuideTool(BaseTool):
    def __init__(self, pdf_path=None, index_root=None):
        super().__init__(TOOL_NAME, "Search the local NIH applicant guide with page references")
        self.pdf_path = pdf_path or settings.APPLICANT_GUIDE_PATH
        self.index_root = index_root or settings.APPLICANT_GUIDE_INDEX_PATH
        self._generation = None
        self._store = None
        self._lock = Lock()

    def run(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        try:
            with self._lock:
                generation = ensure_guide_index(self.pdf_path, self.index_root)
                if generation != self._generation:
                    self._store = SimpleVectorStore(str(generation))
                    self._generation = generation
                store = self._store
            citations = []
            program_sections = _program_sections(query)
            dedicated = {"G.330", "G.340", "G.400", "G.410", "G.420", "G.430", "G.440"}
            # Activity codes choose applicability above; leaving a rare code in
            # TF-IDF makes a program listing outrank the requested form itself.
            search_query = query
            if program_sections:
                search_query = re.sub(r"\b[A-Z]\d{2}\b|\bU2R\b", "", query, flags=re.I).strip() or query
            # Search all passages before filtering so another program's repeated
            # wording cannot crowd the applicable form out of the candidate pool.
            hits = store.search(search_query, top_k=len(store.chunks))
            for hit in hits:
                meta = dict(hit["doc"]["metadata"])
                section_code = meta.get("section", "").split(" ")[0]
                if program_sections and section_code in dedicated and section_code not in program_sections:
                    continue
                meta.update(semantic_score=hit["score"],
                            term_coverage=store.term_coverage(search_query, hit["content"]))
                citations.append(Citation(source=hit["source"], passage=hit["content"],
                                          relevance_score=hit["score"], metadata=meta))
                if len(citations) >= max(1, min(top_k, 10)):
                    break
            return ToolResult(text="Applicant guide passages retrieved.", citations=citations,
                              structured={"generation": generation.name}, success=True)
        except Exception as exc:
            return ToolResult(text="Applicant guide unavailable.", success=False, error=str(exc))
