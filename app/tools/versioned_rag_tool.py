"""
Version-aware RAG tool:
- retrieves semantically relevant chunks
- compares document recency/version signals
- prioritizes newer/revised sources in ranking
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.core.types import Citation, ToolResult
from app.tools.base import BaseTool
from app.tools.vector_store import SimpleVectorStore


class VersionedRAGTool(BaseTool):
    """Vector retrieval with recency/version-aware reranking."""

    def __init__(self, vector_store: Optional[SimpleVectorStore] = None):
        super().__init__("versioned_rag_tool", "Version-aware RAG with recency comparison")
        self.vector_store = vector_store or SimpleVectorStore(
            storage_path=getattr(settings, "VECTOR_STORE_PATH", "data/vector_store")
        )
        self.documents_dir = Path(getattr(settings, "DOCUMENTS_DIR", "data/documents"))

    @staticmethod
    def _extract_year_from_name(source: str) -> Optional[int]:
        # Prefer explicit 4-digit year in filename.
        m = re.search(r"(19\d{2}|20\d{2})", source)
        if m:
            return int(m.group(1))

        # Support compact date patterns like 1.12.25 or 01-12-2025.
        d = re.search(r"(\d{1,2})[._-](\d{1,2})[._-](\d{2,4})", source)
        if not d:
            return None
        y = int(d.group(3))
        if y < 100:
            # Treat 00-30 as 2000+, otherwise 1900+.
            return 2000 + y if y <= 30 else 1900 + y
        return y

    def _extract_year(self, source: str) -> Optional[int]:
        by_name = self._extract_year_from_name(source)
        if by_name:
            return by_name

        full_path = self.documents_dir / source
        if full_path.exists():
            try:
                return datetime.fromtimestamp(Path(full_path).stat().st_mtime).year
            except Exception:
                return None
        return None

    @staticmethod
    def _revision_boost(source: str) -> float:
        s = source.lower()
        signals = ["rev", "revised", "revision", "update", "updated", "v2", "final", "renih"]
        return 0.08 if any(sig in s for sig in signals) else 0.0

    def run(self, query: str, top_k: int = 6, newest_k: int = 3, **kwargs) -> ToolResult:
        try:
            raw_results = self.vector_store.search(query=query, top_k=max(top_k * 3, 12))
            if not raw_results:
                return ToolResult(
                    text="No version-aware matches found.",
                    structured={"matches": [], "comparison": {}},
                    success=True,
                )

            enriched: List[Dict] = []
            years = []
            for item in raw_results:
                src = item["source"]
                year = self._extract_year(src)
                if year:
                    years.append(year)
                enriched.append(
                    {
                        "source": src,
                        "content": item["content"],
                        "semantic_score": float(item["score"]),
                        "year": year,
                        "revision_boost": self._revision_boost(src),
                    }
                )

            min_year = min(years) if years else None
            max_year = max(years) if years else None

            def recency_score(y: Optional[int]) -> float:
                if y is None or min_year is None or max_year is None or min_year == max_year:
                    return 0.5
                return (y - min_year) / (max_year - min_year)

            for row in enriched:
                row["recency_score"] = recency_score(row["year"])
                row["final_score"] = (
                    0.72 * row["semantic_score"]
                    + 0.22 * row["recency_score"]
                    + row["revision_boost"]
                )

            enriched.sort(key=lambda x: x["final_score"], reverse=True)
            selected = enriched[:top_k]

            # Build source-level comparison.
            source_best: Dict[str, Dict] = {}
            for row in selected:
                if row["source"] not in source_best:
                    source_best[row["source"]] = row

            source_rows = list(source_best.values())
            source_rows.sort(key=lambda x: ((x["year"] or 0), x["final_score"]), reverse=True)
            newest = source_rows[0] if source_rows else None
            older = source_rows[1] if len(source_rows) > 1 else None

            comparison = {
                "query": query,
                "newest_source": newest["source"] if newest else None,
                "newest_year": newest["year"] if newest else None,
                "older_source": older["source"] if older else None,
                "older_year": older["year"] if older else None,
                "source_count": len(source_rows),
            }

            lines = []
            lines.append("Version-aware retrieval summary")
            if newest:
                lines.append(
                    f"- Preferred source: {newest['source']} (year={newest['year']}, final={newest['final_score']:.3f})"
                )
            if older:
                lines.append(
                    f"- Compared with older source: {older['source']} (year={older['year']}, final={older['final_score']:.3f})"
                )
            lines.append("- Top evidence snippets (reranked by semantic relevance + recency):")
            for i, row in enumerate(selected[:newest_k], 1):
                lines.append(
                    f"{i}. [{row['source']} | year={row['year']} | final={row['final_score']:.3f}] "
                    f"{row['content'][:260].replace(chr(10), ' ')}..."
                )

            citations = []
            for row in selected:
                citations.append(
                    Citation(
                        source=row["source"],
                        passage=row["content"][:500] + ("..." if len(row["content"]) > 500 else ""),
                        relevance_score=row["final_score"],
                        metadata={
                            "semantic_score": row["semantic_score"],
                            "recency_score": row["recency_score"],
                            "year": row["year"],
                            "revision_boost": row["revision_boost"],
                        },
                    )
                )

            return ToolResult(
                text="\n".join(lines),
                structured={
                    "matches": selected,
                    "comparison": comparison,
                },
                citations=citations,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                text=f"Versioned retrieval error: {str(e)}",
                success=False,
                error=str(e),
            )

