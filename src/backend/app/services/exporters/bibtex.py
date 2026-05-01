"""Export citations from Atlas in BibTeX format.

Generates standards-compliant .bib files from Document metadata,
suitable for import into Zotero, Mendeley, LaTeX editors, etc.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from app.core.database import Document, get_project_session

logger = logging.getLogger(__name__)


class BibTeXExporter:
    """Export Document metadata as BibTeX entries."""

    # Map Atlas paper_type to BibTeX entry types
    ENTRY_TYPE_MAP = {
        "empirical": "article",
        "review": "article",
        "theoretical": "article",
        "meta-analysis": "article",
        "conference": "inproceedings",
        "book": "book",
        "chapter": "incollection",
        "thesis": "phdthesis",
        "report": "techreport",
        "preprint": "misc",
    }

    def export_project(self, project_id: str) -> str:
        """Export all documents in a project as a .bib file."""
        session = get_project_session(project_id)
        try:
            documents = (
                session.query(Document)
                .order_by(Document.uploaded_at.desc())
                .all()
            )
            entries = []
            for doc in documents:
                entry = self._document_to_bibtex(doc)
                if entry:
                    entries.append(entry)
            return "\n\n".join(entries) + "\n" if entries else ""
        finally:
            session.close()

    def export_documents(self, project_id: str, doc_ids: List[str]) -> str:
        """Export specific documents as BibTeX entries."""
        session = get_project_session(project_id)
        try:
            documents = (
                session.query(Document)
                .filter(Document.id.in_(doc_ids))
                .all()
            )
            entries = []
            for doc in documents:
                entry = self._document_to_bibtex(doc)
                if entry:
                    entries.append(entry)
            return "\n\n".join(entries) + "\n" if entries else ""
        finally:
            session.close()

    def _document_to_bibtex(self, doc: Document) -> Optional[str]:
        """Convert a Document record to a BibTeX entry string."""
        meta = doc.doc_metadata or {}
        if not meta.get("title") and not doc.filename:
            return None

        title = meta.get("title", doc.filename.rsplit(".", 1)[0] if doc.filename else "Untitled")
        authors = meta.get("authors", [])
        year = meta.get("year")

        # Generate a bibtex key if not present
        bibtex_key = meta.get("bibtex_key", "")
        if not bibtex_key:
            bibtex_key = self._generate_key(authors, year, title)

        # Determine entry type
        paper_type = meta.get("entry_type", meta.get("paper_type", "article"))
        entry_type = self.ENTRY_TYPE_MAP.get(paper_type, "article")

        # Build fields
        fields = []
        fields.append(f"  title = {{{self._escape_bibtex(title)}}}")

        if authors:
            author_str = " and ".join(authors)
            fields.append(f"  author = {{{self._escape_bibtex(author_str)}}}")

        if year:
            fields.append(f"  year = {{{year}}}")

        if meta.get("journal"):
            if entry_type == "inproceedings":
                fields.append(f"  booktitle = {{{self._escape_bibtex(meta['journal'])}}}")
            else:
                fields.append(f"  journal = {{{self._escape_bibtex(meta['journal'])}}}")

        if meta.get("volume"):
            fields.append(f"  volume = {{{meta['volume']}}}")

        if meta.get("number"):
            fields.append(f"  number = {{{meta['number']}}}")

        if meta.get("pages"):
            # Normalize page ranges to use --
            pages = meta["pages"].replace("-", "--").replace("----", "--")
            fields.append(f"  pages = {{{pages}}}")

        if meta.get("doi"):
            fields.append(f"  doi = {{{meta['doi']}}}")

        if meta.get("url"):
            fields.append(f"  url = {{{meta['url']}}}")

        if meta.get("publisher"):
            fields.append(f"  publisher = {{{self._escape_bibtex(meta['publisher'])}}}")

        if meta.get("abstract"):
            fields.append(f"  abstract = {{{self._escape_bibtex(meta['abstract'])}}}")

        if meta.get("keywords"):
            fields.append(f"  keywords = {{{self._escape_bibtex(meta['keywords'])}}}")

        fields_str = ",\n".join(fields)
        return f"@{entry_type}{{{bibtex_key},\n{fields_str}\n}}"

    def format_citation(
        self, doc_metadata: Dict[str, Any], style: str = "apa"
    ) -> str:
        """Format a single citation in APA, MLA, or Chicago style."""
        title = doc_metadata.get("title", "Untitled")
        authors = doc_metadata.get("authors", [])
        year = doc_metadata.get("year", "n.d.")
        journal = doc_metadata.get("journal", "")

        if style == "apa":
            return self._format_apa(authors, year, title, journal, doc_metadata)
        elif style == "mla":
            return self._format_mla(authors, title, journal, doc_metadata)
        elif style == "chicago":
            return self._format_chicago(authors, year, title, journal, doc_metadata)
        else:
            return self._format_apa(authors, year, title, journal, doc_metadata)

    def _format_apa(
        self, authors: List[str], year, title: str, journal: str, meta: Dict
    ) -> str:
        """APA 7th edition format."""
        # Authors: Last, F. I., & Last, F. I.
        if authors:
            formatted = []
            for a in authors[:20]:  # APA includes up to 20
                parts = a.split(",", 1)
                if len(parts) == 2:
                    last = parts[0].strip()
                    first = parts[1].strip()
                    initials = ". ".join(n[0] for n in first.split() if n) + "."
                    formatted.append(f"{last}, {initials}")
                else:
                    formatted.append(a.strip())

            if len(formatted) == 1:
                author_str = formatted[0]
            elif len(formatted) == 2:
                author_str = f"{formatted[0]}, & {formatted[1]}"
            else:
                author_str = ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
        else:
            author_str = "Unknown"

        citation = f"{author_str} ({year}). {title}."
        if journal:
            citation += f" *{journal}*"
            if meta.get("volume"):
                citation += f", *{meta['volume']}*"
            if meta.get("number"):
                citation += f"({meta['number']})"
            if meta.get("pages"):
                citation += f", {meta['pages']}"
            citation += "."
        if meta.get("doi"):
            citation += f" https://doi.org/{meta['doi']}"

        return citation

    def _format_mla(
        self, authors: List[str], title: str, journal: str, meta: Dict
    ) -> str:
        """MLA 9th edition format."""
        if authors:
            if len(authors) == 1:
                author_str = authors[0]
            elif len(authors) == 2:
                author_str = f"{authors[0]}, and {authors[1]}"
            else:
                author_str = f"{authors[0]}, et al."
        else:
            author_str = "Unknown"

        citation = f'{author_str}. "{title}."'
        if journal:
            citation += f" *{journal}*"
            if meta.get("volume"):
                citation += f", vol. {meta['volume']}"
            if meta.get("number"):
                citation += f", no. {meta['number']}"
            if meta.get("year"):
                citation += f", {meta['year']}"
            if meta.get("pages"):
                citation += f", pp. {meta['pages']}"
            citation += "."
        if meta.get("doi"):
            citation += f" https://doi.org/{meta['doi']}"

        return citation

    def _format_chicago(
        self, authors: List[str], year, title: str, journal: str, meta: Dict
    ) -> str:
        """Chicago 17th edition (author-date) format."""
        if authors:
            if len(authors) == 1:
                author_str = authors[0]
            elif len(authors) <= 3:
                author_str = ", ".join(authors[:-1]) + f", and {authors[-1]}"
            else:
                author_str = f"{authors[0]} et al."
        else:
            author_str = "Unknown"

        citation = f'{author_str}. {year}. "{title}."'
        if journal:
            citation += f" *{journal}*"
            if meta.get("volume"):
                citation += f" {meta['volume']}"
            if meta.get("number"):
                citation += f", no. {meta['number']}"
            if meta.get("pages"):
                citation += f": {meta['pages']}"
            citation += "."
        if meta.get("doi"):
            citation += f" https://doi.org/{meta['doi']}"

        return citation

    def _generate_key(
        self, authors: List[str], year: Optional[int], title: str
    ) -> str:
        """Generate a BibTeX key from authors, year, and title."""
        if authors:
            first_author = authors[0].split(",")[0].strip().lower()
            first_author = re.sub(r"[^a-z]", "", first_author)
        else:
            # Use first word of title
            first_author = re.sub(r"[^a-z]", "", title.split()[0].lower()) if title else "unknown"

        year_str = str(year) if year else "nd"
        return f"{first_author}{year_str}"

    def _escape_bibtex(self, text: str) -> str:
        """Escape special BibTeX characters."""
        if not text:
            return ""
        # Protect special chars that aren't already in braces
        text = text.replace("&", r"\&")
        text = text.replace("%", r"\%")
        text = text.replace("#", r"\#")
        return text
