"""Import papers from BibTeX (.bib) and RIS (.ris) files.

Supports Zotero, Mendeley, and manual BibTeX exports. Creates Document
records with status="metadata_only" so PDFs can be attached later.
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.database import Document, get_project_session

logger = logging.getLogger(__name__)


class BibTeXImporter:
    """Parse BibTeX files and create Document metadata records."""

    def import_from_string(
        self, bib_content: str, project_id: str
    ) -> Dict[str, Any]:
        """Parse a .bib string and create Document records for each entry.

        Returns:
            {
                "status": "success",
                "imported": [{"doc_id": "...", "bibtex_key": "...", "title": "..."}],
                "skipped": [{"bibtex_key": "...", "reason": "..."}],
                "total_entries": int,
                "total_imported": int,
            }
        """
        entries = self._parse_bibtex(bib_content)
        imported = []
        skipped = []

        session = get_project_session(project_id)
        try:
            for entry in entries:
                bibtex_key = entry.get("ID", "")
                title = entry.get("title", "Untitled")

                # Check for duplicate by bibtex_key in doc_metadata
                existing = (
                    session.query(Document)
                    .filter(
                        Document.project_id == project_id,
                        Document.doc_metadata["bibtex_key"].as_string() == bibtex_key,
                    )
                    .first()
                )
                if existing:
                    skipped.append(
                        {"bibtex_key": bibtex_key, "reason": "duplicate"}
                    )
                    continue

                doc_id = str(uuid.uuid4())
                authors = self._parse_authors(entry.get("author", ""))
                year = self._parse_year(entry.get("year", ""))

                doc_metadata = {
                    "title": self._clean_latex(title),
                    "authors": authors,
                    "year": year,
                    "journal": self._clean_latex(entry.get("journal", entry.get("booktitle", ""))),
                    "doi": entry.get("doi", ""),
                    "bibtex_key": bibtex_key,
                    "abstract": self._clean_latex(entry.get("abstract", "")),
                    "entry_type": entry.get("ENTRYTYPE", "article"),
                    "volume": entry.get("volume", ""),
                    "number": entry.get("number", ""),
                    "pages": entry.get("pages", ""),
                    "publisher": entry.get("publisher", ""),
                    "url": entry.get("url", ""),
                    "keywords": entry.get("keywords", ""),
                    "import_source": "bibtex",
                }

                document = Document(
                    id=doc_id,
                    filename=f"{bibtex_key}.bib",
                    file_hash=f"bibtex:{bibtex_key}",
                    file_path="",  # No file yet
                    file_size=0,
                    mime_type="application/x-bibtex",
                    status="metadata_only",
                    project_id=project_id,
                    uploaded_at=datetime.utcnow(),
                    doc_metadata=doc_metadata,
                )
                session.add(document)
                imported.append({
                    "doc_id": doc_id,
                    "bibtex_key": bibtex_key,
                    "title": doc_metadata["title"],
                    "authors": authors,
                    "year": year,
                })

            session.commit()
            logger.info(
                f"BibTeX import: {len(imported)} imported, {len(skipped)} skipped"
            )
        except Exception as e:
            session.rollback()
            logger.error(f"BibTeX import failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "imported": [],
                "skipped": [],
                "total_entries": len(entries),
                "total_imported": 0,
            }
        finally:
            session.close()

        return {
            "status": "success",
            "imported": imported,
            "skipped": skipped,
            "total_entries": len(entries),
            "total_imported": len(imported),
        }

    def _parse_bibtex(self, content: str) -> List[Dict[str, str]]:
        """Parse BibTeX content into a list of entry dicts.

        Uses a simple regex-based parser to avoid requiring bibtexparser
        as a hard dependency. Handles standard .bib format.
        """
        entries = []
        # Match @type{key, ... }
        entry_pattern = re.compile(
            r"@(\w+)\s*\{\s*([^,]*?)\s*,\s*(.*?)\n\s*\}",
            re.DOTALL,
        )

        for match in entry_pattern.finditer(content):
            entry_type = match.group(1).lower()
            entry_key = match.group(2).strip()
            fields_str = match.group(3)

            entry = {"ENTRYTYPE": entry_type, "ID": entry_key}

            # Parse field = {value} or field = "value" or field = number
            field_pattern = re.compile(
                r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|(\d+))",
                re.DOTALL,
            )
            for field_match in field_pattern.finditer(fields_str):
                key = field_match.group(1).lower()
                value = (
                    field_match.group(2)
                    or field_match.group(3)
                    or field_match.group(4)
                    or ""
                )
                entry[key] = value.strip()

            entries.append(entry)

        return entries

    def _parse_authors(self, author_str: str) -> List[str]:
        """Parse BibTeX author string ('Last, First and Last, First')."""
        if not author_str:
            return []
        authors = [self._clean_latex(a.strip()) for a in author_str.split(" and ")]
        return [a for a in authors if a]

    def _parse_year(self, year_str: str) -> Optional[int]:
        """Extract year as integer."""
        match = re.search(r"(\d{4})", year_str)
        return int(match.group(1)) if match else None

    def _clean_latex(self, text: str) -> str:
        """Remove common LaTeX markup from BibTeX fields."""
        if not text:
            return ""
        # Remove braces used for capitalization protection
        text = text.replace("{", "").replace("}", "")
        # Remove common LaTeX commands
        text = re.sub(r"\\textit\s*", "", text)
        text = re.sub(r"\\textbf\s*", "", text)
        text = re.sub(r"\\emph\s*", "", text)
        text = re.sub(r"\\\w+\s*", "", text)  # Generic \command
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


class RISImporter:
    """Parse RIS files and create Document metadata records."""

    # RIS tag to field name mapping
    TAG_MAP = {
        "TY": "entry_type",
        "TI": "title",
        "T1": "title",
        "AU": "author",
        "A1": "author",
        "PY": "year",
        "Y1": "year",
        "DA": "date",
        "JO": "journal",
        "JF": "journal",
        "T2": "journal",
        "VL": "volume",
        "IS": "number",
        "SP": "start_page",
        "EP": "end_page",
        "DO": "doi",
        "UR": "url",
        "AB": "abstract",
        "N2": "abstract",
        "KW": "keyword",
        "PB": "publisher",
        "SN": "issn",
        "ID": "id",
    }

    def import_from_string(
        self, ris_content: str, project_id: str
    ) -> Dict[str, Any]:
        """Parse a .ris string and create Document records."""
        entries = self._parse_ris(ris_content)
        imported = []
        skipped = []

        session = get_project_session(project_id)
        try:
            for entry in entries:
                title = entry.get("title", "Untitled")
                # Generate a key from first author + year
                authors = entry.get("authors", [])
                year = entry.get("year")
                first_author_last = authors[0].split(",")[0].strip().lower() if authors else "unknown"
                ris_key = f"{first_author_last}{year or 'nd'}"

                doc_id = str(uuid.uuid4())
                pages = ""
                if entry.get("start_page"):
                    pages = entry["start_page"]
                    if entry.get("end_page"):
                        pages += f"-{entry['end_page']}"

                doc_metadata = {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "journal": entry.get("journal", ""),
                    "doi": entry.get("doi", ""),
                    "bibtex_key": ris_key,
                    "abstract": entry.get("abstract", ""),
                    "entry_type": entry.get("entry_type", "article"),
                    "volume": entry.get("volume", ""),
                    "number": entry.get("number", ""),
                    "pages": pages,
                    "publisher": entry.get("publisher", ""),
                    "url": entry.get("url", ""),
                    "keywords": ", ".join(entry.get("keywords", [])),
                    "import_source": "ris",
                }

                document = Document(
                    id=doc_id,
                    filename=f"{ris_key}.ris",
                    file_hash=f"ris:{ris_key}:{doc_id[:8]}",
                    file_path="",
                    file_size=0,
                    mime_type="application/x-research-info-systems",
                    status="metadata_only",
                    project_id=project_id,
                    uploaded_at=datetime.utcnow(),
                    doc_metadata=doc_metadata,
                )
                session.add(document)
                imported.append({
                    "doc_id": doc_id,
                    "bibtex_key": ris_key,
                    "title": title,
                    "authors": authors,
                    "year": year,
                })

            session.commit()
            logger.info(
                f"RIS import: {len(imported)} imported, {len(skipped)} skipped"
            )
        except Exception as e:
            session.rollback()
            logger.error(f"RIS import failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "imported": [],
                "skipped": [],
                "total_entries": len(entries),
                "total_imported": 0,
            }
        finally:
            session.close()

        return {
            "status": "success",
            "imported": imported,
            "skipped": skipped,
            "total_entries": len(entries),
            "total_imported": len(imported),
        }

    def _parse_ris(self, content: str) -> List[Dict[str, Any]]:
        """Parse RIS format into structured entries."""
        entries = []
        current: Dict[str, Any] = {}
        current_authors: List[str] = []
        current_keywords: List[str] = []

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue

            # RIS format: XX  - Value
            match = re.match(r"^([A-Z][A-Z0-9])\s+-\s+(.*)", line)
            if not match:
                continue

            tag = match.group(1)
            value = match.group(2).strip()

            if tag == "ER":
                # End of record
                if current:
                    current["authors"] = current_authors
                    current["keywords"] = current_keywords
                    entries.append(current)
                current = {}
                current_authors = []
                current_keywords = []
                continue

            if tag == "TY":
                current["entry_type"] = value.lower()
                continue

            field = self.TAG_MAP.get(tag)
            if not field:
                continue

            if field == "author":
                current_authors.append(value)
            elif field == "keyword":
                current_keywords.append(value)
            elif field == "year":
                year_match = re.search(r"(\d{4})", value)
                current["year"] = int(year_match.group(1)) if year_match else None
            else:
                current[field] = value

        # Handle file that doesn't end with ER
        if current:
            current["authors"] = current_authors
            current["keywords"] = current_keywords
            entries.append(current)

        return entries
