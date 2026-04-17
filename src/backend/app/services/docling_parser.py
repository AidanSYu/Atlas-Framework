"""Docling-based document parser for structure-preserving extraction.

Preserves table structure, headings, and layout that basic PDF extractors lose.
Falls back gracefully if Docling is not installed.
"""
from pathlib import Path
from typing import List, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except Exception as _docling_exc:
    # Docling is notoriously fragile — ImportError when missing, AttributeError or
    # RuntimeError when a transitive dep has drifted versions. Any failure means we
    # fall back to pdfplumber; we must not let docling's broken import kill the server.
    DOCLING_AVAILABLE = False
    DocumentConverter = None  # type: ignore
    logger.warning(
        "Docling unavailable (%s: %s) — falling back to pdfplumber",
        type(_docling_exc).__name__,
        _docling_exc,
    )


class DoclingParser:
    """Wrapper for Docling VLM parsing with lazy initialization."""

    _instance = None

    def __init__(self):
        self._converter = None

    def _ensure_converter(self):
        """Lazy-load the DocumentConverter (heavy initialization)."""
        if self._converter is not None:
            return

        if not DOCLING_AVAILABLE:
            raise ImportError("docling is not installed")

        self._converter = DocumentConverter()
        logger.info("Docling DocumentConverter initialized")

    def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse document preserving tables, images, and structure.

        Args:
            file_path: Path to the PDF or document file

        Returns:
            List of page dicts with text (markdown-formatted), tables, and metadata
        """
        self._ensure_converter()

        try:
            t0 = time.perf_counter()
            logger.info(f"Docling: starting PDF conversion for {Path(file_path).name} (this uses CPU-based layout/OCR models and may take a few minutes)...")
            result = self._converter.convert(file_path)
            logger.info(f"Docling: PDF conversion completed in {time.perf_counter() - t0:.1f}s")
            doc = result.document

            # Export as markdown to preserve table structure
            markdown_text = doc.export_to_markdown()

            if not markdown_text or not markdown_text.strip():
                logger.warning(f"Docling produced empty output for {file_path}")
                return []

            # Split by page breaks if available, otherwise treat as single page
            pages = []

            # Docling provides page-level iteration through the document items
            page_texts: Dict[int, List[str]] = {}
            page_tables: Dict[int, List[str]] = {}

            for item, _level in doc.iterate_items():
                # Get page number from item's provenance
                page_num = 1
                if hasattr(item, 'prov') and item.prov:
                    page_num = item.prov[0].page_no if item.prov[0].page_no else 1

                if page_num not in page_texts:
                    page_texts[page_num] = []
                    page_tables[page_num] = []

                # Export item to markdown; skip items that would yield Python repr (RefItem, ContentLayer, etc.)
                if hasattr(item, 'export_to_markdown'):
                    item_text = item.export_to_markdown()
                else:
                    # Avoid str(item) - it produces internal reprs (RefItem, ContentLayer.BODY, etc.)
                    continue

                if hasattr(item, 'label') and 'table' in str(item.label).lower():
                    page_tables[page_num].append(item_text)
                else:
                    page_texts[page_num].append(item_text)

            if page_texts:
                for page_num in sorted(page_texts.keys()):
                    text_parts = page_texts.get(page_num, [])
                    table_parts = page_tables.get(page_num, [])
                    combined_text = "\n\n".join(text_parts)

                    if table_parts:
                        combined_text += "\n\n" + "\n\n".join(table_parts)

                    if combined_text.strip():
                        pages.append({
                            "page_number": page_num,
                            "text": combined_text.strip(),
                            "tables": table_parts,
                            "char_count": len(combined_text),
                            "has_tables": len(table_parts) > 0,
                            "metadata": {
                                "parser": "docling",
                                "structure_preserved": True,
                            }
                        })
            else:
                # Fallback: use full markdown as single page
                pages.append({
                    "page_number": 1,
                    "text": markdown_text.strip(),
                    "tables": [],
                    "char_count": len(markdown_text),
                    "has_tables": False,
                    "metadata": {
                        "parser": "docling",
                        "structure_preserved": True,
                    }
                })

            elapsed = time.perf_counter() - t0
            logger.info(f"Docling extracted {len(pages)} pages from {Path(file_path).name} in {elapsed:.1f}s total")
            return pages

        except Exception as e:
            logger.error(f"Docling parsing failed for {file_path}: {e}", exc_info=True)
            raise

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        return DOCLING_AVAILABLE
