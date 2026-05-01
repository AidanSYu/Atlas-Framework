"""Export research synthesis as structured Markdown.

Compatible with Pandoc for LaTeX/PDF/DOCX conversion. Generates YAML
front matter, Pandoc-style citations [@key], and a bibliography section.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.database import Document, get_project_session
from app.services.exporters.bibtex import BibTeXExporter

logger = logging.getLogger(__name__)


class MarkdownExporter:
    """Export editor content + citations as academic Markdown."""

    def __init__(self):
        self.bibtex_exporter = BibTeXExporter()

    def export_synthesis(
        self,
        content: str,
        citations: List[Dict[str, Any]],
        project_id: str,
        title: str = "Research Synthesis",
        author: str = "",
        style: str = "apa",
    ) -> Dict[str, str]:
        """Export editor content + citations as Pandoc-compatible Markdown.

        Args:
            content: HTML or plaintext content from the TipTap editor.
            citations: List of citation dicts with at least {source, page, doc_id}.
            project_id: Project to pull Document metadata from.
            title: Document title for YAML front matter.
            author: Author name for front matter.
            style: Citation style (apa, mla, chicago).

        Returns:
            {
                "markdown": str,   # Full Markdown with front matter
                "bibtex": str,     # Companion .bib file content
                "filename": str,   # Suggested filename
            }
        """
        # Collect unique doc_ids from citations
        doc_ids = list({c["doc_id"] for c in citations if c.get("doc_id")})

        # Fetch document metadata for cited sources
        doc_metadata_map = self._fetch_document_metadata(project_id, doc_ids)

        # Build YAML front matter
        front_matter = self._build_front_matter(title, author, style)

        # Convert content from HTML to Markdown (basic conversion)
        md_content = self._html_to_markdown(content)

        # Replace inline citations with Pandoc citation syntax
        md_content = self._insert_pandoc_citations(
            md_content, citations, doc_metadata_map
        )

        # Build bibliography section
        bibliography = self._build_bibliography(
            citations, doc_metadata_map, style
        )

        # Assemble final document
        markdown = f"{front_matter}\n{md_content}\n\n{bibliography}"

        # Generate companion BibTeX file
        bibtex = self.bibtex_exporter.export_documents(project_id, doc_ids)

        # Suggested filename
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[\s]+", "-", slug)[:50]
        filename = f"{slug}-{datetime.now().strftime('%Y%m%d')}"

        return {
            "markdown": markdown,
            "bibtex": bibtex,
            "filename": filename,
        }

    def export_chat_history(
        self,
        messages: List[Dict[str, Any]],
        project_name: str = "Atlas Research",
    ) -> str:
        """Export a chat or orchestrator conversation as Markdown."""
        lines = [
            f"# {project_name} - Research Log",
            f"*Exported from Atlas on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "",
        ]

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                lines.append(f"## Q: {content}")
                lines.append("")
            elif role == "assistant":
                lines.append(content)
                lines.append("")

                # Add citations if present
                citations = msg.get("citations", [])
                if citations:
                    lines.append("**Sources:**")
                    for c in citations:
                        source = c.get("source", "Unknown")
                        page = c.get("page", "")
                        page_str = f", p. {page}" if page else ""
                        lines.append(f"- {source}{page_str}")
                    lines.append("")

                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def _build_front_matter(
        self, title: str, author: str, style: str
    ) -> str:
        """Generate YAML front matter for Pandoc."""
        lines = [
            "---",
            f"title: \"{title}\"",
        ]
        if author:
            lines.append(f"author: \"{author}\"")
        lines.extend([
            f"date: \"{datetime.now().strftime('%Y-%m-%d')}\"",
            "bibliography: references.bib",
            f"csl: {style}.csl",
            "link-citations: true",
            "---",
            "",
        ])
        return "\n".join(lines)

    def _html_to_markdown(self, html: str) -> str:
        """Basic HTML to Markdown conversion for TipTap output."""
        if not html:
            return ""

        text = html

        # Headers
        text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"# \1\n", text, flags=re.DOTALL)
        text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"## \1\n", text, flags=re.DOTALL)
        text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"### \1\n", text, flags=re.DOTALL)
        text = re.sub(r"<h4[^>]*>(.*?)</h4>", r"#### \1\n", text, flags=re.DOTALL)

        # Bold and italic
        text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text, flags=re.DOTALL)
        text = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.DOTALL)
        text = re.sub(r"<em>(.*?)</em>", r"*\1*", text, flags=re.DOTALL)
        text = re.sub(r"<i>(.*?)</i>", r"*\1*", text, flags=re.DOTALL)

        # Links
        text = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", text, flags=re.DOTALL)

        # Lists
        text = re.sub(r"<ul[^>]*>", "\n", text)
        text = re.sub(r"</ul>", "\n", text)
        text = re.sub(r"<ol[^>]*>", "\n", text)
        text = re.sub(r"</ol>", "\n", text)
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL)

        # Paragraphs and line breaks
        text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", text, flags=re.DOTALL)
        text = re.sub(r"<br\s*/?>", "\n", text)

        # Blockquotes
        text = re.sub(
            r"<blockquote[^>]*>(.*?)</blockquote>",
            lambda m: "\n".join("> " + line for line in m.group(1).strip().splitlines()) + "\n",
            text,
            flags=re.DOTALL,
        )

        # Code blocks
        text = re.sub(r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", r"```\n\1\n```\n", text, flags=re.DOTALL)
        text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)

        # Horizontal rules
        text = re.sub(r"<hr\s*/?>", "\n---\n", text)

        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Clean up excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Decode HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")

        return text.strip()

    def _insert_pandoc_citations(
        self,
        content: str,
        citations: List[Dict[str, Any]],
        doc_meta_map: Dict[str, Dict],
    ) -> str:
        """Replace [Source: filename, Page: X] style citations with Pandoc [@key, p. X]."""
        for citation in citations:
            doc_id = citation.get("doc_id", "")
            source = citation.get("source", "")
            page = citation.get("page", "")

            meta = doc_meta_map.get(doc_id, {})
            bibtex_key = meta.get("bibtex_key", "")

            if not bibtex_key:
                # Generate from author+year
                authors = meta.get("authors", [])
                year = meta.get("year")
                if authors:
                    first = re.sub(r"[^a-z]", "", authors[0].split(",")[0].lower())
                else:
                    first = re.sub(r"[^a-z]", "", source.split(".")[0].lower())
                bibtex_key = f"{first}{year or 'nd'}"

            # Build Pandoc citation
            pandoc_cite = f"[@{bibtex_key}"
            if page:
                pandoc_cite += f", p. {page}"
            pandoc_cite += "]"

            # Replace common citation patterns in the text
            patterns = [
                rf"\[Source:\s*{re.escape(source)}(?:,\s*Page:\s*{re.escape(str(page))})?\]",
                rf"\({re.escape(source)},?\s*(?:p\.?\s*{re.escape(str(page))})?\)",
            ]
            for pattern in patterns:
                content = re.sub(pattern, pandoc_cite, content, flags=re.IGNORECASE)

        return content

    def _build_bibliography(
        self,
        citations: List[Dict[str, Any]],
        doc_meta_map: Dict[str, Dict],
        style: str,
    ) -> str:
        """Build a formatted bibliography section."""
        if not citations:
            return ""

        lines = ["## References", ""]

        seen_ids = set()
        formatted = []
        for citation in citations:
            doc_id = citation.get("doc_id", "")
            if doc_id in seen_ids or not doc_id:
                continue
            seen_ids.add(doc_id)

            meta = doc_meta_map.get(doc_id, {})
            if meta:
                formatted_citation = self.bibtex_exporter.format_citation(meta, style)
                formatted.append(formatted_citation)

        # Sort alphabetically
        formatted.sort()
        for i, cite in enumerate(formatted, 1):
            lines.append(f"{i}. {cite}")
            lines.append("")

        return "\n".join(lines)

    def _fetch_document_metadata(
        self, project_id: str, doc_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch doc_metadata for a list of document IDs in a project."""
        if not doc_ids:
            return {}

        session = get_project_session(project_id)
        try:
            documents = (
                session.query(Document)
                .filter(Document.id.in_(doc_ids))
                .all()
            )
            return {
                doc.id: doc.doc_metadata or {}
                for doc in documents
            }
        finally:
            session.close()
