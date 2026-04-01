"""Document parsing utilities for PDF and DOCX files.

Provides magic-byte file type detection and per-page text extraction with
section title inference. All public functions are synchronous; call them
from a thread-pool executor if you need to keep the event loop free.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Literal

import structlog

from app.core.exceptions import DocumentProcessingError, UnsupportedFileTypeError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Magic bytes
# ---------------------------------------------------------------------------
_PDF_MAGIC = b"\x25\x50\x44\x46"  # %PDF
_ZIP_MAGIC = b"\x50\x4b\x03\x04"  # PK.. (ZIP / DOCX / XLSX …)

# Approximate characters per synthetic DOCX "page"
_CHARS_PER_PAGE = 3000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ParsedPage:
    text: str
    page_number: int
    section_title: str | None


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------


def detect_file_type(file_bytes: bytes) -> Literal["pdf", "docx"]:
    """Return ``'pdf'`` or ``'docx'`` by inspecting magic bytes.

    Raises :class:`~app.core.exceptions.UnsupportedFileTypeError` for any
    other byte sequence.
    """
    if len(file_bytes) < 4:
        raise UnsupportedFileTypeError(
            "File is too small to determine type (fewer than 4 bytes)"
        )

    header = file_bytes[:4]

    if header == _PDF_MAGIC:
        logger.debug("File type detected as PDF")
        return "pdf"

    if header == _ZIP_MAGIC:
        # A DOCX must be a ZIP that contains a ``word/`` directory entry.
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                names = zf.namelist()
            if any(n.startswith("word/") for n in names):
                logger.debug("File type detected as DOCX")
                return "docx"
        except zipfile.BadZipFile:
            pass
        raise UnsupportedFileTypeError(
            "File looks like a ZIP archive but is not a valid DOCX (no word/ entries)"
        )

    raise UnsupportedFileTypeError(
        f"Unsupported file type — unrecognised magic bytes: {header!r}"
    )


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------


def _extract_section_title_from_pdfplumber_page(page) -> str | None:  # type: ignore[no-untyped-def]
    """Try to extract a section title by looking for the largest or boldest text
    on the page.  Returns ``None`` when no heuristic fires."""
    try:
        chars = page.chars
        if not chars:
            return None

        # Group characters by (fontname, size) and look for a "heading" style.
        # We consider a run heading-like when its font size is >= 1.2× the
        # median size of the body text on the page.
        sizes = [c.get("size", 0) for c in chars if c.get("size")]
        if not sizes:
            return None

        sizes_sorted = sorted(sizes)
        median_size = sizes_sorted[len(sizes_sorted) // 2]
        threshold = median_size * 1.2

        heading_chars = [c for c in chars if c.get("size", 0) >= threshold]
        if not heading_chars:
            return None

        # Reconstruct text from heading characters preserving order.
        heading_text = "".join(c.get("text", "") for c in heading_chars).strip()
        # Truncate to a reasonable title length.
        if heading_text:
            return heading_text[:120]
    except Exception:  # noqa: BLE001
        pass
    return None


def parse_pdf(file_bytes: bytes) -> list[ParsedPage]:
    """Parse a PDF and return one :class:`ParsedPage` per page.

    Tries *pdfplumber* first (richer layout analysis).  Falls back to
    *pypdf* when pdfplumber fails or returns empty text.

    Raises :class:`~app.core.exceptions.DocumentProcessingError` if neither
    library can extract any text.
    """
    pages: list[ParsedPage] = []

    # --- pdfplumber attempt ---
    try:
        import pdfplumber  # noqa: PLC0415

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                section_title = _extract_section_title_from_pdfplumber_page(page)
                pages.append(
                    ParsedPage(
                        text=text.strip(),
                        page_number=page_num,
                        section_title=section_title,
                    )
                )

        if any(p.text for p in pages):
            logger.info("PDF parsed with pdfplumber", page_count=len(pages))
            return pages
        logger.debug("pdfplumber returned empty text, falling back to pypdf")
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber failed, falling back to pypdf", error=str(exc))

    # --- pypdf fallback ---
    pages = []
    try:
        from pypdf import PdfReader  # noqa: PLC0415

        reader = PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(
                ParsedPage(
                    text=text.strip(),
                    page_number=page_num,
                    section_title=None,
                )
            )

        if any(p.text for p in pages):
            logger.info("PDF parsed with pypdf", page_count=len(pages))
            return pages
    except Exception as exc:  # noqa: BLE001
        logger.error("pypdf also failed", error=str(exc))

    if not pages:
        raise DocumentProcessingError("Could not extract any text from PDF")

    return pages


# ---------------------------------------------------------------------------
# DOCX parsing
# ---------------------------------------------------------------------------

# Heading styles used by python-docx / Word to identify section titles.
_HEADING_STYLES = frozenset(
    {
        "heading 1",
        "heading 2",
        "heading 3",
        "heading 4",
        "title",
        "subtitle",
    }
)


def parse_docx(file_bytes: bytes) -> list[ParsedPage]:
    """Parse a DOCX and return a list of :class:`ParsedPage` objects.

    Because DOCX files have no physical pages, synthetic page numbers are
    assigned based on cumulative character count (≈ ``_CHARS_PER_PAGE``
    characters per page).

    Section titles are inferred from Word heading paragraph styles.

    Raises :class:`~app.core.exceptions.DocumentProcessingError` on failure.
    """
    try:
        from docx import Document  # noqa: PLC0415
    except ImportError as exc:
        raise DocumentProcessingError(
            "python-docx is not installed — cannot parse DOCX files"
        ) from exc

    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentProcessingError(f"Failed to open DOCX file: {exc}") from exc

    pages: list[ParsedPage] = []
    current_chars = 0
    current_page = 1
    current_section: str | None = None
    page_buffer: list[str] = []

    def _flush_page(page_num: int, section: str | None, text_parts: list[str]) -> None:
        text = "\n".join(text_parts).strip()
        if text:
            pages.append(
                ParsedPage(
                    text=text,
                    page_number=page_num,
                    section_title=section,
                )
            )

    for para in doc.paragraphs:
        style_name = (para.style.name or "").lower()
        para_text = para.text.strip()
        if not para_text:
            continue

        # Detect heading → update current section title
        if style_name in _HEADING_STYLES:
            current_section = para_text

        # Accumulate text; break into a new synthetic page when threshold hit
        page_buffer.append(para_text)
        current_chars += len(para_text)

        if current_chars >= _CHARS_PER_PAGE:
            _flush_page(current_page, current_section, page_buffer)
            current_page += 1
            current_chars = 0
            page_buffer = []

    # Flush any remaining text into the last page.
    if page_buffer:
        _flush_page(current_page, current_section, page_buffer)

    if not pages:
        raise DocumentProcessingError("Could not extract any text from DOCX")

    logger.info("DOCX parsed", page_count=len(pages))
    return pages


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def parse_document(file_bytes: bytes, file_type: str) -> list[ParsedPage]:
    """Dispatch to the appropriate parser based on *file_type*.

    Raises :class:`~app.core.exceptions.UnsupportedFileTypeError` for unknown
    types, :class:`~app.core.exceptions.DocumentProcessingError` on parse
    failure.
    """
    if file_type == "pdf":
        return parse_pdf(file_bytes)
    if file_type == "docx":
        return parse_docx(file_bytes)
    raise UnsupportedFileTypeError(f"Unsupported file type: '{file_type}'")
