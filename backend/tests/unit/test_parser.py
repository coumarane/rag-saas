"""Unit tests for app.ingestion.parser."""

from __future__ import annotations

import io
import sys
import zipfile
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import DocumentProcessingError, UnsupportedFileTypeError
from app.ingestion.parser import (
    ParsedPage,
    detect_file_type,
    parse_docx,
    parse_document,
    parse_pdf,
)

# ---------------------------------------------------------------------------
# Helpers — minimal valid byte sequences
# ---------------------------------------------------------------------------

_PDF_MAGIC = b"%PDF-1.4\n%%EOF"  # Starts with %PDF
_ZIP_MAGIC = b"\x50\x4b\x03\x04"  # ZIP magic

_GARBAGE_BYTES = b"\xDE\xAD\xBE\xEF this is not a recognised file"


def _make_docx_bytes() -> bytes:
    """Create a minimal ZIP archive that looks like a DOCX (has word/ entries)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "<root/>")
        zf.writestr("[Content_Types].xml", "<Types/>")
    return buf.getvalue()


def _make_zip_no_word_bytes() -> bytes:
    """A valid ZIP that does NOT contain word/ entries (not a DOCX)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "world")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# detect_file_type
# ---------------------------------------------------------------------------


class TestDetectFileType:
    def test_detects_pdf_magic(self):
        result = detect_file_type(_PDF_MAGIC)
        assert result == "pdf"

    def test_detects_docx_via_zip_with_word_entries(self):
        docx_bytes = _make_docx_bytes()
        result = detect_file_type(docx_bytes)
        assert result == "docx"

    def test_raises_for_zip_without_word_entries(self):
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(_make_zip_no_word_bytes())

    def test_raises_for_unrecognised_magic(self):
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(_GARBAGE_BYTES)

    def test_raises_for_empty_bytes(self):
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(b"\x00")

    def test_raises_for_too_short(self):
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(b"AB")


# ---------------------------------------------------------------------------
# parse_pdf
# ---------------------------------------------------------------------------


def _make_mock_pdfplumber_module(pages_spec: list[dict]) -> ModuleType:
    """Return a fake pdfplumber module whose ``open()`` yields *pages_spec* pages."""
    mock_pages = []
    for spec in pages_spec:
        page = MagicMock()
        page.extract_text.return_value = spec.get("text", "")
        page.chars = spec.get("chars", [])
        mock_pages.append(page)

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_ctx.pages = mock_pages

    mod = ModuleType("pdfplumber")
    mod.open = MagicMock(return_value=mock_ctx)  # type: ignore[attr-defined]
    return mod


def _make_mock_pypdf_module(pages_spec: list[dict]) -> ModuleType:
    """Return a fake pypdf module whose ``PdfReader`` yields *pages_spec* pages."""
    mock_reader = MagicMock()
    mock_pages = []
    for spec in pages_spec:
        page = MagicMock()
        page.extract_text.return_value = spec.get("text", "")
        mock_pages.append(page)
    mock_reader.pages = mock_pages

    mod = ModuleType("pypdf")
    mod.PdfReader = MagicMock(return_value=mock_reader)  # type: ignore[attr-defined]
    return mod


def _make_mock_docx_module(paragraphs_spec: list[dict]) -> ModuleType:
    """Return a fake docx module whose ``Document`` yields *paragraphs_spec*."""
    mock_doc = MagicMock()
    mock_paras = []
    for spec in paragraphs_spec:
        para = MagicMock()
        para.text = spec["text"]
        style = MagicMock()
        style.name = spec.get("style", "Normal")
        para.style = style
        mock_paras.append(para)
    mock_doc.paragraphs = mock_paras

    mod = ModuleType("docx")
    mod.Document = MagicMock(return_value=mock_doc)  # type: ignore[attr-defined]
    return mod


class TestParsePDF:
    def test_parse_pdf_uses_pdfplumber_first(self):
        """pdfplumber is tried first; mock it returning one page of text."""
        plumber_mod = _make_mock_pdfplumber_module(
            [{"text": "Hello from pdfplumber", "chars": [{"size": 12, "text": "H"}]}]
        )
        with patch.dict(sys.modules, {"pdfplumber": plumber_mod}):
            pages = parse_pdf(_PDF_MAGIC)

        assert len(pages) == 1
        assert pages[0].text == "Hello from pdfplumber"
        assert pages[0].page_number == 1

    def test_parse_pdf_falls_back_to_pypdf_on_pdfplumber_failure(self):
        """When pdfplumber raises, pypdf should be tried."""
        failing_plumber = ModuleType("pdfplumber")
        failing_plumber.open = MagicMock(side_effect=Exception("pdfplumber broke"))  # type: ignore[attr-defined]

        pypdf_mod = _make_mock_pypdf_module([{"text": "Hello from pypdf"}])

        with patch.dict(sys.modules, {"pdfplumber": failing_plumber, "pypdf": pypdf_mod}):
            pages = parse_pdf(_PDF_MAGIC)

        assert len(pages) == 1
        assert pages[0].text == "Hello from pypdf"

    def test_parse_pdf_falls_back_to_pypdf_when_pdfplumber_returns_empty(self):
        """When pdfplumber returns empty text, fall back to pypdf."""
        plumber_mod = _make_mock_pdfplumber_module([{"text": "", "chars": []}])
        pypdf_mod = _make_mock_pypdf_module([{"text": "pypdf fallback text"}])

        with patch.dict(sys.modules, {"pdfplumber": plumber_mod, "pypdf": pypdf_mod}):
            pages = parse_pdf(_PDF_MAGIC)

        assert any(p.text == "pypdf fallback text" for p in pages)

    def test_parse_pdf_raises_when_both_fail(self):
        failing_plumber = ModuleType("pdfplumber")
        failing_plumber.open = MagicMock(side_effect=Exception("plumber fail"))  # type: ignore[attr-defined]

        failing_pypdf = ModuleType("pypdf")
        failing_pypdf.PdfReader = MagicMock(side_effect=Exception("pypdf fail"))  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"pdfplumber": failing_plumber, "pypdf": failing_pypdf}):
            with pytest.raises(DocumentProcessingError):
                parse_pdf(_PDF_MAGIC)


# ---------------------------------------------------------------------------
# parse_docx
# ---------------------------------------------------------------------------


class TestParseDOCX:
    def test_parse_docx_returns_parsed_pages(self):
        docx_mod = _make_mock_docx_module(
            [
                {"text": "Introduction", "style": "Heading 1"},
                {"text": "This is the intro text.", "style": "Normal"},
            ]
        )
        with patch.dict(sys.modules, {"docx": docx_mod}):
            pages = parse_docx(_make_docx_bytes())

        assert len(pages) >= 1
        assert all(isinstance(p, ParsedPage) for p in pages)

    def test_parse_docx_detects_section_title_from_heading(self):
        docx_mod = _make_mock_docx_module(
            [
                {"text": "Chapter One", "style": "Heading 1"},
                {"text": "Some body text here.", "style": "Normal"},
            ]
        )
        with patch.dict(sys.modules, {"docx": docx_mod}):
            pages = parse_docx(_make_docx_bytes())

        titles = [p.section_title for p in pages]
        assert "Chapter One" in titles

    def test_parse_docx_assigns_page_numbers(self):
        # Create enough text to exceed _CHARS_PER_PAGE (3000 chars) → force multiple pages
        long_text = "word " * 700  # ~3500 chars
        docx_mod = _make_mock_docx_module(
            [
                {"text": long_text, "style": "Normal"},
                {"text": long_text, "style": "Normal"},
            ]
        )
        with patch.dict(sys.modules, {"docx": docx_mod}):
            pages = parse_docx(_make_docx_bytes())

        assert len(pages) >= 2
        page_numbers = [p.page_number for p in pages]
        assert page_numbers == sorted(page_numbers)  # monotonically increasing

    def test_parse_docx_raises_on_empty_document(self):
        docx_mod = _make_mock_docx_module([])
        with (
            patch.dict(sys.modules, {"docx": docx_mod}),
            pytest.raises(DocumentProcessingError),
        ):
            parse_docx(_make_docx_bytes())


# ---------------------------------------------------------------------------
# parse_document dispatch
# ---------------------------------------------------------------------------


class TestParseDocument:
    def test_dispatches_to_pdf(self):
        pages = [ParsedPage(text="pdf content", page_number=1, section_title=None)]
        with patch("app.ingestion.parser.parse_pdf", return_value=pages) as mock_fn:
            result = parse_document(b"data", "pdf")
        mock_fn.assert_called_once_with(b"data")
        assert result == pages

    def test_dispatches_to_docx(self):
        pages = [ParsedPage(text="docx content", page_number=1, section_title=None)]
        with patch("app.ingestion.parser.parse_docx", return_value=pages) as mock_fn:
            result = parse_document(b"data", "docx")
        mock_fn.assert_called_once_with(b"data")
        assert result == pages

    def test_raises_for_unknown_type(self):
        with pytest.raises(UnsupportedFileTypeError):
            parse_document(b"data", "txt")
