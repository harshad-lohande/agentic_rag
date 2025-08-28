# app/document_parser.py

from pathlib import Path
from typing import Dict, Any, Iterator
import docx
from pypdf import PdfReader

class DocumentParser:
    """
    A class to parse different document types and extract text and metadata.
    It now supports streaming for large documents to be more memory-efficient.
    """

    def parse(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Parses a file and yields its text content in chunks along with metadata.

        Args:
            file_path: The path to the file.

        Yields:
            A dictionary containing a chunk of text content and metadata.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found at: {file_path}")

        file_extension = path.suffix.lower()
        metadata = self._get_metadata(path)

        parsing_method = {
            ".pdf": self._stream_pdf,
            ".docx": self._stream_docx,
            ".txt": self._stream_txt,
        }.get(file_extension)

        if parsing_method:
            for chunk in parsing_method(path):
                yield {"text": chunk, "metadata": metadata}
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _get_metadata(self, path: Path) -> Dict[str, Any]:
        """Extracts basic metadata from a file."""
        return {
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_size": path.stat().st_size,
            "creation_date": path.stat().st_ctime,
            "modification_date": path.stat().st_mtime,
        }

    def _stream_pdf(self, path: Path) -> Iterator[str]:
        """Streams text from a PDF file, page by page."""
        reader = PdfReader(path)
        for page in reader.pages:
            yield page.extract_text()

    def _stream_docx(self, path: Path) -> Iterator[str]:
        """Streams text from a DOCX file, paragraph by paragraph."""
        doc = docx.Document(path)
        for para in doc.paragraphs:
            if para.text:
                yield para.text

    def _stream_txt(self, path: Path, chunk_size: int = 4096) -> Iterator[str]:
        """Streams text from a TXT file in chunks."""
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk