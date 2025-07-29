import os
from pathlib import Path
from typing import Dict, List, Any
import docx
from pypdf import PdfReader

class DocumentParser:
    """
    A class to parse different document types and extract text and metadata.
    """
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a file and returns its text content and metadata.

        Args:
            file_path: The path to the file.

        Returns:
            A dictionary containing the text content and metadata.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found at: {file_path}")

        file_extension = path.suffix.lower()
        
        if file_extension == ".pdf":
            return self._parse_pdf(path)
        elif file_extension == ".docx":
            return self._parse_docx(path)
        elif file_extension == ".txt":
            return self._parse_txt(path)
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

    def _parse_pdf(self, path: Path) -> Dict[str, Any]:
        """Parses a PDF file."""
        metadata = self._get_metadata(path)
        reader = PdfReader(path)
        text_content = "".join(page.extract_text() for page in reader.pages)
        
        # Add PDF-specific metadata
        doc_info = reader.metadata
        if doc_info:
            metadata["author"] = doc_info.author
            metadata["title"] = doc_info.title
        
        return {"text": text_content, "metadata": metadata}

    def _parse_docx(self, path: Path) -> Dict[str, Any]:
        """Parses a DOCX file."""
        metadata = self._get_metadata(path)
        doc = docx.Document(path)
        text_content = "\n".join(para.text for para in doc.paragraphs)
        
        # Add DOCX-specific metadata
        core_props = doc.core_properties
        metadata["author"] = core_props.author
        metadata["title"] = core_props.title

        return {"text": text_content, "metadata": metadata}

    def _parse_txt(self, path: Path) -> Dict[str, Any]:
        """Parses a TXT file."""
        metadata = self._get_metadata(path)
        with open(path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        return {"text": text_content, "metadata": metadata}