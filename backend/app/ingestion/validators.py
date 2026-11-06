
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_pdf(file_path: str | Path) -> dict:
    """
    Validate an uploaded PDF document.

    Returns:
        Dictionary containing validation status and file information.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError("File does not exist.")

    if not pdf_path.is_file():
        raise ValueError("Provided path is not a file.")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    file_size = pdf_path.stat().st_size

    if file_size == 0:
        raise ValueError("File is empty.")

    if file_size > MAX_FILE_SIZE:
        raise ValueError("PDF exceeds the maximum size limit of 10 MB.")

    try:
        reader = PdfReader(str(pdf_path), strict=True)

        if reader.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")

        page_count = len(reader.pages)

        if page_count == 0:
            raise ValueError("PDF contains no pages.")

    except PdfReadError as error:
        raise ValueError(f"Invalid PDF file: {error}") from error

    return {
        "valid": True,
        "filename": pdf_path.name,
        "file_size": file_size,
        "page_count": page_count,
        "file_type": "application/pdf"
    }
