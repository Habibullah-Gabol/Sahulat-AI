
from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF document page by page.

    Args:
        file_path: Path to the PDF document.

    Returns:
        A list containing extracted text and page numbers.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError("The provided path is not a file.")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF documents are supported.")

    reader = PdfReader(str(pdf_path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported yet.")

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        extracted_pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    return extracted_pages
