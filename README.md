
# Sahulat AI

**A multilingual AI-powered personal assistant for understanding documents and simplifying everyday tasks.**

Sahulat AI aims to support English, Urdu, and Sindhi, helping users understand official documents, bills, notices, and administrative information.

## Project Vision

Build an intelligent assistant that can:
- Understand multilingual documents.
- Extract information from digital and scanned PDFs.
- Answer questions using document-grounded evidence.
- Retrieve relevant official information.
- Help users prepare drafts, checklists, and reminders.

## Current Development Status

### M0 — Project Initialization
- [x] Repository structure
- [x] Development environment
- [x] GitHub integration

### M1 — Document Ingestion
- [x] PDF validation
- [x] Digital PDF text extraction
- [x] Scanned PDF OCR
- [x] Mixed PDF processing
- [x] English, Urdu, and Sindhi OCR language packs
- [x] Page-level extraction metadata
- [x] OCR confidence scores
- [x] Baseline language detection
- [x] Automated tests (32 passed)

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| PDF Processing | pypdf, PyMuPDF |
| OCR | Tesseract, pytesseract |
| Image Processing | Pillow |
| Language Detection | Custom baseline detector |
| Testing | pytest, httpx |

## Project Structure

```text
Sahulat-AI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── ingestion/
│   │   │   ├── pdf_extractor.py
│   │   │   ├── validators.py
│   │   │   ├── document_processor.py
│   │   │   ├── ocr_engine.py
│   │   │   └── language_detector.py
│   │   ├── nlp/
│   │   ├── rag/
│   │   ├── agents/
│   │   ├── evaluation/
│   │   └── services/
│   └── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
