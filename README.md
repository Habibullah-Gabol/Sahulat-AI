
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
- [x] English, Urdu, and Sindhi OCR support
- [x] Page-level extraction metadata
- [x] OCR confidence scores
- [x] Baseline language detection
- [x] Automated tests

### M2 — Multilingual NLP

- [x] M2.1 — Language identification using fastText
- [x] M2.2 — Multilingual text normalization
- [x] M2.3 — Text preprocessing and sentence segmentation
- [x] M2.4 — Multilingual embeddings using BGE-M3
- [x] M2.5 — Semantic similarity
- [x] M2.6 — Language-aware text chunking

**M2.4 verification:**
- BGE-M3 embedding dimension: 1024
- English, Urdu, and Sindhi embedding generation verified.
- Normalized embeddings verified.
- Initial cross-language similarity experiment completed.
- Cross-language experiment is exploratory, not a validated benchmark.

**Automated tests:** 108 passed in the latest full regression run.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| PDF Processing | pypdf, PyMuPDF |
| OCR | Tesseract, pytesseract |
| Image Processing | Pillow |
| Language Identification | fasttext-langdetect |
| Text Normalization | Python, Unicode |
| Text Preprocessing | Rule-based multilingual processing |
| Embeddings | BAAI/bge-m3 |
| Embedding Framework | Sentence Transformers |
| Numerical Computing | NumPy |
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
│   │   │   ├── text_normalizer.py
│   │   │   ├── text_preprocessor.py
│   │   │   └── embedding_service.py
│   │   ├── rag/
│   │   ├── agents/
│   │   ├── evaluation/
│   │   └── services/
│   └── tests/
│       ├── test_pdf_extractor.py
│       ├── test_validators.py
│       ├── test_document_processor.py
│       ├── test_ocr_engine.py
│       ├── test_language_detector.py
│       ├── test_text_normalizer.py
│       ├── test_text_preprocessor.py
│       └── test_embedding_service.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/Habibullah-Gabol/Sahulat-AI.git
cd Sahulat-AI
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

On Ubuntu or Google Colab:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-urd \
    tesseract-ocr-snd
```

Tesseract and its language data are system dependencies and are not installed through `requirements.txt`.

### 4. Run tests

From the project root:

```bash
python -m pytest backend/tests -v
```

### 5. Load the multilingual embedding model

The embedding service uses:

```text
BAAI/bge-m3
```

The model is downloaded from Hugging Face when first loaded. A GPU is recommended for faster embedding generation.

## Engineering Approach

Sahulat AI follows a modular development approach:

1. Understand requirements.
2. Design components.
3. Implement reusable modules.
4. Write and run tests.
5. Document changes.
6. Commit and push verified work.

## Future Development

Planned components include:

- Semantic search and cross-language retrieval.
- Language-aware document chunking.
- Vector database integration.
- Retrieval-Augmented Generation (RAG).
- Evidence-grounded question answering.
- Multilingual response generation.
- Evaluation and reliability testing.

---

**Project Status:** M0, M1, and M2 complete.
