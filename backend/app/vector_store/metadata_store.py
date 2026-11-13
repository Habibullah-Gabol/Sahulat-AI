
import sqlite3
from pathlib import Path
from contextlib import contextmanager


SUPPORTED_LANGUAGES = {"english", "urdu", "sindhi"}


class SQLiteMetadataStore:
    """Store and retrieve document chunk metadata using SQLite."""

    def __init__(self, db_path="data/vector_store/metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path)

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self):
        """Create the chunks table if it does not exist."""

        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    vector_id INTEGER PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    language TEXT NOT NULL,
                    text TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    CHECK (chunk_index >= 0),
                    CHECK (page_number IS NULL OR page_number >= 1),
                    CHECK (language IN ('english', 'urdu', 'sindhi')),
                    CHECK (char_count >= 0)
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id)
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_language
                ON chunks(language)
            """)

    @staticmethod
    def _validate_chunk(vector_id, chunk):
        """Validate vector ID and chunk metadata."""

        if not isinstance(vector_id, int) or isinstance(vector_id, bool):
            raise TypeError("vector_id must be an integer.")

        if vector_id < 0:
            raise ValueError("vector_id cannot be negative.")

        if not isinstance(chunk, dict):
            raise TypeError("chunk must be a dictionary.")

        document_id = chunk.get("source_id")
        chunk_index = chunk.get("chunk_index")
        page_number = chunk.get("page_number")
        language = chunk.get("language")
        text = chunk.get("text")

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("source_id must be a non-empty string.")

        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
        ):
            raise ValueError("chunk_index must be a non-negative integer.")

        if page_number is not None:
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
            ):
                raise ValueError("page_number must be a positive integer.")

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")

        return (
            vector_id,
            document_id.strip(),
            chunk_index,
            page_number,
            language,
            text,
            len(text)
        )

    def add_chunk(self, vector_id: int, chunk: dict) -> None:
        """Insert one document chunk."""

        record = self._validate_chunk(vector_id, chunk)

        with self._connect() as connection:
            connection.execute("""
                INSERT INTO chunks (
                    vector_id,
                    document_id,
                    chunk_index,
                    page_number,
                    language,
                    text,
                    char_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, record)

    def get_chunk(self, vector_id: int) -> dict | None:
        """Retrieve one chunk using its FAISS vector ID."""

        if not isinstance(vector_id, int) or isinstance(vector_id, bool):
            raise TypeError("vector_id must be an integer.")

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute("""
                SELECT *
                FROM chunks
                WHERE vector_id = ?
            """, (vector_id,)).fetchone()

        return dict(row) if row else None

    def get_chunks(
        self,
        document_id: str | None = None,
        language: str | None = None,
        page_number: int | None = None
    ) -> list[dict]:
        """Retrieve chunks using optional metadata filters."""

        conditions = []
        parameters = []

        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("document_id must be a non-empty string.")

            conditions.append("document_id = ?")
            parameters.append(document_id.strip())

        if language is not None:
            if language not in SUPPORTED_LANGUAGES:
                raise ValueError("Unsupported language.")

            conditions.append("language = ?")
            parameters.append(language)

        if page_number is not None:
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
            ):
                raise ValueError("page_number must be a positive integer.")

            conditions.append("page_number = ?")
            parameters.append(page_number)

        query = "SELECT * FROM chunks"

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY document_id, page_number, chunk_index, vector_id"

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                query,
                parameters
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_chunk(self, vector_id: int) -> bool:
        """Delete one metadata record. Returns whether a row was deleted."""

        if not isinstance(vector_id, int) or isinstance(vector_id, bool):
            raise TypeError("vector_id must be an integer.")

        with self._connect() as connection:
            cursor = connection.execute("""
                DELETE FROM chunks
                WHERE vector_id = ?
            """, (vector_id,))

        return cursor.rowcount > 0

    @property
    def total_chunks(self) -> int:
        """Return the number of metadata records."""

        with self._connect() as connection:
            result = connection.execute("""
                SELECT COUNT(*) FROM chunks
            """).fetchone()

        return result[0]
