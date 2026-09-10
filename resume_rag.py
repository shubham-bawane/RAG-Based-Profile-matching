"""Ingest resume files into ChromaDB for retrieval-augmented job matching."""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


KNOWN_SKILLS = (
    "Python", "Java", "JavaScript", "TypeScript", "SQL", "AWS", "Azure",
    "GCP", "Docker", "Kubernetes", "Machine Learning", "Deep Learning",
    "Data Analysis", "React", "Node.js", "Spark", "TensorFlow", "PyTorch",
)
SECTION_HEADINGS = ("summary", "experience", "education", "skills", "projects", "certifications")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HuggingFaceEmbedder:
    """Local semantic embeddings; the model is downloaded on its first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class HashingEmbedder:
    """Dependency-free fallback for tests and demonstrations, not production semantics."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[a-z0-9+#.]+", text.lower()):
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions] += 1
            magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / magnitude for value in vector])
        return vectors


@dataclass(frozen=True)
class ResumeMetadata:
    name: str
    skills: list[str]
    experience_years: int
    education: str

    def as_chroma_metadata(self, resume_path: Path, section: str) -> dict[str, str | int]:
        return {
            "candidate_name": self.name,
            "resume_path": str(resume_path),
            "skills": ", ".join(self.skills),
            "experience_years": self.experience_years,
            "education": self.education,
            "section": section,
        }


def read_resume(path: Path) -> str:
    """Read plain text, Markdown, or PDF resumes from the filesystem."""
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raise ValueError(f"Unsupported resume format: {path.suffix}")


def extract_metadata(text: str, fallback_name: str) -> ResumeMetadata:
    lines = [line.strip(" #-\t") for line in text.splitlines() if line.strip()]
    labelled_name = re.search(r"(?im)^name\s*[:|-]\s*(.+)$", text)
    name = labelled_name.group(1).strip() if labelled_name else (lines[0] if lines else fallback_name)
    skills = [skill for skill in KNOWN_SKILLS if re.search(rf"(?i)(?<!\w){re.escape(skill)}(?!\w)", text)]
    year_matches = re.findall(r"(?i)(\d+)\s*\+?\s*years?", text)
    education_match = re.search(r"(?im)^.*\b(B\.?S\.?|M\.?S\.?|Ph\.?D\.?|Bachelor|Master|University|College)\b.*$", text)
    return ResumeMetadata(name, skills, max(map(int, year_matches), default=0), education_match.group(0).strip() if education_match else "Not specified")


def chunk_resume(text: str, chunk_size: int = 900) -> list[tuple[str, str]]:
    """Keep section context with each chunk, splitting only oversized sections."""
    chunks: list[tuple[str, str]] = []
    current_section = "profile"
    current_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        for start in range(0, len(content), chunk_size):
            if content[start : start + chunk_size].strip():
                chunks.append((current_section, content[start : start + chunk_size]))

    for line in text.splitlines():
        normalized = line.strip().strip("#:").lower()
        if normalized in SECTION_HEADINGS:
            flush()
            current_lines = []
            current_section = normalized
        else:
            current_lines.append(line)
    flush()
    return chunks


def create_collection(database_path: Path, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(database_path))
    return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})


def ingest_resumes(input_directory: Path, database_path: Path, collection_name: str, embedder: Embedder) -> int:
    collection = create_collection(database_path, collection_name)
    files = [path for path in input_directory.rglob("*") if path.suffix.lower() in {".txt", ".md", ".pdf"}]
    documents: list[str] = []
    ids: list[str] = []
    metadata: list[dict[str, str | int]] = []
    for path in files:
        text = read_resume(path)
        candidate = extract_metadata(text, path.stem.replace("_", " ").title())
        for index, (section, chunk) in enumerate(chunk_resume(text)):
            documents.append(chunk)
            ids.append(hashlib.sha256(f"{path.resolve()}:{index}".encode()).hexdigest())
            metadata.append(candidate.as_chroma_metadata(path, section))
    if documents:
        collection.upsert(ids=ids, documents=documents, metadatas=metadata, embeddings=embedder.embed(documents))
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest resumes into ChromaDB.")
    parser.add_argument("--input", type=Path, default=Path("resumes"))
    parser.add_argument("--database", type=Path, default=Path(".chroma-semantic"))
    parser.add_argument("--collection", default="resumes")
    parser.add_argument("--embedder", choices=("huggingface", "hashing"), default="huggingface")
    args = parser.parse_args()
    embedder: Embedder = HashingEmbedder() if args.embedder == "hashing" else HuggingFaceEmbedder()
    print(f"Indexed {ingest_resumes(args.input, args.database, args.collection, embedder)} resume chunks.")


if __name__ == "__main__":
    main()
