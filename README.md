# RAG-Based-Profile-matching

A learning project that matches resumes to a job description using retrieval-augmented generation (RAG) foundations. It uses **Hugging Face embeddings** and a persistent **ChromaDB** vector database.

## The mental model

RAG retrieves trustworthy source text before producing an answer:

```text
resume files -> section-aware chunks -> embeddings -> ChromaDB
job description -> embedding -> nearest resume chunks -> hybrid ranking -> JSON matches
```

- **Chunking** keeps a resume's `Skills`, `Experience`, and `Education` content separate, so a retrieved excerpt has useful context.
- **Embedding** maps text with similar meaning near one another in vector space. `all-MiniLM-L6-v2` runs locally and is downloaded on first use.
- **Vector database** stores each chunk plus metadata. ChromaDB finds chunks close to the job-description vector efficiently.
- **Hybrid search** protects important exact terms: $0.7$ of the score is semantic similarity and $0.3$ is overlap in known skills.

## Setup

```bash
python -m pip install -r requirements.txt
```

Put `.txt`, `.md`, or `.pdf` resumes in `resumes/`. A sample resume is included. Then build the index:

```bash
python resume_rag.py --input resumes
```

Search it:

```bash
python job_matcher.py "Senior ML engineer required: Python, AWS, and 5+ years of experience."
```

The result is JSON with candidate identity, source path, score, skills, retrieved excerpts, and an explanation. The default local database is `.chroma-semantic/`. `--collection` selects a different Chroma collection; `--database` selects a database directory.

Embeddings only have meaning within the model that created them. Do not query a database indexed with `--embedder hashing` using the default Hugging Face embedder, or the reverse. Use a separate database path for the offline lesson:

```bash
python resume_rag.py --input resumes --database .chroma-hashing --embedder hashing
python job_matcher.py "Python developer" --database .chroma-hashing --embedder hashing
```

## Moving to Chroma Cloud later

This project starts with `chromadb.PersistentClient`, which saves everything in the local `.chroma-semantic/` directory and requires no account. When you need a shared, deployed index, create a Chroma Cloud account and replace the local client with Chroma's `CloudClient`, passing its tenant, database, and API key through environment variables. The chunking, embeddings, metadata, and `match_job` logic stay the same; only the database connection changes. Do not commit the API key.

## Learning in order

1. Read `chunk_resume` and change `chunk_size`; re-index to see why chunk boundaries affect the retrieved excerpts.
2. Read `extract_metadata`; it stores scalar values in Chroma metadata because those can be filtered later.
3. Read `match_job`; its candidate score is $100(0.7s + 0.3k)$, where $s$ is the best semantic similarity and $k$ is keyword overlap.
4. Try `--embedder hashing` for repeatable offline experiments. It is a word-hash baseline, not genuine semantic search; use the Hugging Face default for actual matching.

## Verification

Run the fast logic tests without installing dependencies:

```bash
python -m unittest -v test_rag.py
```