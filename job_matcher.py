"""Retrieve, filter, and rank candidate resumes for a job description."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from resume_rag import Embedder, HashingEmbedder, HuggingFaceEmbedder, KNOWN_SKILLS, create_collection


def extract_skills(text: str) -> list[str]:
    return [skill for skill in KNOWN_SKILLS if re.search(rf"(?i)(?<!\w){re.escape(skill)}(?!\w)", text)]


def extract_must_haves(job_description: str) -> tuple[list[str], int]:
    critical_skills = []
    for sentence in re.split(r"[.\n]", job_description):
        if re.search(r"(?i)\b(must[- ]have|required|mandatory)\b", sentence):
            critical_skills.extend(extract_skills(sentence))
    years = [int(year) for year in re.findall(r"(?i)(\d+)\s*\+?\s*years?", job_description)]
    return list(dict.fromkeys(critical_skills)), max(years, default=0)


def match_job(job_description: str, embedder: Embedder, database_path: Path = Path(".chroma-semantic"), collection_name: str = "resumes", top_k: int = 10) -> dict[str, Any]:
    """Combine semantic similarity and explicit skill overlap into a 0-100 ranking."""
    collection = create_collection(database_path, collection_name)
    indexed_chunks = collection.count()
    if indexed_chunks == 0:
        return {"job_description": job_description, "top_matches": []}
    results = collection.query(query_embeddings=embedder.embed([job_description]), n_results=min(top_k * 4, indexed_chunks), include=["documents", "metadatas", "distances"])
    job_skills = extract_skills(job_description)
    must_have_skills, minimum_years = extract_must_haves(job_description)
    candidates: dict[str, dict[str, Any]] = defaultdict(lambda: {"excerpts": [], "sections": set(), "best_similarity": 0.0})

    for document, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        skills = [skill.strip() for skill in metadata["skills"].split(",") if skill.strip()]
        if any(skill not in skills for skill in must_have_skills) or metadata["experience_years"] < minimum_years:
            continue
        key = metadata["resume_path"]
        candidate = candidates[key]
        candidate.update(metadata)
        candidate["excerpts"].append(document[:300])
        candidate["sections"].add(metadata["section"])
        candidate["best_similarity"] = max(candidate["best_similarity"], 1 - float(distance))

    matches = []
    for candidate in candidates.values():
        candidate_skills = [skill.strip() for skill in candidate["skills"].split(",") if skill.strip()]
        matched_skills = [skill for skill in job_skills if skill in candidate_skills]
        keyword_score = len(matched_skills) / max(len(job_skills), 1)
        score = round(100 * (0.7 * max(candidate["best_similarity"], 0) + 0.3 * keyword_score))
        sections = ", ".join(sorted(candidate["sections"]))
        matches.append({
            "candidate_name": candidate["candidate_name"],
            "resume_path": candidate["resume_path"],
            "match_score": score,
            "matched_skills": matched_skills,
            "relevant_excerpts": candidate["excerpts"][:2],
            "reasoning": f"Matched {', '.join(matched_skills) or 'relevant context'} in {sections}; {candidate['experience_years']} years of experience extracted.",
        })
    return {"job_description": job_description, "top_matches": sorted(matches, key=lambda item: item["match_score"], reverse=True)[:top_k]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the best resume matches for a job description.")
    parser.add_argument("job_description", help="Text of the job description")
    parser.add_argument("--database", type=Path, default=Path(".chroma-semantic"))
    parser.add_argument("--collection", default="resumes")
    parser.add_argument("--embedder", choices=("huggingface", "hashing"), default="huggingface")
    args = parser.parse_args()
    embedder: Embedder = HashingEmbedder() if args.embedder == "hashing" else HuggingFaceEmbedder()
    print(json.dumps(match_job(args.job_description, embedder, args.database, args.collection), indent=2))


if __name__ == "__main__":
    main()
