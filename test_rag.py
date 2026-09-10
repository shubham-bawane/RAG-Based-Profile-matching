import unittest

from job_matcher import extract_must_haves, extract_skills
from resume_rag import HashingEmbedder, chunk_resume, extract_metadata


class ResumeRagTests(unittest.TestCase):
    def test_section_chunks_keep_their_context(self):
        chunks = chunk_resume("Jane Doe\nSkills\nPython, SQL\nExperience\n6 years building ML systems")
        self.assertEqual([section for section, _ in chunks], ["profile", "skills", "experience"])

    def test_metadata_extraction(self):
        metadata = extract_metadata("Jane Doe\nSkills: Python, Machine Learning\n6 years experience\nMaster of Science", "jane")
        self.assertEqual(metadata.name, "Jane Doe")
        self.assertEqual(metadata.experience_years, 6)
        self.assertIn("Python", metadata.skills)

    def test_must_have_parser(self):
        skills, years = extract_must_haves("Must-have Python and AWS with 5+ years of experience")
        self.assertIn("Python", skills)
        self.assertIn("AWS", skills)
        self.assertEqual(years, 5)

    def test_hashing_embedder_is_normalized_and_repeatable(self):
        first, second = HashingEmbedder().embed(["Python machine learning", "Python machine learning"])
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0)

    def test_skill_matching_is_case_insensitive(self):
        self.assertEqual(extract_skills("python and AWS"), ["Python", "AWS"])


if __name__ == "__main__":
    unittest.main()