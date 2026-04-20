# analyzer.py
import re
import logging
from typing import Dict, List, Set, Any
from collections import Counter

logger = logging.getLogger(__name__)


class ResumeAnalyzer:

    def __init__(self):
        self.common_skills = {
            "python", "java", "javascript", "c", "c++", "cpp",
            "react", "node", "nodejs", "express",
            "django", "flask", "fastapi",
            "sql", "mysql", "postgresql", "mongodb",
            "aws", "docker", "kubernetes",
            "machine learning", "deep learning", "ai",
            "nlp", "computer vision",
            "html", "css",
            "git", "github",
            "rest", "api", "microservices",
            "system design","computer"
        }

        self.stop_words = {
            "the", "is", "in", "and", "to", "of", "for", "on", "with",
            "a", "an", "this", "that"
        }

    # ✅ FIXED: inside class
    def extract_skills(self, text: str) -> Set[str]:
        if not text or not text.strip():
            return set()

        text = text.lower()
        found_skills = set()

        for skill in self.common_skills:
            if skill in text:
                found_skills.add(skill)

        return found_skills

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()

        words = [w for w in words if w not in self.stop_words and len(w) > 2]

        freq = Counter(words)
        return [w for w, _ in freq.most_common(top_n)]

    def analyze(self, resume_text: str, job_description: str) -> Dict[str, Any]:

        if not resume_text or not job_description:
            return {"error": "Missing input"}

        # 🔥 Extract skills
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(job_description)

        # 🔍 DEBUG (remove later if you want)
        print("RESUME SKILLS:", resume_skills)
        print("JD SKILLS:", jd_skills)

        # ✅ Matching logic
        matched_skills = resume_skills & jd_skills
        missing_skills = jd_skills - resume_skills

        print("MISSING SKILLS:", missing_skills)

        # Score
        score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0

        return {
            "match_score": round(score, 2),
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "total_resume_skills": len(resume_skills),
            "total_jd_skills": len(jd_skills),
            "total_matched_skills": len(matched_skills),
        }


# global instance
resume_analyzer = ResumeAnalyzer()