# backend/similarity.py
# Uses Cohere Embed API instead of local SentenceTransformer to avoid OOM on free tier

import math
import os
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re

import cohere

# Set up logging
logger = logging.getLogger(__name__)

# Configuration
SKILL_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.3
EXPERIENCE_WEIGHT = 0.2

# Cohere client — loaded once, lazily
_cohere_client = None

def get_cohere_client() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY environment variable not set")
        _cohere_client = cohere.Client(api_key)
        logger.info("Cohere client initialized")
    return _cohere_client


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag1 * mag2)))


class SimilarityCalculator:
    """
    Calculate similarity between resume and job description using multiple metrics.
    Uses Cohere Embed API for semantic similarity (no local model needed).
    """

    def __init__(self,
                 skill_weight: float = SKILL_WEIGHT,
                 semantic_weight: float = SEMANTIC_WEIGHT,
                 experience_weight: float = EXPERIENCE_WEIGHT):

        total_weight = skill_weight + semantic_weight + experience_weight
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")

        self.skill_weight = skill_weight / total_weight
        self.semantic_weight = semantic_weight / total_weight
        self.experience_weight = experience_weight / total_weight

        self.stop_words = {
            'a','an','and','are','as','at','be','by','for','from','has','he',
            'in','is','it','its','of','on','that','the','to','was','will','with',
            'this','but','they','have','had','or','if','can','all','we','do',
            'not','you','your'
        }

    def preprocess(self, text: str) -> List[str]:
        if not text:
            return []
        text = re.sub(r"[^a-z0-9\s+#]", " ", text.lower())
        return [w for w in text.split() if len(w) > 2 and w not in self.stop_words]

    def bow_cosine(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        try:
            freq1 = Counter(self.preprocess(text1))
            freq2 = Counter(self.preprocess(text2))
            if not freq1 or not freq2:
                return 0.0
            all_words = set(freq1) | set(freq2)
            dot = sum(freq1[w] * freq2[w] for w in all_words)
            mag1 = math.sqrt(sum(v * v for v in freq1.values()))
            mag2 = math.sqrt(sum(v * v for v in freq2.values()))
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return max(0.0, min(1.0, dot / (mag1 * mag2)))
        except Exception as e:
            logger.error(f"BOW cosine error: {e}")
            return 0.0

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Uses Cohere Embed API — no local model, ~5MB RAM usage."""
        if not text1 or not text2:
            return 0.0
        try:
            # Cohere has a 2048-token limit per text; truncate to be safe
            text1 = text1[:4000]
            text2 = text2[:4000]

            co = get_cohere_client()
            response = co.embed(
                texts=[text1, text2],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            embeddings = response.embeddings
            similarity = cosine_similarity(embeddings[0], embeddings[1])
            logger.debug(f"Semantic similarity: {similarity:.4f}")
            return similarity
        except Exception as e:
            logger.error(f"Cohere semantic similarity error: {e}")
            # Fall back to BOW if Cohere fails
            return self.bow_cosine(text1, text2)

    def calculate_skill_match(self,
                              resume_skills: List[str],
                              jd_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        normalization_map = {
            "reactjs": "react", "react.js": "react", "react native": "react",
            "nodejs": "node.js", "node": "node.js",
            "javascript": "js", "js": "javascript",
            "ci": "ci/cd", "cd": "ci/cd",
            "continuous integration": "ci/cd", "continuous delivery": "ci/cd",
            "ci/cd pipeline": "ci/cd",
            "aws cloud": "aws", "amazon web services": "aws",
            "mongo": "mongodb", "sql database": "sql",
        }

        def normalize(skills):
            return [normalization_map.get(s.lower().strip(), s.lower().strip()) for s in skills]

        resume_set = set(normalize(resume_skills))
        jd_set = set(normalize(jd_skills))

        if not jd_set:
            return 0.0, [], []

        matched = sorted(resume_set & jd_set)
        missing = sorted(jd_set - resume_set)
        ratio = len(matched) / len(jd_set)

        logger.info(f"Skill match: {len(matched)}/{len(jd_set)} ({ratio*100:.1f}%)")
        return ratio, matched, missing

    def calculate_experience_match(self, resume_years: float, jd_years: float) -> float:
        if jd_years <= 0:
            return 100.0
        if resume_years <= 0:
            return 0.0
        return min(resume_years / jd_years, 1.0) * 100

    def generate_improvement_suggestions(self,
                                         missing_skills: List[str],
                                         overall_score: float,
                                         experience_match: float,
                                         skill_match: float,
                                         semantic_match: float) -> List[str]:
        suggestions = []

        if missing_skills:
            n = len(missing_skills)
            if n <= 3:
                suggestions.append(f"Add these {n} key skills to your resume: " + ", ".join(missing_skills))
            else:
                suggestions.append("Prioritize learning these critical skills: " + ", ".join(missing_skills[:5]))
                if n > 5:
                    suggestions.append(f"Consider developing {n - 5} additional skills mentioned in the job description.")

        if semantic_match < 40:
            suggestions.append("Significantly rewrite your resume to better align with the job description.")
        elif semantic_match < 60:
            suggestions.append("Tailor your resume content to better match the job description language.")

        if skill_match < 50 and skill_match >= 30:
            suggestions.append("Consider taking online courses or certifications to acquire missing skills.")
        elif skill_match < 30:
            suggestions.append("There is a significant skill gap. Consider building practical projects first.")

        if experience_match < 50:
            suggestions.append("Gain more relevant experience through internships, freelancing or open-source.")
        elif experience_match < 80:
            suggestions.append("Highlight all relevant experience, including academic or freelance projects.")

        if overall_score >= 80:
            suggestions.append("Excellent match! Apply confidently.")
        elif overall_score >= 60:
            suggestions.append("Good match overall. Address the gaps above.")

        if not suggestions:
            suggestions.append("Your resume shows potential. Review feedback to strengthen it.")

        return suggestions

    def generate_interview_questions(self,
                                     matched_skills: List[str],
                                     missing_skills: List[str],
                                     experience_match: float) -> List[str]:
        questions = []

        for skill in matched_skills[:5]:
            questions.append(f"Describe a challenging project where you used {skill}. What was the outcome?")

        for skill in missing_skills[:3]:
            questions.append(f"This role requires {skill}. Have you worked with it? If not, how will you learn quickly?")

        questions.extend([
            "What are your top 3 technical strengths relevant to this position?",
            "Describe a time you learned a new technology quickly. What was your method?",
            "Tell me about a challenging problem you solved. How did you approach it?",
            "How do you stay updated with software industry trends?",
        ])

        if experience_match < 70:
            questions.append("You may have less experience than required. How do you plan to bridge this gap?")

        questions.extend([
            "What motivates you to apply for this role?",
            "Where do you see yourself in the next 2-3 years?",
        ])

        return questions

    def calculate(self,
                  resume_skills: List[str],
                  project_skills: List[str],
                  cert_skills: List[str],
                  jd_skills: List[str],
                  resume_text: str,
                  jd_text: str,
                  resume_years: float,
                  jd_years: float) -> Dict:
        try:
            logger.info("Starting similarity calculation...")

            all_resume_skills = sorted(set(resume_skills + project_skills + cert_skills))
            jd_skills_set = sorted(set(jd_skills))

            skill_ratio, matched, missing = self.calculate_skill_match(all_resume_skills, jd_skills_set)
            semantic_ratio = self.semantic_similarity(resume_text, jd_text)
            experience_pct = self.calculate_experience_match(resume_years, jd_years)
            bow_sim = self.bow_cosine(resume_text, jd_text)

            score_with_exp = (
                skill_ratio * self.skill_weight +
                semantic_ratio * self.semantic_weight +
                (experience_pct / 100.0) * self.experience_weight
            ) * 100

            score_without_exp = (
                skill_ratio * (self.skill_weight + self.experience_weight) +
                semantic_ratio * self.semantic_weight
            ) * 100

            improvements = self.generate_improvement_suggestions(
                missing, score_with_exp, experience_pct,
                skill_ratio * 100, semantic_ratio * 100
            )
            questions = self.generate_interview_questions(matched, missing, experience_pct)

            logger.info(f"Similarity done. Score: {score_with_exp:.2f}%")
            return {
                "skill_match": round(skill_ratio * 100, 2),
                "semantic_match": round(semantic_ratio * 100, 2),
                "bow_similarity": round(bow_sim * 100, 2),
                "experience_match": round(experience_pct, 2),
                "score_with_experience": round(score_with_exp, 2),
                "score_without_experience": round(score_without_exp, 2),
                "matched_skills": matched,
                "missing_skills": missing,
                "matched_skills_count": len(matched),
                "missing_skills_count": len(missing),
                "total_resume_skills": len(all_resume_skills),
                "total_jd_skills": len(jd_skills_set),
                "categories": {
                    "resume": resume_skills,
                    "project": project_skills,
                    "certification": cert_skills,
                },
                "improvements": improvements,
                "interview_questions": questions,
                "weights_used": {
                    "skill": self.skill_weight,
                    "semantic": self.semantic_weight,
                    "experience": self.experience_weight,
                }
            }

        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}", exc_info=True)
            return self._create_error_result(str(e))

    def _create_error_result(self, error_message: str) -> Dict:
        return {
            "skill_match": 0.0, "semantic_match": 0.0, "bow_similarity": 0.0,
            "experience_match": 0.0, "score_with_experience": 0.0,
            "score_without_experience": 0.0, "matched_skills": [], "missing_skills": [],
            "matched_skills_count": 0, "missing_skills_count": 0,
            "total_resume_skills": 0, "total_jd_skills": 0,
            "categories": {"resume": [], "project": [], "certification": []},
            "improvements": ["Error occurred during analysis. Please try again."],
            "interview_questions": [], "error": error_message
        }


# Module-level instance — no model loading happens here anymore
similarity = SimilarityCalculator()


def compute_similarity(resume_skills: List[str],
                       project_skills: List[str],
                       cert_skills: List[str],
                       jd_skills: List[str],
                       resume_text: str,
                       jd_text: str,
                       resume_years: float,
                       jd_years: float) -> Dict:
    return similarity.calculate(
        resume_skills, project_skills, cert_skills,
        jd_skills, resume_text, jd_text,
        resume_years, jd_years,
    )