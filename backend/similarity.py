# backend/similarity.py

import math
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import re

from sentence_transformers import SentenceTransformer, util
import torch

# Set up logging
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"
SKILL_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.3
EXPERIENCE_WEIGHT = 0.2

# Load sentence transformer model with error handling
try:
    embedder = SentenceTransformer(DEFAULT_MODEL)
    logger.info(f"Loaded SentenceTransformer model: {DEFAULT_MODEL}")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    raise


class SimilarityCalculator:
    """
    Calculate similarity between resume and job description using multiple metrics
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
        
        logger.info(f"Initialized SimilarityCalculator with weights: "
                   f"skill={self.skill_weight:.2f}, "
                   f"semantic={self.semantic_weight:.2f}, "
                   f"experience={self.experience_weight:.2f}")
        
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
        
        words = [
            w for w in text.split() 
            if len(w) > 2 and w not in self.stop_words
        ]
        return words
    
    def bow_cosine(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            logger.warning("Empty text provided for BOW cosine similarity")
            return 0.0
        
        try:
            words1 = self.preprocess(text1)
            words2 = self.preprocess(text2)
            
            if not words1 or not words2:
                return 0.0
            
            freq1 = Counter(words1)
            freq2 = Counter(words2)
            all_words = set(freq1.keys()) | set(freq2.keys())
            
            dot_product = sum(freq1[w] * freq2[w] for w in all_words)
            
            magnitude1 = math.sqrt(sum(v * v for v in freq1.values()))
            magnitude2 = math.sqrt(sum(v * v for v in freq2.values()))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating BOW cosine similarity: {e}")
            return 0.0
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            logger.warning("Empty text provided for semantic similarity")
            return 0.0
        
        try:
            max_length = 5000
            text1 = text1[:max_length]
            text2 = text2[:max_length]
            
            embedding1 = embedder.encode(text1, convert_to_tensor=True)
            embedding2 = embedder.encode(text2, convert_to_tensor=True)
            
            similarity = util.cos_sim(embedding1, embedding2).item()
            
            similarity = max(0.0, min(1.0, similarity))
            
            logger.debug(f"Semantic similarity calculated: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_skill_match(self, 
                            resume_skills: List[str], 
                            jd_skills: List[str]) -> Tuple[float, List[str], List[str]]:

        # Normalization map for equivalent variants
        normalization_map = {
            "reactjs": "react",
            "react.js": "react",
            "react native": "react",
            "nodejs": "node.js",
            "node": "node.js",
            "javascript": "js",
            "js": "javascript",
            "ci": "ci/cd",
            "cd": "ci/cd",
            "continuous integration": "ci/cd",
            "continuous delivery": "ci/cd",
            "ci/cd pipeline": "ci/cd",
            "aws cloud": "aws",
            "amazon web services": "aws",
            "mongo": "mongodb",
            "sql database": "sql",
        }

        def normalize(skills: List[str]) -> List[str]:
            normalized = []
            for skill in skills:
                s = skill.lower().strip()
                normalized.append(normalization_map.get(s, s))
            return normalized
        
        resume_normalized = normalize(resume_skills)
        jd_normalized = normalize(jd_skills)

        if not jd_normalized:
            logger.warning("No job description skills provided")
            return 0.0, [], []
        
        resume_set = set(resume_normalized)
        jd_set = set(jd_normalized)
        
        matched = sorted(list(resume_set & jd_set))
        missing = sorted(list(jd_set - resume_set))
        
        match_ratio = len(matched) / len(jd_set) if jd_set else 0.0
        
        logger.info(f"Skill match: {len(matched)}/{len(jd_set)} ({match_ratio*100:.1f}%)")
        
        return match_ratio, matched, missing
    
    def calculate_experience_match(self, resume_years: float, jd_years: float) -> float:
        if jd_years <= 0:
            return 100.0
        
        if resume_years <= 0:
            return 0.0
        
        match_pct = min(resume_years / jd_years, 1.0) * 100
        return match_pct
    
    def generate_improvement_suggestions(self, 
                                        missing_skills: List[str],
                                        overall_score: float,
                                        experience_match: float,
                                        skill_match: float,
                                        semantic_match: float) -> List[str]:
        suggestions = []
        
        if missing_skills:
            num_missing = len(missing_skills)
            if num_missing <= 3:
                suggestions.append(
                    f"Add these {num_missing} key skills to your resume: "
                    + ", ".join(missing_skills)
                )
            else:
                suggestions.append(
                    f"Prioritize learning these critical skills: "
                    + ", ".join(missing_skills[:5])
                )
                if num_missing > 5:
                    suggestions.append(
                        f"Consider developing {num_missing - 5} additional skills "
                        "mentioned in the job description"
                    )
        
        if semantic_match < 40:
            suggestions.append(
                "Significantly rewrite your resume to better align with the job description."
            )
        elif semantic_match < 60:
            suggestions.append(
                "Tailor your resume content to better match the job description language."
            )
        
        if skill_match < 50 and skill_match >= 30:
            suggestions.append(
                "Consider taking online courses or certifications to acquire missing skills."
            )
        elif skill_match < 30:
            suggestions.append(
                "There is a significant skill gap. Consider building practical projects first."
            )
        
        if experience_match < 50:
            suggestions.append(
                "Gain more relevant experience through internships, freelancing or open-source."
            )
        elif experience_match < 80:
            suggestions.append(
                "Highlight all relevant experience, including academic or freelance projects."
            )
        
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
            questions.append(
                f"Describe a challenging project where you used {skill}. What was the outcome?"
            )
        
        for skill in missing_skills[:3]:
            questions.append(
                f"This role requires {skill}. Have you worked with it? If not, how will you learn quickly?"
            )
        
        questions.extend([
            "What are your top 3 technical strengths relevant to this position?",
            "Describe a time you learned a new technology quickly. What was your method?",
            "Tell me about a challenging problem you solved. How did you approach it?",
            "How do you stay updated with software industry trends?",
        ])
        
        if experience_match < 70:
            questions.append(
                "You may have less experience than required. How do you plan to bridge this gap?"
            )
        
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
            
            all_resume_skills = sorted(list(set(resume_skills + project_skills + cert_skills)))
            jd_skills_set = sorted(list(set(jd_skills)))
            
            skill_match_ratio, matched, missing = self.calculate_skill_match(
                all_resume_skills, jd_skills_set
            )
            
            semantic_ratio = self.semantic_similarity(resume_text, jd_text)
            experience_pct = self.calculate_experience_match(resume_years, jd_years)
            bow_similarity = self.bow_cosine(resume_text, jd_text)
            
            score_with_experience = (
                skill_match_ratio * self.skill_weight +
                semantic_ratio * self.semantic_weight +
                (experience_pct / 100.0) * self.experience_weight
            ) * 100
            
            score_without_experience = (
                skill_match_ratio * (self.skill_weight + self.experience_weight) +
                semantic_ratio * self.semantic_weight
            ) * 100
            
            improvements = self.generate_improvement_suggestions(
                missing,
                score_with_experience,
                experience_pct,
                skill_match_ratio * 100,
                semantic_ratio * 100
            )
            
            questions = self.generate_interview_questions(
                matched, missing, experience_pct
            )
            
            result = {
                "skill_match": round(skill_match_ratio * 100, 2),
                "semantic_match": round(semantic_ratio * 100, 2),
                "bow_similarity": round(bow_similarity * 100, 2),
                "experience_match": round(experience_pct, 2),
                "score_with_experience": round(score_with_experience, 2),
                "score_without_experience": round(score_without_experience, 2),
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
                    "experience": self.experience_weight
                }
            }
            
            logger.info(f"Similarity calculation completed. Overall score: {score_with_experience:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}", exc_info=True)
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> Dict:
        return {
            "skill_match": 0.0,
            "semantic_match": 0.0,
            "bow_similarity": 0.0,
            "experience_match": 0.0,
            "score_with_experience": 0.0,
            "score_without_experience": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "matched_skills_count": 0,
            "missing_skills_count": 0,
            "total_resume_skills": 0,
            "total_jd_skills": 0,
             "categories": {
                "resume": [],
                "project": [],
                "certification": [],
            },
            "improvements": ["Error occurred during analysis. Please try again."],
            "interview_questions": [],
            "error": error_message
        }


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
        resume_skills,
        project_skills,
        cert_skills,
        jd_skills,
        resume_text,
        jd_text,
        resume_years,
        jd_years,
    )
