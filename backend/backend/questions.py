# backend/questions.py

import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# Enums for better type safety
class QuestionCategory(str, Enum):
    """Question category types"""
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    SYSTEM_DESIGN = "system_design"
    HR_MANAGERIAL = "hr_managerial"
    ROLE_SPECIFIC = "role_specific"


class DifficultyLevel(str, Enum):
    """Difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RoleType(str, Enum):
    """Common role types"""
    DATA_SCIENTIST = "Data Scientist"
    DATA_ANALYST = "Data Analyst"
    DATA_ENGINEER = "Data Engineer"
    ML_ENGINEER = "Machine Learning Engineer"
    BACKEND_DEVELOPER = "Backend Developer"
    FRONTEND_DEVELOPER = "Frontend Developer"
    FULLSTACK_DEVELOPER = "Full Stack Developer"
    MOBILE_DEVELOPER = "Mobile Developer"
    DEVOPS_ENGINEER = "DevOps Engineer"
    CLOUD_ENGINEER = "Cloud Engineer"
    QA_ENGINEER = "QA Engineer"
    PRODUCT_MANAGER = "Product Manager"
    PROJECT_MANAGER = "Project Manager"
    BUSINESS_ANALYST = "Business Analyst"
    FINANCE = "Finance / Accounting"
    HR = "Human Resources"
    MARKETING = "Marketing"
    SALES = "Sales"
    GENERAL = "General"


# Question Banks
BASE_BEHAVIORAL_QUESTIONS = [
    "Tell me about yourself and your background.",
    "Why are you interested in this role and our company?",
    "Describe a challenging situation you faced at work and how you handled it.",
    "Tell me about a time you worked effectively in a team.",
    "What are your greatest strengths and how do they apply to this role?",
    "What is an area you're working to improve?",
    "Where do you see yourself in 3-5 years?",
    "Describe a time when you had to learn something new quickly.",
    "Tell me about a project you're particularly proud of.",
    "How do you handle failure or setbacks?",
]

BASE_HR_MANAGERIAL_QUESTIONS = [
    "How do you handle tight deadlines and pressure?",
    "Tell me about a time you had a conflict with a teammate and how you resolved it.",
    "How do you prioritize tasks when you have multiple responsibilities?",
    "Describe a time when you showed leadership, even if you weren't in a leadership role.",
    "Tell me about a time you had to give difficult feedback to someone.",
    "How do you stay organized and manage your time effectively?",
    "Describe a situation where you had to adapt to significant changes.",
    "Tell me about a time you went above and beyond your job responsibilities.",
]

SYSTEM_DESIGN_TEMPLATES = [
    "How would you design a system to handle {}?",
    "If you had to design a scalable solution for {}, what would your approach be?",
    "How would you ensure reliability and performance in a system that {}?",
    "Walk me through the architecture you would use for {}.",
    "What are the key considerations when designing a system that deals with {}?",
]

TECH_QUESTION_TEMPLATES = {
    DifficultyLevel.EASY: [
        "Explain what {} is in simple terms.",
        "Where have you used {} in your work or projects?",
        "What is your experience with {}?",
        "Can you describe the basics of {}?",
    ],
    DifficultyLevel.MEDIUM: [
        "What are the main advantages and limitations of {}?",
        "Describe a real-world scenario where {} is useful.",
        "How does {} compare to similar technologies or approaches?",
        "Explain when you would choose to use {} over alternatives.",
    ],
    DifficultyLevel.HARD: [
        "What common pitfalls or challenges have you encountered when using {}?",
        "How would you optimize or improve a system that heavily relies on {}?",
        "Describe a complex problem you solved using {}.",
        "What are the performance and scalability considerations when using {}?",
    ]
}

# Role-specific question templates
ROLE_SPECIFIC_QUESTIONS = {
    RoleType.DATA_SCIENTIST: [
        "Walk me through your approach to building a machine learning model from scratch.",
        "How do you handle imbalanced datasets?",
        "Explain a time when your model didn't perform as expected and how you debugged it.",
        "How do you communicate technical findings to non-technical stakeholders?",
    ],
    RoleType.BACKEND_DEVELOPER: [
        "How do you approach API design and versioning?",
        "Explain your strategy for database optimization.",
        "How do you handle authentication and authorization in your applications?",
        "Describe your experience with microservices architecture.",
    ],
    RoleType.FRONTEND_DEVELOPER: [
        "How do you ensure your applications are accessible and responsive?",
        "Explain your approach to state management in modern frontend applications.",
        "How do you optimize frontend performance?",
        "Describe your experience with testing frontend code.",
    ],
    RoleType.DEVOPS_ENGINEER: [
        "How do you approach CI/CD pipeline design?",
        "Explain your strategy for monitoring and alerting in production systems.",
        "How do you handle infrastructure as code?",
        "Describe a time when you had to troubleshoot a critical production issue.",
    ],
    RoleType.PRODUCT_MANAGER: [
        "How do you prioritize features in a product roadmap?",
        "Describe your approach to gathering and analyzing user feedback.",
        "Tell me about a time you had to make a difficult product decision.",
        "How do you work with engineering teams to deliver products?",
    ],
}


class InterviewQuestionGenerator:
    """Generate comprehensive interview questions based on role, skills, and experience"""
    
    def __init__(self):
        self.role_patterns = self._initialize_role_patterns()
    
    def _initialize_role_patterns(self) -> Dict[RoleType, List[str]]:
        """Initialize patterns for role detection"""
        return {
            RoleType.DATA_SCIENTIST: [
                r"data scientist", r"ml scientist", r"machine learning scientist"
            ],
            RoleType.DATA_ANALYST: [
                r"data analyst", r"business intelligence", r"analytics"
            ],
            RoleType.DATA_ENGINEER: [
                r"data engineer", r"etl", r"data pipeline"
            ],
            RoleType.ML_ENGINEER: [
                r"ml engineer", r"machine learning engineer", r"ai engineer"
            ],
            RoleType.BACKEND_DEVELOPER: [
                r"backend", r"back-end", r"server-side", r"api developer"
            ],
            RoleType.FRONTEND_DEVELOPER: [
                r"frontend", r"front-end", r"ui developer", r"react developer", r"angular developer"
            ],
            RoleType.FULLSTACK_DEVELOPER: [
                r"full stack", r"fullstack", r"full-stack"
            ],
            RoleType.MOBILE_DEVELOPER: [
                r"mobile", r"android", r"ios", r"react native", r"flutter"
            ],
            RoleType.DEVOPS_ENGINEER: [
                r"devops", r"sre", r"site reliability", r"infrastructure"
            ],
            RoleType.CLOUD_ENGINEER: [
                r"cloud engineer", r"aws engineer", r"azure engineer", r"gcp engineer"
            ],
            RoleType.QA_ENGINEER: [
                r"qa", r"quality assurance", r"test engineer", r"sdet"
            ],
            RoleType.PRODUCT_MANAGER: [
                r"product manager", r"pm", r"product owner"
            ],
            RoleType.PROJECT_MANAGER: [
                r"project manager", r"program manager", r"scrum master"
            ],
            RoleType.BUSINESS_ANALYST: [
                r"business analyst", r"ba", r"systems analyst"
            ],
            RoleType.FINANCE: [
                r"finance", r"accounting", r"financial analyst", r"accountant"
            ],
            RoleType.HR: [
                r"human resources", r"\bhr\b", r"recruiter", r"talent"
            ],
            RoleType.MARKETING: [
                r"marketing", r"digital marketing", r"content marketing"
            ],
            RoleType.SALES: [
                r"sales", r"business development", r"account executive"
            ],
        }
    
    def detect_role_from_jd(self, jd_text: str) -> RoleType:
        """
        Detect role type from job description using pattern matching
        
        Args:
            jd_text: Job description text
            
        Returns:
            Detected RoleType
        """
        if not jd_text:
            logger.warning("Empty job description provided for role detection")
            return RoleType.GENERAL
        
        jd_lower = jd_text.lower()
        
        # Check each role pattern
        for role, patterns in self.role_patterns.items():
            for pattern in patterns:
                if re.search(pattern, jd_lower):
                    logger.info(f"Detected role: {role.value}")
                    return role
        
        logger.info("No specific role detected, using General")
        return RoleType.GENERAL
    
    def extract_key_responsibilities(self, jd_text: str, top_n: int = 5) -> List[str]:
        """
        Extract key responsibilities from job description
        
        Args:
            jd_text: Job description text
            top_n: Number of responsibilities to extract
            
        Returns:
            List of responsibilities
        """
        responsibilities = []
        
        # Common responsibility indicators
        patterns = [
            r"(?:responsibilities|duties|you will|you'll):\s*(.+?)(?:\n\n|\Z)",
            r"•\s*(.+?)(?:\n|$)",
            r"-\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, jd_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                resp = match.group(1).strip()
                if 20 < len(resp) < 200:  # Reasonable length
                    responsibilities.append(resp)
        
        return responsibilities[:top_n]
    
    def generate_answer_template(self,
                                 question: str,
                                 category: QuestionCategory,
                                 difficulty: DifficultyLevel,
                                 skills_context: Dict,
                                 role: RoleType) -> str:
        """
        Generate suggested answer template for a question
        
        Args:
            question: The interview question
            category: Question category
            difficulty: Difficulty level
            skills_context: Context about skills (matched/missing)
            role: Role type
            
        Returns:
            Suggested answer template
        """
        missing = skills_context.get("missing", [])
        matched = skills_context.get("matched", [])
        
        if category == QuestionCategory.BEHAVIORAL:
            return (
                "**Use the STAR method:**\n"
                "- **Situation**: Set the context (1-2 sentences)\n"
                "- **Task**: Describe what you needed to accomplish\n"
                "- **Action**: Explain the specific steps you took\n"
                "- **Result**: Share the outcomes and what you learned\n\n"
                f"Keep your answer relevant to {role.value} and focus on demonstrating "
                "the skills and qualities this role requires."
            )
        
        if category == QuestionCategory.HR_MANAGERIAL:
            return (
                "**Structure your answer:**\n"
                "1. Briefly describe the situation or challenge\n"
                "2. Explain your approach (communication, collaboration, problem-solving)\n"
                "3. Highlight the positive outcome\n"
                "4. Mention what you learned or would do differently\n\n"
                "Focus on soft skills like teamwork, ownership, adaptability, and leadership."
            )
        
        if category == QuestionCategory.SYSTEM_DESIGN:
            return (
                "**Follow this framework:**\n"
                "1. **Clarify requirements**: Ask about scale, constraints, use cases\n"
                "2. **High-level design**: Draw main components and their interactions\n"
                "3. **Deep dive**: Discuss data models, APIs, key algorithms\n"
                "4. **Scale & optimize**: Address bottlenecks, caching, load balancing\n"
                "5. **Trade-offs**: Discuss alternatives and why you chose this approach\n\n"
                f"Use technologies relevant to {role.value} and explain your reasoning."
            )
        
        if category == QuestionCategory.TECHNICAL:
            # Try to detect skill in question
            skill = ""
            for s in matched + missing:
                if s.lower() in question.lower():
                    skill = s
                    break
            
            skill_display = skill or "the concept"
            
            if difficulty == DifficultyLevel.EASY:
                return (
                    f"**Answer structure:**\n"
                    f"1. Define {skill_display} clearly and concisely\n"
                    f"2. Provide 1-2 simple, relatable examples\n"
                    f"3. Mention where you've used it (if applicable)\n\n"
                    "Keep it straightforward and avoid unnecessary jargon."
                )
            
            if difficulty == DifficultyLevel.MEDIUM:
                return (
                    f"**Answer structure:**\n"
                    f"1. Define {skill_display} and its purpose\n"
                    f"2. Explain key benefits and use cases\n"
                    f"3. Discuss limitations or when NOT to use it\n"
                    f"4. Share a specific example from your experience\n\n"
                    "Show depth of understanding while remaining clear."
                )
            
            if difficulty == DifficultyLevel.HARD:
                return (
                    f"**Answer structure:**\n"
                    f"1. Demonstrate deep understanding of {skill_display}\n"
                    f"2. Discuss common pitfalls and how to avoid them\n"
                    f"3. Explain performance/scalability considerations\n"
                    f"4. Share how you've optimized or improved its usage\n"
                    f"5. Compare with alternatives and justify choices\n\n"
                    "Show expert-level knowledge and practical problem-solving skills."
                )
        
        if category == QuestionCategory.ROLE_SPECIFIC:
            return (
                f"**Tailor your answer to {role.value}:**\n"
                "1. Draw from specific experiences in this domain\n"
                "2. Use relevant terminology and best practices\n"
                "3. Demonstrate both technical and business understanding\n"
                "4. Show how your approach aligns with industry standards\n\n"
                "Be specific and back up your claims with concrete examples."
            )
        
        # Fallback
        return (
            "**General answer structure:**\n"
            "1. Start with a clear, concise definition or overview\n"
            "2. Provide a relevant example from your experience\n"
            "3. Explain the impact or results\n"
            "4. Connect it back to the role requirements\n\n"
            "Be specific, quantify when possible, and stay focused."
        )
    
    def make_question_objects(self,
                             questions: List[str],
                             difficulty: DifficultyLevel,
                             category: QuestionCategory,
                             skills_context: Dict,
                             role: RoleType) -> List[Dict]:
        """
        Create structured question objects with metadata
        
        Args:
            questions: List of question strings
            difficulty: Difficulty level
            category: Question category
            skills_context: Skills context
            role: Role type
            
        Returns:
            List of question dictionaries
        """
        result = []
        for question in questions:
            result.append({
                "question": question.strip(),
                "difficulty": difficulty.value,
                "category": category.value,
                "suggested_answer": self.generate_answer_template(
                    question, category, difficulty, skills_context, role
                ),
                "tags": self._generate_tags(question, category, skills_context)
            })
        return result
    
    def _generate_tags(self, question: str, category: QuestionCategory, skills_context: Dict) -> List[str]:
        """Generate relevant tags for a question"""
        tags = [category.value]
        
        # Add skill tags if mentioned in question
        all_skills = skills_context.get("matched", []) + skills_context.get("missing", [])
        for skill in all_skills:
            if skill.lower() in question.lower():
                tags.append(skill.lower())
        
        return tags
    
    def generate_technical_questions(self,
                                    skills: List[str],
                                    skills_context: Dict,
                                    role: RoleType,
                                    questions_per_skill: int = 2) -> List[Dict]:
        """
        Generate technical questions for given skills
        
        Args:
            skills: List of skills
            skills_context: Skills context
            role: Role type
            questions_per_skill: Number of questions per skill per difficulty
            
        Returns:
            List of technical question objects
        """
        all_questions = []
        
        # Limit skills to prevent too many questions
        skills = skills[:10]
        
        for skill in skills:
            # Easy questions
            for template in TECH_QUESTION_TEMPLATES[DifficultyLevel.EASY][:questions_per_skill]:
                all_questions.append({
                    "text": template.format(skill),
                    "difficulty": DifficultyLevel.EASY,
                    "skill": skill
                })
            
            # Medium questions
            for template in TECH_QUESTION_TEMPLATES[DifficultyLevel.MEDIUM][:questions_per_skill]:
                all_questions.append({
                    "text": template.format(skill),
                    "difficulty": DifficultyLevel.MEDIUM,
                    "skill": skill
                })
            
            # Hard questions (fewer)
            for template in TECH_QUESTION_TEMPLATES[DifficultyLevel.HARD][:1]:
                all_questions.append({
                    "text": template.format(skill),
                    "difficulty": DifficultyLevel.HARD,
                    "skill": skill
                })
        
        # Convert to question objects
        easy_objs = self.make_question_objects(
            [q["text"] for q in all_questions if q["difficulty"] == DifficultyLevel.EASY],
            DifficultyLevel.EASY,
            QuestionCategory.TECHNICAL,
            skills_context,
            role
        )
        
        medium_objs = self.make_question_objects(
            [q["text"] for q in all_questions if q["difficulty"] == DifficultyLevel.MEDIUM],
            DifficultyLevel.MEDIUM,
            QuestionCategory.TECHNICAL,
            skills_context,
            role
        )
        
        hard_objs = self.make_question_objects(
            [q["text"] for q in all_questions if q["difficulty"] == DifficultyLevel.HARD],
            DifficultyLevel.HARD,
            QuestionCategory.TECHNICAL,
            skills_context,
            role
        )
        
        return easy_objs + medium_objs + hard_objs
    
    def generate_system_design_questions(self,
                                        skills: List[str],
                                        role: RoleType,
                                        skills_context: Dict,
                                        num_questions: int = 5) -> List[Dict]:
        """
        Generate system design questions
        
        Args:
            skills: List of skills
            role: Role type
            skills_context: Skills context
            num_questions: Number of questions to generate
            
        Returns:
            List of system design question objects
        """
        questions = []
        
        # Only generate for technical roles
        technical_roles = [
            RoleType.BACKEND_DEVELOPER,
            RoleType.FULLSTACK_DEVELOPER,
            RoleType.DATA_ENGINEER,
            RoleType.ML_ENGINEER,
            RoleType.DEVOPS_ENGINEER,
            RoleType.CLOUD_ENGINEER,
        ]
        
        if role not in technical_roles:
            return []
        
        # Use top skills or generic topics
        topics = skills[:3] if skills else ["a scalable web application", "real-time data processing"]
        
        for topic in topics:
            for template in SYSTEM_DESIGN_TEMPLATES[:2]:
                questions.append(template.format(topic))
        
        # Limit to requested number
        questions = questions[:num_questions]
        
        return self.make_question_objects(
            questions,
            DifficultyLevel.HARD,
            QuestionCategory.SYSTEM_DESIGN,
            skills_context,
            role
        )
    
    def generate_interview_questions(self,
                                    missing_skills: List[str],
                                    matched_skills: List[str],
                                    jd_text: str,
                                    resume_text: str) -> Dict[str, List[Dict]]:
        """
        Generate comprehensive interview questions
        
        Args:
            missing_skills: Skills missing from resume
            matched_skills: Skills that match
            jd_text: Job description text
            resume_text: Resume text
            
        Returns:
            Dictionary of categorized questions
        """
        try:
            logger.info("Starting interview question generation")
            
            # Detect role
            role = self.detect_role_from_jd(jd_text)
            
            # Combine and deduplicate skills
            skills_combined = list(dict.fromkeys(matched_skills + missing_skills))[:10]
            
            skills_context = {
                "missing": missing_skills,
                "matched": matched_skills,
                "role": role.value,
            }
            
            # 1. Behavioral questions
            logger.debug("Generating behavioral questions")
            behavioral_objs = self.make_question_objects(
                BASE_BEHAVIORAL_QUESTIONS,
                DifficultyLevel.EASY,
                QuestionCategory.BEHAVIORAL,
                skills_context,
                role
            )
            
            # 2. HR/Managerial questions
            logger.debug("Generating HR/managerial questions")
            hr_objs = self.make_question_objects(
                BASE_HR_MANAGERIAL_QUESTIONS,
                DifficultyLevel.MEDIUM,
                QuestionCategory.HR_MANAGERIAL,
                skills_context,
                role
            )
            
            # 3. Technical questions
            logger.debug("Generating technical questions")
            technical_objs = self.generate_technical_questions(
                skills_combined,
                skills_context,
                role,
                questions_per_skill=2
            )
            
            # 4. System design questions
            logger.debug("Generating system design questions")
            system_design_objs = self.generate_system_design_questions(
                skills_combined,
                role,
                skills_context,
                num_questions=5
            )
            
            # 5. Role-specific questions
            logger.debug("Generating role-specific questions")
            role_specific_objs = []
            if role in ROLE_SPECIFIC_QUESTIONS:
                role_specific_objs = self.make_question_objects(
                    ROLE_SPECIFIC_QUESTIONS[role],
                    DifficultyLevel.MEDIUM,
                    QuestionCategory.ROLE_SPECIFIC,
                    skills_context,
                    role
                )
            
            result = {
                "behavioral": behavioral_objs,
                "technical": technical_objs,
                "system_design": system_design_objs,
                "hr_managerial": hr_objs,
                "role_specific": role_specific_objs,
                "metadata": {
                    "role": role.value,
                    "total_questions": (
                        len(behavioral_objs) +
                        len(technical_objs) +
                        len(system_design_objs) +
                        len(hr_objs) +
                        len(role_specific_objs)
                    ),
                    "skills_analyzed": len(skills_combined)
                }
            }
            
            logger.info(f"Generated {result['metadata']['total_questions']} questions for {role.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating interview questions: {e}", exc_info=True)
            return {
                "behavioral": [],
                "technical": [],
                "system_design": [],
                "hr_managerial": [],
                "role_specific": [],
                "error": str(e)
            }


# Global instance
question_generator = InterviewQuestionGenerator()


# Convenience function
def generate_interview_questions(missing_skills: List[str],
                                matched_skills: List[str],
                                jd_text: str,
                                resume_text: str) -> Dict[str, List[Dict]]:
    """
    Generate interview questions using the global generator
    
    Args:
        missing_skills: Skills missing from resume
        matched_skills: Skills that match
        jd_text: Job description text
        resume_text: Resume text
        
    Returns:
        Dictionary of categorized questions
    """
    return question_generator.generate_interview_questions(
        missing_skills,
        matched_skills,
        jd_text,
        resume_text
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test data
    matched = ["python", "django", "postgresql", "docker"]
    missing = ["kubernetes", "react", "mongodb"]
    
    jd = """
    We're looking for a Backend Developer with 3-5 years of experience.
    You will design and build scalable APIs, work with databases, and deploy to cloud.
    Required: Python, Django, PostgreSQL, Docker, Kubernetes
    """
    
    resume = "Software engineer with 4 years experience in Python and Django"
    
    # Generate questions
    questions = generate_interview_questions(missing, matched, jd, resume)
    
    print("\n=== Interview Questions Generated ===")
    print(f"Role: {questions['metadata']['role']}")
    print(f"Total Questions: {questions['metadata']['total_questions']}")
    
    print(f"\n--- Behavioral ({len(questions['behavioral'])}) ---")
    for q in questions['behavioral'][:2]:
        print(f"  Q: {q['question']}")
        print(f"  Difficulty: {q['difficulty']}")
    
    print(f"\n--- Technical ({len(questions['technical'])}) ---")
    for q in questions['technical'][:3]:
        print(f"  Q: {q['question']}")
        print(f"  Difficulty: {q['difficulty']}")
    
    print(f"\n--- System Design ({len(questions['system_design'])}) ---")
    for q in questions['system_design'][:2]:
        print(f"  Q: {q['question']}")