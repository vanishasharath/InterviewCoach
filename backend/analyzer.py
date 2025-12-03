# analyzer.py
import re
import logging
from typing import Dict, List, Set, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class ResumeAnalyzer:
    """Analyze resume against job description with improved accuracy and configurability"""
    
    # Configuration constants
    DEFAULT_SKILL_MATCH_THRESHOLD = 0.5
    MIN_RECOMMENDED_WORD_COUNT = 100
    MAX_RECOMMENDED_WORD_COUNT = 1000
    DEFAULT_TOP_KEYWORDS = 20
    DEFAULT_TOP_RECOMMENDATIONS = 5
    
    def __init__(self, 
                 skill_match_threshold: float = DEFAULT_SKILL_MATCH_THRESHOLD,
                 min_word_count: int = MIN_RECOMMENDED_WORD_COUNT,
                 max_word_count: int = MAX_RECOMMENDED_WORD_COUNT):
        """
        Initialize Resume Analyzer
        
        Args:
            skill_match_threshold: Minimum percentage of skills that should match (0.0 to 1.0)
            min_word_count: Minimum recommended word count for resume
            max_word_count: Maximum recommended word count for resume
        """
        self.skill_match_threshold = skill_match_threshold
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        
        # Common technical skills with variations
        self.common_skills = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'cpp', 'c#', 'csharp',
            'ruby', 'go', 'golang', 'rust', 'php', 'swift', 'kotlin', 'scala', 'r',
            'perl', 'matlab', 'dart', 'objective-c',
            
            # Web Technologies
            'html', 'html5', 'css', 'css3', 'react', 'reactjs', 'react.js',
            'angular', 'angularjs', 'vue', 'vuejs', 'vue.js', 'nodejs', 'node.js',
            'express', 'expressjs', 'django', 'flask', 'fastapi', 'spring', 
            'spring boot', 'asp.net', 'next.js', 'nextjs', 'nuxt', 'gatsby',
            'jquery', 'bootstrap', 'tailwind', 'sass', 'webpack', 'vite',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'mongo',
            'redis', 'elasticsearch', 'dynamodb', 'cassandra', 'oracle',
            'sqlite', 'mariadb', 'couchdb', 'neo4j', 'firebase', 'firestore',
            
            # Cloud & DevOps
            'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp',
            'google cloud', 'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab',
            'github', 'terraform', 'ansible', 'ci/cd', 'cicd', 'linux', 'unix',
            'bash', 'shell scripting', 'vagrant', 'nginx', 'apache', 'heroku',
            'vercel', 'netlify', 'cloudflare',
            
            # Data Science & ML
            'machine learning', 'deep learning', 'artificial intelligence',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
            'tableau', 'power bi', 'data analysis', 'data visualization',
            'nlp', 'computer vision', 'neural networks', 'data mining',
            
            # Mobile Development
            'android', 'ios', 'react native', 'flutter', 'xamarin', 'cordova',
            'ionic', 'swift', 'swiftui', 'jetpack compose',
            
            # Testing & QA
            'testing', 'unit testing', 'integration testing', 'junit', 'testng',
            'selenium', 'pytest', 'jest', 'mocha', 'chai', 'cypress', 'puppeteer',
            
            # Version Control & Collaboration
            'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
            
            # Architecture & Design
            'api', 'rest', 'restful', 'graphql', 'microservices', 'soap',
            'websocket', 'grpc', 'architecture', 'design patterns', 'mvc',
            'mvvm', 'solid', 'oop', 'object oriented programming',
            
            # Methodologies
            'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'tdd',
            'test driven development', 'bdd', 'ci/cd', 'continuous integration',
            
            # Project Management
            'jira', 'trello', 'asana', 'confluence', 'slack', 'teams',
            
            # Others
            'blockchain', 'ethereum', 'solidity', 'web3', 'cryptography',
            'security', 'oauth', 'jwt', 'sso', 'ldap', 'rabbitmq', 'kafka',
            'hadoop', 'spark', 'airflow', 'etl'
        }
        
        # Expanded stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
            'has', 'do', 'does', 'did', 'doing', 'would', 'could', 'should',
            'may', 'might', 'can', 'must', 'shall', 'being', 'been', 'were',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'also', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'all', 'any', 'these',
            'those', 'am', 'been', 'being', 'having', 'or', 'if', 'because',
            'while', 'until', 'upon', 'within', 'without'
        }
    
    def normalize_skill(self, skill: str) -> str:
        """
        Normalize skill name for better matching
        
        Args:
            skill: Skill name
            
        Returns:
            Normalized skill name
        """
        # Convert to lowercase and remove special characters for matching
        normalized = skill.lower().strip()
        normalized = re.sub(r'[^\w\s+#.]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def extract_skills(self, text: str) -> Set[str]:
        """
        Extract skills from text with improved matching
        
        Args:
            text: Input text
            
        Returns:
            Set of identified skills
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for skill extraction")
            return set()
        
        text_lower = text.lower()
        found_skills = set()
        
        # Sort skills by length (longest first) to match multi-word skills first
        sorted_skills = sorted(self.common_skills, key=len, reverse=True)
        
        for skill in sorted_skills:
            # Create pattern with word boundaries for exact matching
            # Handle special characters in skill names
            escaped_skill = re.escape(skill)
            # Allow for variations in spacing and special characters
            pattern = r'\b' + escaped_skill.replace(r'\ ', r'[\s\-_]*') + r'\b'
            
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_skills.add(skill)
        
        logger.debug(f"Extracted {len(found_skills)} skills from text")
        return found_skills
    
    def extract_keywords(self, text: str, top_n: int = DEFAULT_TOP_KEYWORDS) -> List[str]:
        """
        Extract important keywords from text with improved filtering
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of important keywords
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for keyword extraction")
            return []
        
        # Preprocess - keep alphanumeric, spaces, +, and #
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s+#]', ' ', text)
        words = text.split()
        
        # Filter stop words and short words
        words = [
            w for w in words 
            if w not in self.stop_words 
            and len(w) > 2 
            and not w.isdigit()
        ]
        
        # Get most common words
        word_freq = Counter(words)
        top_keywords = [word for word, _ in word_freq.most_common(top_n)]
        
        logger.debug(f"Extracted {len(top_keywords)} keywords")
        return top_keywords
    
    def identify_strengths(self, resume_skills: Set[str], jd_skills: Set[str]) -> List[str]:
        """
        Identify matching strengths
        
        Args:
            resume_skills: Skills from resume
            jd_skills: Skills from job description
            
        Returns:
            List of strength statements
        """
        matched_skills = resume_skills & jd_skills
        
        strengths = []
        if matched_skills:
            match_percentage = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0
            strengths.append(
                f"Strong match with {len(matched_skills)} out of {len(jd_skills)} "
                f"required skills ({match_percentage:.1f}% match)"
            )
            
            # Group skills for better presentation
            if len(matched_skills) <= 5:
                skill_list = ", ".join(sorted(matched_skills))
                strengths.append(f"Relevant skills: {skill_list}")
            else:
                top_skills = sorted(list(matched_skills))[:5]
                skill_list = ", ".join(top_skills)
                remaining = len(matched_skills) - 5
                strengths.append(
                    f"Key matching skills: {skill_list} and {remaining} more"
                )
        else:
            strengths.append("Limited direct skill matches found - consider highlighting transferable skills")
        
        return strengths
    
    def identify_weaknesses(self, resume_skills: Set[str], jd_skills: Set[str]) -> List[str]:
        """
        Identify missing skills (weaknesses)
        
        Args:
            resume_skills: Skills from resume
            jd_skills: Skills from job description
            
        Returns:
            List of weakness statements
        """
        missing_skills = jd_skills - resume_skills
        
        weaknesses = []
        if missing_skills:
            if len(missing_skills) <= 5:
                skill_list = ", ".join(sorted(missing_skills))
                weaknesses.append(f"Missing required skills: {skill_list}")
            else:
                top_missing = sorted(list(missing_skills))[:5]
                skill_list = ", ".join(top_missing)
                remaining = len(missing_skills) - 5
                weaknesses.append(
                    f"Missing skills: {skill_list} and {remaining} more"
                )
            
            # Additional context
            if len(missing_skills) > len(jd_skills) * 0.5:
                weaknesses.append(
                    "Significant skill gap detected - consider acquiring or highlighting these skills"
                )
        else:
            weaknesses.append("No major skill gaps identified - excellent match!")
        
        return weaknesses
    
    def generate_recommendations(self, 
                                resume_skills: Set[str], 
                                jd_skills: Set[str],
                                resume_text: str,
                                max_recommendations: int = DEFAULT_TOP_RECOMMENDATIONS) -> List[str]:
        """
        Generate improvement recommendations
        
        Args:
            resume_skills: Skills from resume
            jd_skills: Skills from job description
            resume_text: Full resume text
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of recommendations
        """
        recommendations = []
        missing_skills = jd_skills - resume_skills
        
        # Skill-based recommendations
        if missing_skills:
            top_missing = sorted(list(missing_skills))[:3]
            recommendations.append(
                f"Prioritize these skills: {', '.join(top_missing)}"
            )
            
            if len(missing_skills) > 3:
                recommendations.append(
                    f"Consider adding {len(missing_skills) - 3} additional skills from the job description"
                )
        
        # Length check
        word_count = len(resume_text.split())
        if word_count < self.min_word_count:
            recommendations.append(
                f"Expand your resume with more detailed experience and achievements "
                f"(current: {word_count} words, recommended: {self.min_word_count}+ words)"
            )
        elif word_count > self.max_word_count:
            recommendations.append(
                f"Consider condensing your resume to 1-2 pages "
                f"(current: {word_count} words, recommended: <{self.max_word_count} words)"
            )
        
        # Keyword optimization
        skill_match_ratio = len(resume_skills & jd_skills) / len(jd_skills) if jd_skills else 0
        if skill_match_ratio < self.skill_match_threshold:
            recommendations.append(
                f"Tailor your resume to include more keywords from the job description "
                f"(current match: {skill_match_ratio * 100:.1f}%, target: {self.skill_match_threshold * 100:.0f}%+)"
            )
        
        # Quantifiable achievements
        quantifiable_patterns = [
            r'\d+%',  # Percentages
            r'\d+\+',  # Numbers with plus
            r'\$\d+[kmb]?',  # Dollar amounts
            r'\d+x',  # Multipliers
            r'increased by \d+',
            r'improved by \d+',
            r'reduced by \d+',
            r'grew to \d+',
            r'saved \d+',
            r'generated \d+'
        ]
        
        has_numbers = any(
            re.search(pattern, resume_text.lower()) 
            for pattern in quantifiable_patterns
        )
        
        if not has_numbers:
            recommendations.append(
                "Add quantifiable achievements with metrics "
                "(e.g., 'Improved performance by 30%', 'Led team of 5 developers')"
            )
        
        # Action verbs check
        action_verbs = [
            'developed', 'created', 'designed', 'implemented', 'led', 'managed',
            'achieved', 'improved', 'increased', 'reduced', 'optimized', 'built'
        ]
        has_action_verbs = any(verb in resume_text.lower() for verb in action_verbs)
        
        if not has_action_verbs:
            recommendations.append(
                "Use strong action verbs to describe your accomplishments "
                "(e.g., 'Developed', 'Led', 'Optimized')"
            )
        
        # If no recommendations or too many, provide general advice
        if not recommendations:
            recommendations.append("Your resume is well-aligned with the job requirements - great job!")
        elif len(recommendations) > max_recommendations:
            recommendations = recommendations[:max_recommendations]
            recommendations.append("Additional improvements can be made - review all feedback carefully")
        
        return recommendations
    
    def calculate_match_score(self, resume_skills: Set[str], jd_skills: Set[str]) -> float:
        """
        Calculate overall match score
        
        Args:
            resume_skills: Skills from resume
            jd_skills: Skills from job description
            
        Returns:
            Match score between 0 and 100
        """
        if not jd_skills:
            return 0.0
        
        matched = len(resume_skills & jd_skills)
        total_jd = len(jd_skills)
        
        return round((matched / total_jd) * 100, 2)
    
    def analyze(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Perform complete resume analysis
        
        Args:
            resume_text: Resume text content
            job_description: Job description text
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            logger.info("Starting resume analysis...")
            
            # Validate inputs
            if not resume_text or not resume_text.strip():
                raise ValueError("Resume text is empty")
            if not job_description or not job_description.strip():
                raise ValueError("Job description is empty")
            
            # Extract skills
            resume_skills = self.extract_skills(resume_text)
            jd_skills = self.extract_skills(job_description)
            
            logger.info(f"Found {len(resume_skills)} skills in resume")
            logger.info(f"Found {len(jd_skills)} skills in job description")
            
            # Calculate matches
            matched_skills = resume_skills & jd_skills
            missing_skills = jd_skills - resume_skills
            additional_skills = resume_skills - jd_skills
            
            # Generate insights
            strengths = self.identify_strengths(resume_skills, jd_skills)
            weaknesses = self.identify_weaknesses(resume_skills, jd_skills)
            recommendations = self.generate_recommendations(
                resume_skills, jd_skills, resume_text
            )
            
            # Extract keywords
            resume_keywords = self.extract_keywords(resume_text, top_n=15)
            jd_keywords = self.extract_keywords(job_description, top_n=15)
            
            # Calculate match score
            match_score = self.calculate_match_score(resume_skills, jd_skills)
            
            analysis_result = {
                "match_score": match_score,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "recommendations": recommendations,
                "matched_skills": sorted(list(matched_skills)),
                "missing_skills": sorted(list(missing_skills)),
                "additional_skills": sorted(list(additional_skills)),
                "resume_keywords": resume_keywords,
                "jd_keywords": jd_keywords,
                "skill_match_percentage": match_score,  # Kept for backward compatibility
                "total_resume_skills": len(resume_skills),
                "total_jd_skills": len(jd_skills),
                "total_matched_skills": len(matched_skills),
                "word_count": len(resume_text.split()),
                "analysis_metadata": {
                    "skill_match_threshold": self.skill_match_threshold,
                    "min_word_count": self.min_word_count,
                    "max_word_count": self.max_word_count
                }
            }
            
            logger.info(f"Resume analysis completed successfully - Match Score: {match_score}%")
            return analysis_result
        
        except ValueError as ve:
            logger.error(f"Validation error in resume analysis: {ve}")
            return self._create_error_result(str(ve))
        
        except Exception as e:
            logger.error(f"Unexpected error in resume analysis: {e}", exc_info=True)
            return self._create_error_result(f"Unexpected error: {str(e)}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error result
        
        Args:
            error_message: Error message to include
            
        Returns:
            Error result dictionary
        """
        return {
            "match_score": 0.0,
            "strengths": ["Analysis could not be completed"],
            "weaknesses": ["Error occurred during analysis"],
            "recommendations": ["Please try again with valid input"],
            "matched_skills": [],
            "missing_skills": [],
            "additional_skills": [],
            "resume_keywords": [],
            "jd_keywords": [],
            "skill_match_percentage": 0.0,
            "total_resume_skills": 0,
            "total_jd_skills": 0,
            "total_matched_skills": 0,
            "word_count": 0,
            "error": error_message,
            "analysis_metadata": {
                "skill_match_threshold": self.skill_match_threshold,
                "min_word_count": self.min_word_count,
                "max_word_count": self.max_word_count
            }
        }


# Global instance with default configuration
resume_analyzer = ResumeAnalyzer()


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example data
    sample_resume = """
    Senior Python Developer with 5+ years of experience.
    Skills: Python, Django, Flask, PostgreSQL, Docker, AWS, Git
    Improved application performance by 40%.
    Led a team of 3 developers to deliver projects on time.
    """
    
    sample_jd = """
    Looking for a Python Developer with experience in:
    - Python, Django, Flask
    - PostgreSQL, MongoDB
    - Docker, Kubernetes
    - AWS, CI/CD
    - React for frontend
    """
    
    # Analyze
    analyzer = ResumeAnalyzer()
    result = analyzer.analyze(sample_resume, sample_jd)
    
    print("\n=== Resume Analysis Results ===")
    print(f"Match Score: {result['match_score']}%")
    print(f"\nStrengths:")
    for strength in result['strengths']:
        print(f"  - {strength}")
    print(f"\nWeaknesses:")
    for weakness in result['weaknesses']:
        print(f"  - {weakness}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")