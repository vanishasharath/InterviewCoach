# backend/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import sys
import traceback

from backend.extractor import (
    extract_from_file,
    extract_information,
    extract_experience_years,
)
from backend.similarity import compute_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", mode="a")
    ],
)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".doc"}
MIN_JD_LENGTH = 50
MAX_JD_LENGTH = 50000

# Initialize FastAPI app
app = FastAPI(
    title="AI Interview Coach API",
    description="Intelligent resume analysis and interview preparation platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "AI Interview Coach Support",
        "email": "support@aiinterviewcoach.com"
    },
    license_info={
        "name": "MIT",
    }
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        # Add production URLs here
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


# Pydantic Models
class BreakdownModel(BaseModel):
    """Score breakdown model"""
    skill_match: float = Field(..., ge=0, le=100, description="Skill match percentage")
    semantic_match: float = Field(..., ge=0, le=100, description="Semantic match percentage")
    experience_match: float = Field(..., ge=0, le=100, description="Experience match percentage")
    bow_similarity: Optional[float] = Field(None, ge=0, le=100, description="Bag-of-words similarity")


class CategoriesModel(BaseModel):
    """Skill categories model"""
    resume: List[str] = Field(default_factory=list, description="Skills found in resume")
    project: List[str] = Field(default_factory=list, description="Skills from projects")
    certification: List[str] = Field(default_factory=list, description="Skills from certifications")


class AnalysisModel(BaseModel):
    """Detailed analysis model"""
    experience_years: float = Field(default=0.0, ge=0, description="Years of experience")
    skills: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted skills")
    education: Optional[List[str]] = Field(default_factory=list, description="Education information")
    contact_info: Optional[Dict[str, Optional[str]]] = Field(default_factory=dict, description="Contact details")
    total_skills_found: Optional[int] = Field(None, ge=0, description="Total number of skills")


class AnalysisResponse(BaseModel):
    """Main analysis response model"""
    success: bool = Field(..., description="Whether the analysis was successful")
    similarity_score_with_experience: float = Field(..., ge=0, le=100, description="Overall score with experience")
    similarity_score_without_experience: float = Field(..., ge=0, le=100, description="Overall score without experience")
    breakdown: BreakdownModel = Field(..., description="Detailed score breakdown")
    matched_skills: List[str] = Field(default_factory=list, description="Skills that match the job description")
    missing_skills: List[str] = Field(default_factory=list, description="Skills missing from resume")
    matched_skills_count: int = Field(default=0, ge=0, description="Number of matched skills")
    missing_skills_count: int = Field(default=0, ge=0, description="Number of missing skills")
    categories: CategoriesModel = Field(..., description="Categorized skills")
    improvements: List[str] = Field(default_factory=list, description="Improvement suggestions")
    interview_questions: List[str] = Field(default_factory=list, description="Potential interview questions")
    analysis: AnalysisModel = Field(..., description="Detailed resume analysis")
    timestamp: str = Field(..., description="Analysis timestamp")
    message: Optional[str] = Field(None, description="Additional message or status")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "similarity_score_with_experience": 75.5,
                "similarity_score_without_experience": 72.3,
                "breakdown": {
                    "skill_match": 80.0,
                    "semantic_match": 70.0,
                    "experience_match": 75.0,
                    "bow_similarity": 65.0
                },
                "matched_skills": ["python", "django", "postgresql"],
                "missing_skills": ["kubernetes", "mongodb"],
                "matched_skills_count": 3,
                "missing_skills_count": 2,
                "categories": {
                    "resume": ["python", "django"],
                    "project": ["react"],
                    "certification": ["aws certified"]
                },
                "improvements": [
                    "Add these 2 key skills to your resume: kubernetes, mongodb"
                ],
                "interview_questions": [
                    "Describe a challenging project where you used python."
                ],
                "analysis": {
                    "experience_years": 5.0,
                    "skills": {
                        "resume_skills": ["python", "django"],
                        "project_skills": ["react"],
                        "certification_skills": ["aws certified"]
                    },
                    "total_skills_found": 4
                },
                "timestamp": "2024-01-01T12:00:00",
                "message": "Analysis successful"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Utility Functions
def validate_file_extension(filename: str) -> bool:
    """Validate file extension"""
    import os
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(file_bytes: bytes) -> bool:
    """Validate file size"""
    return len(file_bytes) <= MAX_FILE_SIZE


def validate_job_description(jd: str) -> bool:
    """Validate job description"""
    if not jd or not jd.strip():
        return False
    jd_length = len(jd.strip())
    return MIN_JD_LENGTH <= jd_length <= MAX_JD_LENGTH


# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if app.debug else "An unexpected error occurred"
        ).dict()
    )


# API Endpoints
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return HealthResponse(
        status="healthy",
        message="AI Interview Coach API is running",
        version="2.0.0"
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Add more health checks here (database, external services, etc.)
        return HealthResponse(
            status="healthy",
            message="All systems operational",
            version="2.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="Service unhealthy",
                detail=str(e)
            ).dict()
        )


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze resume against job description",
    description="Upload a resume file and job description to get detailed analysis and recommendations"
)
async def analyze_resume(
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    job_description: str = Form(..., description="Job description text"),
):
    """
    Analyze resume against job description
    
    This endpoint accepts a resume file and job description, then returns:
    - Similarity scores
    - Matched and missing skills
    - Improvement suggestions
    - Interview preparation questions
    - Detailed breakdown of all metrics
    """
    try:
        logger.info(f"Starting analysis for file: {file.filename}")
        
        # Validate file extension
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file
        file_bytes = await file.read()
        
        # Validate file size
        if not validate_file_size(file_bytes):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f} MB"
            )
        
        # Validate job description
        if not validate_job_description(job_description):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job description. Must be between {MIN_JD_LENGTH} and {MAX_JD_LENGTH} characters"
            )
        
        # Extract resume text
        logger.info(f"Extracting text from file: {file.filename}")
        resume_text = extract_from_file(file_bytes, file.filename)
        
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract meaningful text from resume. Please ensure the file is not empty or corrupted."
            )
        
        logger.info(f"Successfully extracted {len(resume_text)} characters from resume")
        
        # Extract resume information
        logger.info("Extracting resume information...")
        resume_info = extract_information(resume_text)
        
        if "error" in resume_info:
            logger.warning(f"Error in resume extraction: {resume_info['error']}")
        
        resume_years = resume_info.get("experience_years", 0.0)
        
        # Extract job description experience requirement
        logger.info("Extracting job description requirements...")
        jd_years = extract_experience_years(job_description)
        
        # Extract skills from resume
        resume_skills = resume_info["skills"].get("resume_skills", [])
        project_skills = resume_info["skills"].get("project_skills", [])
        cert_skills = resume_info["skills"].get("certification_skills", [])
        
        logger.info(f"Extracted skills - Resume: {len(resume_skills)}, "
                   f"Project: {len(project_skills)}, Certification: {len(cert_skills)}")
        
        # Extract skills from job description
        jd_info = extract_information(job_description.lower())
        jd_skills = jd_info["skills"].get("resume_skills", [])
        
        logger.info(f"Extracted {len(jd_skills)} skills from job description")
        
        # Calculate similarity
        logger.info("Calculating similarity metrics...")
        similarity_result = compute_similarity(
            resume_skills,
            project_skills,
            cert_skills,
            jd_skills,
            resume_text,
            job_description.lower(),
            resume_years,
            jd_years,
        )
        
        if "error" in similarity_result:
            logger.error(f"Error in similarity calculation: {similarity_result['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to calculate similarity. Please try again."
            )
        
        # Prepare response
        response = AnalysisResponse(
            success=True,
            similarity_score_with_experience=similarity_result["score_with_experience"],
            similarity_score_without_experience=similarity_result["score_without_experience"],
            breakdown=BreakdownModel(
                skill_match=similarity_result["skill_match"],
                semantic_match=similarity_result["semantic_match"],
                experience_match=similarity_result["experience_match"],
                bow_similarity=similarity_result.get("bow_similarity", 0.0)
            ),
            matched_skills=similarity_result["matched_skills"],
            missing_skills=similarity_result["missing_skills"],
            matched_skills_count=similarity_result.get("matched_skills_count", len(similarity_result["matched_skills"])),
            missing_skills_count=similarity_result.get("missing_skills_count", len(similarity_result["missing_skills"])),
            categories=CategoriesModel(
                resume=similarity_result["categories"]["resume"],
                project=similarity_result["categories"]["project"],
                certification=similarity_result["categories"]["certification"]
            ),
            improvements=similarity_result["improvements"],
            interview_questions=similarity_result["interview_questions"],
            analysis=AnalysisModel(
                experience_years=resume_years,
                skills=resume_info["skills"],
                education=resume_info.get("education", []),
                contact_info=resume_info.get("contact_info", {}),
                total_skills_found=resume_info.get("total_skills_found", len(resume_skills))
            ),
            timestamp=datetime.now().isoformat(),
            message="Analysis completed successfully"
        )
        
        logger.info(f"Analysis completed successfully. Overall score: "
                   f"{similarity_result['score_with_experience']:.2f}%")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during analysis: {str(e)}"
        )


@app.post("/analyze-text", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_resume_text(
    resume_text: str = Form(..., min_length=50, description="Resume text content"),
    job_description: str = Form(..., min_length=50, description="Job description text")
):
    """
    Analyze resume text directly (without file upload)
    
    Alternative endpoint for when resume is already in text format
    """
    try:
        logger.info("Starting text-based analysis")
        
        # Validate inputs
        if len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume text too short. Minimum 50 characters required."
            )
        
        if not validate_job_description(job_description):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job description. Must be between {MIN_JD_LENGTH} and {MAX_JD_LENGTH} characters"
            )
        
        # Process as before
        resume_text_lower = resume_text.lower()
        resume_info = extract_information(resume_text_lower)
        resume_years = resume_info.get("experience_years", 0.0)
        jd_years = extract_experience_years(job_description)
        
        resume_skills = resume_info["skills"].get("resume_skills", [])
        project_skills = resume_info["skills"].get("project_skills", [])
        cert_skills = resume_info["skills"].get("certification_skills", [])
        
        jd_info = extract_information(job_description.lower())
        jd_skills = jd_info["skills"].get("resume_skills", [])
        
        similarity_result = compute_similarity(
            resume_skills, project_skills, cert_skills, jd_skills,
            resume_text_lower, job_description.lower(),
            resume_years, jd_years
        )
        
        return AnalysisResponse(
            success=True,
            similarity_score_with_experience=similarity_result["score_with_experience"],
            similarity_score_without_experience=similarity_result["score_without_experience"],
            breakdown=BreakdownModel(
                skill_match=similarity_result["skill_match"],
                semantic_match=similarity_result["semantic_match"],
                experience_match=similarity_result["experience_match"],
                bow_similarity=similarity_result.get("bow_similarity", 0.0)
            ),
            matched_skills=similarity_result["matched_skills"],
            missing_skills=similarity_result["missing_skills"],
            matched_skills_count=similarity_result.get("matched_skills_count", len(similarity_result["matched_skills"])),
            missing_skills_count=similarity_result.get("missing_skills_count", len(similarity_result["missing_skills"])),
            categories=CategoriesModel(
                resume=similarity_result["categories"]["resume"],
                project=similarity_result["categories"]["project"],
                certification=similarity_result["categories"]["certification"]
            ),
            improvements=similarity_result["improvements"],
            interview_questions=similarity_result["interview_questions"],
            analysis=AnalysisModel(
                experience_years=resume_years,
                skills=resume_info["skills"],
                education=resume_info.get("education", []),
                contact_info=resume_info.get("contact_info", {}),
                total_skills_found=resume_info.get("total_skills_found", len(resume_skills))
            ),
            timestamp=datetime.now().isoformat(),
            message="Text analysis completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Lifecycle Events
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 60)
    logger.info("AI Interview Coach Backend Starting...")
    logger.info(f"Version: 2.0.0")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Max File Size: {MAX_FILE_SIZE / (1024*1024):.1f} MB")
    logger.info(f"Allowed Extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info("=" * 60)
    
    # Add any initialization tasks here
    # e.g., database connections, cache warming, etc.
    
    logger.info("AI Interview Coach backend started successfully ✓")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("=" * 60)
    logger.info("AI Interview Coach Backend Shutting Down...")
    
    # Add any cleanup tasks here
    # e.g., close database connections, save state, etc.
    
    logger.info("Backend shutdown complete ✓")
    logger.info("=" * 60)


# Run with: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000