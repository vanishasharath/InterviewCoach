# backend/main.py
import uuid
import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from interviewer import evaluate_and_advance
from backend.extractor import (
    extract_from_file,
    extract_information,
    extract_experience_years,
)
from backend.similarity import compute_similarity

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".doc"}
MIN_JD_LENGTH = 50
MAX_JD_LENGTH = 50000

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Interview Coach API",
    description="Intelligent resume analysis and interview preparation platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "AI Interview Coach Support",
        "email": "support@aiinterviewcoach.com",
    },
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Models — Resume Analysis
# ---------------------------------------------------------------------------

class BreakdownModel(BaseModel):
    skill_match: float = Field(..., ge=0, le=100)
    semantic_match: float = Field(..., ge=0, le=100)
    experience_match: float = Field(..., ge=0, le=100)
    bow_similarity: Optional[float] = Field(None, ge=0, le=100)


class CategoriesModel(BaseModel):
    resume: List[str] = Field(default_factory=list)
    project: List[str] = Field(default_factory=list)
    certification: List[str] = Field(default_factory=list)


class AnalysisModel(BaseModel):
    experience_years: float = Field(default=0.0, ge=0)
    skills: Dict[str, List[str]] = Field(default_factory=dict)
    education: Optional[List[str]] = Field(default_factory=list)
    contact_info: Optional[Dict[str, Optional[str]]] = Field(default_factory=dict)
    total_skills_found: Optional[int] = Field(None, ge=0)


class AnalysisResponse(BaseModel):
    success: bool
    similarity_score_with_experience: float = Field(..., ge=0, le=100)
    similarity_score_without_experience: float = Field(..., ge=0, le=100)
    breakdown: BreakdownModel
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    matched_skills_count: int = Field(default=0, ge=0)
    missing_skills_count: int = Field(default=0, ge=0)
    categories: CategoriesModel
    improvements: List[str] = Field(default_factory=list)
    interview_questions: List[str] = Field(default_factory=list)
    analysis: AnalysisModel
    timestamp: str
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Pydantic Models — Adaptive Interview
# ---------------------------------------------------------------------------

class InterviewStartRequest(BaseModel):
    jd: str = Field(..., min_length=10, description="Job description text")
    resume: str = Field(..., min_length=10, description="Resume text")
    missing_skills: List[str] = Field(default_factory=list, description="Skills to probe")
    first_question: str = Field(..., min_length=5, description="Opening interview question")


class InterviewRespondRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from /interview/start")
    answer: str = Field(..., min_length=1, description="Candidate's answer")


class InterviewStartResponse(BaseModel):
    session_id: str
    question: str


class InterviewRespondResponse(BaseModel):
    feedback: str
    score: float
    follow_up: str
    next_question: str
    difficulty: str
    skill: Optional[str] = None 


class InterviewReportResponse(BaseModel):
    session_id: str
    total_questions: int
    average_score: float
    history: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# In-memory session store  (swap for Redis in production)
# ---------------------------------------------------------------------------
interview_sessions: Dict[str, Dict] = {}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def validate_file_extension(filename: str) -> bool:
    import os
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(file_bytes: bytes) -> bool:
    return len(file_bytes) <= MAX_FILE_SIZE


def validate_job_description(jd: str) -> bool:
    if not jd or not jd.strip():
        return False
    return MIN_JD_LENGTH <= len(jd.strip()) <= MAX_JD_LENGTH


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, detail=str(exc)).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).dict(),
    )


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    return HealthResponse(
        status="healthy",
        message="AI Interview Coach API is running",
        version="2.0.0",
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        return HealthResponse(
            status="healthy",
            message="All systems operational",
            version="2.0.0",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(error="Service unhealthy", detail=str(e)).dict(),
        )


# ---------------------------------------------------------------------------
# Resume analysis endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze resume against job description",
)
async def analyze_resume(
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    job_description: str = Form(..., description="Job description text"),
):
    try:
        logger.info(f"Starting analysis for file: {file.filename}")

        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        file_bytes = await file.read()

        if not validate_file_size(file_bytes):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max: {MAX_FILE_SIZE / (1024*1024):.1f} MB",
            )

        if not validate_job_description(job_description):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job description must be {MIN_JD_LENGTH}–{MAX_JD_LENGTH} characters.",
            )

        resume_text = extract_from_file(file_bytes, file.filename)
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract meaningful text from resume.",
            )

        logger.info(f"Extracted {len(resume_text)} characters from resume")

        resume_info = extract_information(resume_text)
        resume_years = resume_info.get("experience_years", 0.0)
        jd_years = extract_experience_years(job_description)

        resume_skills = resume_info["skills"].get("resume_skills", [])
        project_skills = resume_info["skills"].get("project_skills", [])
        cert_skills = resume_info["skills"].get("certification_skills", [])

        jd_info = extract_information(job_description.lower())
        jd_skills = jd_info["skills"].get("resume_skills", [])

        similarity_result = compute_similarity(
            resume_skills, project_skills, cert_skills, jd_skills,
            resume_text, job_description.lower(),
            resume_years, jd_years,
        )

        if "error" in similarity_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to calculate similarity.",
            )

        response = AnalysisResponse(
            success=True,
            similarity_score_with_experience=similarity_result["score_with_experience"],
            similarity_score_without_experience=similarity_result["score_without_experience"],
            breakdown=BreakdownModel(
                skill_match=similarity_result["skill_match"],
                semantic_match=similarity_result["semantic_match"],
                experience_match=similarity_result["experience_match"],
                bow_similarity=similarity_result.get("bow_similarity", 0.0),
            ),
            matched_skills=similarity_result["matched_skills"],
            missing_skills=similarity_result["missing_skills"],
            matched_skills_count=similarity_result.get("matched_skills_count", len(similarity_result["matched_skills"])),
            missing_skills_count=similarity_result.get("missing_skills_count", len(similarity_result["missing_skills"])),
            categories=CategoriesModel(
                resume=similarity_result["categories"]["resume"],
                project=similarity_result["categories"]["project"],
                certification=similarity_result["categories"]["certification"],
            ),
            improvements=similarity_result["improvements"],
            interview_questions=similarity_result["interview_questions"],
            analysis=AnalysisModel(
                experience_years=resume_years,
                skills=resume_info["skills"],
                education=resume_info.get("education", []),
                contact_info=resume_info.get("contact_info", {}),
                total_skills_found=resume_info.get("total_skills_found", len(resume_skills)),
            ),
            timestamp=datetime.now().isoformat(),
            message="Analysis completed successfully",
        )

        logger.info(f"Analysis done. Score: {similarity_result['score_with_experience']:.2f}%")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during analysis: {str(e)}",
        )


@app.post("/analyze-text", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_resume_text(
    resume_text: str = Form(..., min_length=50),
    job_description: str = Form(..., min_length=50),
):
    """Analyze resume text directly (no file upload needed)"""
    try:
        if not validate_job_description(job_description):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job description must be {MIN_JD_LENGTH}–{MAX_JD_LENGTH} characters.",
            )

        resume_info = extract_information(resume_text.lower())
        resume_years = resume_info.get("experience_years", 0.0)
        jd_years = extract_experience_years(job_description)

        resume_skills = resume_info["skills"].get("resume_skills", [])
        project_skills = resume_info["skills"].get("project_skills", [])
        cert_skills = resume_info["skills"].get("certification_skills", [])

        jd_info = extract_information(job_description.lower())
        jd_skills = jd_info["skills"].get("resume_skills", [])

        similarity_result = compute_similarity(
            resume_skills, project_skills, cert_skills, jd_skills,
            resume_text.lower(), job_description.lower(),
            resume_years, jd_years,
        )

        return AnalysisResponse(
            success=True,
            similarity_score_with_experience=similarity_result["score_with_experience"],
            similarity_score_without_experience=similarity_result["score_without_experience"],
            breakdown=BreakdownModel(
                skill_match=similarity_result["skill_match"],
                semantic_match=similarity_result["semantic_match"],
                experience_match=similarity_result["experience_match"],
                bow_similarity=similarity_result.get("bow_similarity", 0.0),
            ),
            matched_skills=similarity_result["matched_skills"],
            missing_skills=similarity_result["missing_skills"],
            matched_skills_count=similarity_result.get("matched_skills_count", len(similarity_result["matched_skills"])),
            missing_skills_count=similarity_result.get("missing_skills_count", len(similarity_result["missing_skills"])),
            categories=CategoriesModel(
                resume=similarity_result["categories"]["resume"],
                project=similarity_result["categories"]["project"],
                certification=similarity_result["categories"]["certification"],
            ),
            improvements=similarity_result["improvements"],
            interview_questions=similarity_result["interview_questions"],
            analysis=AnalysisModel(
                experience_years=resume_years,
                skills=resume_info["skills"],
                education=resume_info.get("education", []),
                contact_info=resume_info.get("contact_info", {}),
                total_skills_found=resume_info.get("total_skills_found", len(resume_skills)),
            ),
            timestamp=datetime.now().isoformat(),
            message="Text analysis completed successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Adaptive interview endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/interview/start",
    response_model=InterviewStartResponse,
    tags=["Interview"],
    summary="Start a new adaptive interview session",
)
async def start_interview(data: InterviewStartRequest):
    session_id = str(uuid.uuid4())
    interview_sessions[session_id] = {
        "jd": data.jd,
        "resume": data.resume,
        "missing_skills": data.missing_skills,
        "history": [],
        "current_question": data.first_question,
    }
    logger.info(f"Interview session started: {session_id}")
    return InterviewStartResponse(session_id=session_id, question=data.first_question)


@app.post(
    "/interview/respond",
    response_model=InterviewRespondResponse,
    tags=["Interview"],
    summary="Submit answer and get adaptive feedback + next question",
)
async def respond(data: InterviewRespondRequest):
    session = interview_sessions.get(data.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please start a new interview session.",
        )

    session["latest_answer"] = data.answer

    try:
        result = evaluate_and_advance(session)
    except Exception as e:
        logger.error(f"evaluate_and_advance failed for session {data.session_id}: {e}")
        raise HTTPException(status_code=500, detail="AI evaluation failed. Please try again.")

    # CORRECT - all indented properly
    session["history"].append({
        "question": session["current_question"],
        "answer": data.answer,
        "score": result["score"],
        "skill": result.get("skill"),
    })
    session["current_question"] = result["next_question"]

    logger.info(
        f"Session {data.session_id} — score: {result['score']:.2f}, "
        f"difficulty: {result['difficulty']}"
    )
    return InterviewRespondResponse(**result)


@app.get(
    "/interview/report/{session_id}",
    response_model=InterviewReportResponse,
    tags=["Interview"],
    summary="Get full Q&A history and scores for a session",
)
async def get_report(session_id: str):
    session = interview_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    history = session.get("history", [])
    avg_score = (
        round(sum(h["score"] for h in history) / len(history), 2)
        if history else 0.0
    )

    return InterviewReportResponse(
        session_id=session_id,
        total_questions=len(history),
        average_score=avg_score,
        history=history,
    )


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("AI Interview Coach Backend Starting...")
    logger.info(f"Version: 2.0.0")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.1f} MB")
    logger.info(f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info("=" * 60)
    logger.info("Backend started successfully ✓")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 60)
    logger.info("AI Interview Coach Backend Shutting Down...")
    logger.info("Shutdown complete ✓")
    logger.info("=" * 60)

# Run with: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000