# AI Interview Coach — Resume JD Analyzer + Adaptive Interviewer

An AI-powered full-stack system that evaluates how well a candidate's resume matches a job description and conducts a live adaptive interview that reacts to answers in real time using Cohere LLM.
The system acts like a virtual recruiter — analyzing resumes, identifying skill gaps, scoring semantic similarity, and then dynamically interviewing the candidate based on those gaps.

---
## 🌐 Live Demo
* Frontend :  ai-interview-coach-three-rosy.vercel.app
* Backend API : https://interviewcoach-1-eden.onrender.com  
----
## 🎯 Objectives

* Provide a comprehensive resume vs JD match analysis.
* Extract and compare skills from resume, project sections, certifications, and JD.
* Compute semantic match using transformer embeddings.
* Highlight matched, unmatched, and missing skills.
* Generate personalized interview questions based on weaknesses or required skills.
* Suggest resume improvements aligned with industry ATS standards.
* Simulate a live adaptive interview where the questions adapt based on the candidates answer.
* Provide scorecard and feedback for each answer.
---

## 🚀 Features

### 📄 Resume Analysis

Upload Resume (PDF / DOCX) and paste Job Description
  
Dual scoring system:

* With Experience Score — weighted match including years of experience
* Without Experience Score — pure skill and semantic match

Includes individual breakdowns:

* Skills Match %
* Semantic Match %
* Experience Match %

Skill extraction from Resume, Projects, and Certifications

Semantic similarity using Cohere Embed API (fully cloud-based, no local model loading)

Identifies Matched Skills and Missing Skills

ATS-focused resume improvement suggestions

AI-generated interview questions 

Generated dynamically based on:

* Missing skills,
* Weak skill categories,
* Unfamiliar technologies,
* General behavioral patterns,

### 🎙 Adaptive AI Interviewer

Starts a live interview session based on resume analysis results

Questions adapt in real time based on answer quality:

* Score ≥ 0.75 → harder question on a new topic
* Score 0.4–0.74 → same difficulty, new topic
* Score < 0.4 → hint provided, easier question


Powered by Cohere command-a-03-2025

Never repeats questions — tracks full conversation history

Probes missing skills specifically identified from JD

Full session report with per-question breakdown and average score
---

## ⚙️ How It Works

### Resume Analysis Pipeline

* User uploads Resume + pastes Job Description
* Text extracted from PDF/DOCX using PyPDF2/python-docx
* spaCy NLP pipeline extracts skills and keywords
* Cohere Embed API generates sentence embeddings (via API call, not local model)
* Cosine similarity computes semantic match score
* System identifies matched and missing skills
* Improvement suggestions and interview questions generated
* Results displayed on frontend

### Adaptive Interview Pipeline

* Analysis results passed to interview module
* Session created with missing skills and JD context
* Cohere LLM generates first question based on analysis
* Candidate types answer → submitted to backend
* Cohere evaluates answer: scores 0.0–1.0, generates feedback
* Next question selected based on score and history
* Questions never repeat — full history tracked per session
* After 8 questions or manual end → full report generated
---
## 🧰 Technologies & Tools

### 🔧 Backend

Python + FastAPI – Rest API server

spaCy NLP – tokenization, phrase extraction

Cohere Embed - APISemantic embeddings (cloud-based)

Cohere command-a-03-2025 - Adaptive interview AI + answer evaluation

SentenceTransformer (MiniLM-L6-v2) – embeddings

NumPy – similarity math

PyPDF2 / python-docx – Resume/JD parsing

Uvicorn – server runtime

python-dotenvEnvironment variable management

### 🎨 Frontend

React – single-page application

Axios – API communication

JavaScript / JSX

### 📄 Data

skill_corpus.txt – canonical skills reference list
---

```
AI-InterviewCoach/
├── backend/
│   ├── main.py              # FastAPI entrypoint + all endpoints
│   ├── interviewer.py       # Adaptive interview engine (Cohere)
│   ├── extractor.py         # Resume/JD parsing and NLP
│   ├── analyzer.py          # Scoring and evaluation logic
│   ├── similarity.py        # Semantic similarity (Cohere Embed API)
│   ├── questions.py         # Interview question generator
│   └── skill_corpus.txt     # Skill dataset
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.js                        # Main app + routing
│       └── components/
│           ├── AdaptiveInterviewer.jsx   # Live adaptive interview UI
│           ├── AnalysisResult.jsx        # Resume analysis results
│           ├── AddResumeJS.jsx           # Upload page
│           └── UploadUI.css
│
├── .env                     # API keys (never commit this)
├── requirements.txt
└── README.md
```
---
🚀 Quickstart

Backend

cd backend

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

uvicorn backend.main:app --reload

Frontend

cd frontend

npm install

npm start

## 🧠 Key Concepts Demonstrated

* NLP-based skill extraction with spaCy
* Semantic similarity using Cohere Embed API (cloud inference)
* Resume–JD ATS-style matching system
* Cosine similarity and vector comparison
* Adaptive AI interviewing with real-time LLM evaluation
* Multi-turn conversation management with session state
* Full-stack integration: React + FastAPI
* REST API design with Pydantic validation
* Cloud deployment: Vercel + Render (free tier optimised)
