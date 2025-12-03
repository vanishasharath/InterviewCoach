AI-InterviewCoach

📌 Overview

AI-InterviewCoach is an AI-powered resume–JD analysis and interview preparation system that evaluates how well a candidate’s resume matches a job description, identifies missing skills, generates personalized improvement suggestions, and produces tailored interview questions based on gaps.
Built using FastAPI (Python) for the backend and React for the frontend, the system performs NLP-based skill extraction, semantic similarity scoring, and experience-based evaluation to deliver a complete readiness assessment.
It acts like a smart recruiter: analyzing resumes, comparing them with JDs, scoring compatibility, and preparing the user for interviews.

🎯 Objectives

Provide a comprehensive resume vs JD match analysis.
Extract and compare skills from resume, project sections, certifications, and JD.
Compute semantic match using transformer embeddings.
Highlight matched, unmatched, and missing skills.
Generate personalized interview questions based on weaknesses or required skills.
Suggest resume improvements aligned with industry ATS standards.

🧠 Key Features

📄 Resume & JD Matching Engine
Computes two readiness scores:
✔ Score (With Experience)
✔ Score (Without Experience)

Includes individual breakdowns:
Skills Match %
Semantic Match %
Experience Match %

🔍 Skill Extraction & Categorization
Extracts skills from:
Resume,
Projects section,
Certifications section,
Job Description,

Categorizes into:
Resume Skills,
Project Skills,
Certification Skills,

🧮 Semantic Similarity Scoring

Uses SentenceTransformer MiniLM-L6-v2 to compute:
Resume phrase ↔ JD phrase similarity,
Synonym/related-term matching,
Context-based alignment beyond exact keywords,

🎯 Matched & Missing Skills

The system produces:
✔ Matched Skills (present in both Resume & JD),
❌ Missing Skills (required by JD but not found in Resume),

💡 Resume Improvement Suggestions

🗣️ AI-Generated Interview Questions

Generated dynamically based on:
Missing skills,
Weak skill categories,
Unfamiliar technologies,
General behavioral patterns,


🏗️ Architecture Flow

Resume + JD Uploaded → Text Extraction → NLP Skill Extraction → Semantic Similarity → Match Scores → Suggestions → Tailored Interview Questions

User uploads Resume + JD via frontend,
Backend extracts text using PDF/DOCX readers,
spaCy pipeline extracts skills, keywords, entities,
MiniLM sentence-transformer computes semantic vectors,

Cosine similarity produces:
Skills Match,
Semantic Match,
Experience Match,

System identifies:
Resume Skills,
Project Skills,
Certification Skills,
Matched Skills,
Missing Skills,

Improvement suggestions generated,
Interview questions generated from gaps,
Results displayed on frontend

🧰 Technologies & Tools

🔧 Backend

FastAPI – API server,
spaCy NLP – tokenization, phrase extraction,
SentenceTransformer (MiniLM-L6-v2) – embeddings,
NumPy – similarity math,
PyPDF2 / python-docx – Resume/JD parsing,
Uvicorn – server runtime,

🎨 Frontend

React – single-page application,
Axios – API communication,
JavaScript / JSX,

📄 Data
skill_corpus.txt – canonical skills reference list

📂 Project Structure
AI-InterviewCoach/
├── backend/
│   ├── main.py                  # FastAPI backend entrypoint
│   ├── extractor.py             # Resume/JD text extraction (PDF/DOCX)
│   ├── analyzer.py              # Resume-JD evaluation + scoring
│   ├── similarity.py            # Embedding similarity model (MiniLM)
│   ├── questions.py             # Tailored interview question generator
│   └── skill_corpus.txt         # Skills dataset
│
├── frontend/
│   ├── public/                  
│   ├── src/                     # React UI
│   ├── package.json
│   └── package-lock.json
│
├── requirements.txt
└── README.md

🚀 Quickstart

Backend
python3 -m venv .venv,
source .venv/bin/activate,
pip install -r requirements.txt,
python backend/main.py,

Frontend
cd frontend,
npm install,
npm start,

