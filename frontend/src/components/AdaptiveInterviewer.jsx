import "./AdaptiveInterviewer.css";
import React, { useState } from "react";
import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL;



export default function AdaptiveInterview({ result, jobDescription }) {
  const [phase, setPhase] = useState("idle");
  const [sessionId, setSessionId] = useState(null);
  const [currentQ, setCurrentQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [qCount, setQCount] = useState(0);

  const MAX_Q = 8;

  const firstQ =
    result?.interview_questions?.[0] ||
    "Tell me about yourself and your relevant experience.";

  const startInterview = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/interview/start`, {
        jd: jobDescription || "Not provided",
        resume: JSON.stringify(result?.analysis || {}),
        missing_skills: result?.missing_skills || [],
        first_question: firstQ,
      });

      setSessionId(res.data.session_id);
      setCurrentQ(res.data.question);
      setPhase("active");
      setQCount(1);
    } catch (err) {
      alert("Backend not running");
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async () => {
    if (!answer.trim()) return;

    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/interview/respond`, {
        session_id: sessionId,
        answer: answer.trim(),
      });

      const entry = {
        question: currentQ,
        answer: answer,
        score: res.data.score,
        feedback: res.data.feedback,
        difficulty: res.data.difficulty,
      };

      setHistory([...history, entry]);
      setFeedback(res.data);
      setAnswer("");

      if (qCount >= MAX_Q) {
        setPhase("done");
      } else {
        setCurrentQ(res.data.next_question);
        setQCount(qCount + 1);
      }
    } catch {
      alert("Error submitting answer");
    } finally {
      setLoading(false);
    }
  };

  const restart = () => {
    setPhase("idle");
    setHistory([]);
    setAnswer("");
    setFeedback(null);
    setQCount(0);
  };

  // ================= IDLE =================
  if (phase === "idle") {
    return (
      <div className="container">
        <h2>Adaptive AI Interviewer</h2>

        <p className="subtitle">
          AI asks questions based on your resume gaps.
        </p>

        {result?.missing_skills?.length > 0 && (
          <div className="question-box">
            <strong>Skills to be tested: </strong>
            {result.missing_skills.join(", ")}
          </div>
        )}

        <button onClick={startInterview} className="button-primary">
          {loading ? "Starting..." : "Start Interview"}
        </button>
      </div>
    );
  }

  // ================= DONE =================
  if (phase === "done") {
    return (
      <div className="container">
        <h2>Interview Complete</h2>

        {history.map((h, i) => (
          <div key={i} className="history-card">
            <b>Q{i + 1}:</b> {h.question}
            <p><b>Your answer:</b> {h.answer}</p>
            <p><b>Score:</b> {Math.round(h.score * 100)}%</p>
            <p className="feedback">{h.feedback}</p>
          </div>
        ))}

        <button onClick={restart} className="button-primary">
          Restart
        </button>
      </div>
    );
  }

  // ================= ACTIVE =================
  return (
    <div className="container">
      <h2>Live Interview ({qCount}/{MAX_Q})</h2>

      <div className="question-box">{currentQ}</div>

      {feedback && (
        <div className="feedback-box">
          <b>Score:</b> {Math.round(feedback.score * 100)}%
          <p>{feedback.feedback}</p>
        </div>
      )}

      <textarea
        className="textarea"
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        placeholder="Type your answer..."
      />

      <button onClick={submitAnswer} className="button-primary">
        {loading ? "Checking..." : "Submit Answer"}
      </button>
    </div>
  );
}