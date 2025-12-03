import React from "react";
import "./AnalysisResult.css";

const AnalysisResult = ({ result }) => {
  if (!result) return <p>No Results Yet</p>;

  const breakdown = result.breakdown || {};
  const categories = result.categories || {};
  const improvements = result.improvements || [];
  const interviewQuestions = result.interview_questions || [];
  const matched = result.matched_skills || [];
  const missing = result.missing_skills || [];

  return (
    <div className="analysis-card">
      <h2>Resume Match Result</h2>

      {/* SCORE SECTION */}
      <div className="score-section">
        <div className="metric-block">
          <div className="metric-header">
            <span>Score (With Experience)</span>
            <span>{(result.score_with_experience || result.similarity_score_with_experience || 0)}%</span>

          </div>
          <div className="progress-container">
            <div
              className="progress-fill"
              style={{ width: `${result.score_with_experience || result.similarity_score_with_experience}%` }}
            />
          </div>
        </div>

        <div className="metric-block">
          <div className="metric-header">
            <span>Score (Without Experience)</span>
            <span>{(result.score_without_experience || result.similarity_score_without_experience || 0)}%</span>
          </div>
          <div className="progress-container">
            <div
              className="progress-fill"
              style={{ width: `${result.score_without_experience || result.similarity_score_without_experience}%` }}
            />
          </div>
        </div>
      </div>

      {/* BREAKDOWN */}
      <h3>Breakdown</h3>
      <div className="score-section">
        <div className="metric-block">
          <div className="metric-header">
            <span>Skills Match</span>
            <span>{breakdown.skill_match}%</span>
          </div>
          <div className="progress-container">
            <div
              className="progress-fill"
              style={{ width: `${breakdown.skill_match}%`, background: "#3b82f6" }}
            />
          </div>
        </div>

        <div className="metric-block">
          <div className="metric-header">
            <span>Semantic Match</span>
            <span>{breakdown.semantic_match}%</span>
          </div>
          <div className="progress-container">
            <div
              className="progress-fill"
              style={{ width: `${breakdown.semantic_match}%`, background: "#a855f7" }}
            />
          </div>
        </div>

        <div className="metric-block">
          <div className="metric-header">
            <span>Experience Match</span>
            <span>{breakdown.experience_match}%</span>
          </div>
          <div className="progress-container">
            <div
              className="progress-fill"
              style={{ width: `${breakdown.experience_match}%`, background: "#f97316" }}
            />
          </div>
        </div>
      </div>

      {/* SKILL CATEGORIES */}
      <h3>Skill Categories</h3>
      <div className="skills-section">
        <p><strong>Resume Skills:</strong></p>
        <div className="badges">
          {(categories.resume || []).map((s, i) => <span key={i} className="badge">{s}</span>)}
        </div>

        <p><strong>Project Skills:</strong></p>
        <div className="badges">
          {(categories.project || []).map((s, i) => <span key={i} className="badge">{s}</span>)}
        </div>

        <p><strong>Certification Skills:</strong></p>
        <div className="badges">
          {(categories.certification || []).map((s, i) => <span key={i} className="badge">{s}</span>)}
        </div>
      </div>

      {/* MATCHED & MISSING */}
      <h3>Matched Skills</h3>
      <div className="badges green-badges">
        {matched.length > 0 ? matched.map((s, i) => <span key={i} className="badge success">{s}</span>) : "None"}
      </div>

      <h3>Missing Skills</h3>
      <div className="badges red-badges">
        {missing.length > 0 ? missing.map((s, i) => <span key={i} className="badge danger">{s}</span>) : "None"}
      </div>

      {/* IMPROVEMENTS */}
      <h3>Improvement Suggestions</h3>
      <ul className="improvements-list">
        {improvements.map((s, i) => <li key={i}>💡 {s}</li>)}
      </ul>

      {/* QUESTIONS */}
      <h3>Interview Questions</h3>
      <ul className="interview-list">
        {interviewQuestions.map((q, i) => <li key={i}>🗣 {q}</li>)}
      </ul>
    </div>
  );
};

export default AnalysisResult;
