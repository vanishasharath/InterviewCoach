// frontend/src/App.js
import React, { useState } from "react";
import axios from "axios";
import AnalysisResult from "./components/AnalysisResult.jsx";
import UploadPage from "./components/AddResumeJS.jsx";
import AdaptiveInterview from "./components/AdaptiveInterviewer.jsx";

const API_BASE_URL = process.env.REACT_APP_API_URL;

function App() {
  const [file, setFile] = useState(null);
  const [jobDesc, setJobDesc] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showInterview, setShowInterview] = useState(false);

  const handleSubmit = async () => {
    if (!file || !jobDesc) {
      alert("Please upload a resume and paste job description.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    formData.append("job_description", jobDesc);
    try {
      const res = await axios.post(`${API_BASE_URL}/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setAnalysisResult(res.data);
      setShowInterview(false);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to API. Check console for details.");
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>AI Resume Analyzer</h1>
      <UploadPage
        onUploadResume={setFile}
        onUploadJD={setJobDesc}
        onAnalyze={handleSubmit}
      />
      {analysisResult && (
        <>
          <AnalysisResult result={analysisResult} />
          {!showInterview && (
            <div style={{ marginTop: 24, textAlign: "center" }}>
              <button
                onClick={() => setShowInterview(true)}
                style={{
                  background: "#4f46e5", color: "#fff", border: "none",
                  borderRadius: 8, padding: "14px 32px", fontSize: 16,
                  fontWeight: 600, cursor: "pointer",
                }}
              >
                🎙 Start Adaptive Interview
              </button>
              <p style={{ color: "#6c757d", fontSize: 13, marginTop: 8 }}>
                AI adapts questions in real time based on your answers
              </p>
            </div>
          )}
          {showInterview && (
            <AdaptiveInterview result={analysisResult} jobDescription={jobDesc} />
          )}
        </>
      )}
    </div>
  );
}

export default App;