// frontend/src/App.js
import React, { useState } from "react";
import axios from "axios";
import AnalysisResult from "./components/AnalysisResult.jsx";
import UploadPage from "./components/AddResumeJS.jsx";

function App() {
  const [file, setFile] = useState(null);
  const [jobDesc, setJobDesc] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleSubmit = async () => {
    console.log("Analyze button clicked");

    if (!file || !jobDesc) {
      alert("Please upload a resume and paste job description.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("job_description", jobDesc);

    try {
      const res = await axios.post("http://127.0.0.1:8000/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("API Response:", res.data);
      setAnalysisResult(res.data);
    } catch (err) {
      console.error("AXIOS ERROR FULL DETAILS:", err);
      if (err.response) {
        console.error("Backend Response Error:", err.response.data);
      }
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

      {analysisResult && <AnalysisResult result={analysisResult} />}
    </div>
  );
}

export default App;
