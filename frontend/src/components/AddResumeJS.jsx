import React, { useState } from "react";
import { FaUpload } from "react-icons/fa";
import "./UploadUI.css";

const UploadPage = ({ onUploadResume, onUploadJD, onAnalyze }) => {

  const [resumeName, setResumeName] = useState(null);

  return (
    <div className="upload-container">
      <h2 className="upload-title">Upload Resume & Job Description</h2>

      <div className="upload-sections">
        {/* Resume Upload */}
        <div
          className="upload-box"
          onClick={() => document.getElementById("resumeInput").click()}
        >
          <FaUpload size={28} className="upload-icon" />
          <p>Upload Resume (PDF / DOCX)</p>
          <span className="file-name">
            {resumeName || "Click to Upload Resume"}
          </span>
          <input
            id="resumeInput"
            type="file"
            accept=".pdf,.doc,.docx"
            style={{ display: "none" }}
            onChange={(e) => {
              setResumeName(e.target.files[0].name);
              onUploadResume(e.target.files[0]);
            }}
          />
        </div>

        {/* Job Description Text Input */}
        <textarea
          className="job-desc-textarea"
          placeholder="Paste Job Description here..."
          onChange={(e) => onUploadJD(e.target.value)}
        />
      </div>

      <button
  className="analyze-button"
  onClick={onAnalyze}
>
  Analyze Match
</button>

    </div>
  );
};

export default UploadPage;
