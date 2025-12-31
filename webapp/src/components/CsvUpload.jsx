import { useState } from "react";
import { uploadCsv } from "../services/api";

export default function CsvUpload({ onResult }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a CSV file");
      return;
    }

    setLoading(true);
    try {
      const data = await uploadCsv(file);
      onResult(data);
    } catch (err) {
      console.error(err);
      alert(err.message || "CSV upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h3>Upload CSV</h3>
      <input
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        disabled={loading}
      />
      <button onClick={handleUpload} disabled={loading || !file}>
        {loading ? "Processing..." : "Upload & Predict"}
      </button>
      {file && !loading && <p>Selected: {file.name}</p>}
    </div>
  );
}