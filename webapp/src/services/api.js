import axios from "axios";

const API = axios.create({
  baseURL: process.env.REACT_APP_API_URL || "http://localhost:5000",
  timeout: 30000, // 30 second timeout for large files
});

export const uploadCsv = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await API.post("/upload-csv", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  } catch (error) {
    // Handle different error types
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.message || "Upload failed");
    } else if (error.request) {
      // Request made but no response
      throw new Error("No response from server. Please check your connection.");
    } else {
      // Something else went wrong
      throw new Error("An unexpected error occurred");
    }
  }
};