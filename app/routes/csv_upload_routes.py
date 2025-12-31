from flask import Blueprint, request, jsonify
import pandas as pd
from services.csv_processor import process_csv_file

csv_bp = Blueprint("csv_upload", __name__)

@csv_bp.route("/upload-csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    df = pd.read_csv(file)

    results = process_csv_file(df)

    return jsonify({
        "rows_processed": len(df),
        "results": results
    })
