import argparse
import json
import tempfile
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from rag_grading_improved import PDFExtractor, RAGEngine, GradingEngine, Config


def _append_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params[key] = value
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def process_grading(assignment_id: str, user_id: str, assignment_url: str, rubric_data: dict):
    token = Config.MOODLE_DOWNLOAD_TOKEN
    if not token:
        return {"error": "Environment variable MOODLE_DOWNLOAD_TOKEN is not set."}

    file_url = _append_query_param(assignment_url, "token", token)

    # 1. Unduh PDF
    try:
        with httpx.Client() as client:
            response = client.get(file_url, timeout=60)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(response.content)
                pdf_path = temp_pdf.name

    except Exception as e:
        return {"error": f"Gagal mengunduh file PDF: {e}"}

    # 2. Ekstraksi
    extracted = PDFExtractor.extract_text_with_metadata(pdf_path)
    if not extracted or not extracted.get("text", "").strip():
        return {"error": "Gagal membaca isi PDF / PDF kosong"}

    # 3. Build RAG
    rag_engine = RAGEngine()
    rag_engine.build_index(extracted["text"])

    # 4. Grading
    grader = GradingEngine()
    grading_result = grader.grade_document(rag_engine, rubric_data)

    return {
        "assignment_id": assignment_id,
        "user_id": user_id,
        "advancedgradingdata": grading_result.get("advancedgradingdata", {}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Auto Grading CLI")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="JSON input string")
    input_group.add_argument("--input-file", help="Path ke file JSON input")
    args = parser.parse_args()

    raw_payload = args.input
    if args.input_file:
        try:
            with open(args.input_file, "r", encoding="utf-8") as fh:
                raw_payload = fh.read()
        except Exception as e:
            print(json.dumps({"error": f"Gagal membaca file input: {e}"}))
            exit(1)

    try:
        data = json.loads(raw_payload)
    except Exception as e:
        print(json.dumps({"error": f"Gagal parsing input JSON: {e}"}))
        exit(1)

    # Ambil field
    user_id = data.get("userId")
    assignment_id = data.get("assignmentId")
    assignment_url = data.get("assignmentUrl")
    rubric_data = data.get("rubric")

    if not (user_id and assignment_id and assignment_url and rubric_data):
        print(json.dumps({"error": "Field wajib (userId, assignmentId, assignmentUrl, rubric) tidak lengkap"}))
        exit(1)

    result = process_grading(
        assignment_id=assignment_id,
        user_id=user_id,
        assignment_url=assignment_url,
        rubric_data=rubric_data,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
