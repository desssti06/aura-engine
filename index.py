import argparse
import json
import sys
import tempfile
import hashlib
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse, urljoin

import httpx

from rag_grading_improved import PDFExtractor, RAGEngine, GradingEngine, Config


def _append_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params[key] = value
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _resolve_submission_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme:
        return url

    base = Config.MOODLE_BASE_URL
    if not base:
        raise ValueError("Base URL Moodle tidak tersedia untuk membentuk submission_url absolut.")

    return urljoin(base.rstrip("/") + "/", url.lstrip("/"))


def process_grading(assignment_id: str, user_id: str, assignment_url: str, rubric_data: dict):
    try:
        assignment_url = _resolve_submission_url(assignment_url)
    except Exception as e:
        return {"error": f"Gagal menyiapkan URL submission: {e}"}

    token = Config.MOODLE_DOWNLOAD_TOKEN
    if not token:
        return {"error": "Environment variable MOODLE_DOWNLOAD_TOKEN is not set."}

    file_url = _append_query_param(assignment_url, "token", token)

    # 1. Unduh PDF
    doc_id = None
    try:
        with httpx.Client() as client:
            response = client.get(file_url, timeout=60)
            response.raise_for_status()

            pdf_bytes = response.content
            doc_id = hashlib.md5(pdf_bytes).hexdigest()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_bytes)
                pdf_path = temp_pdf.name

    except Exception as e:
        return {"error": f"Gagal mengunduh file PDF: {e}"}

    # 2. Ekstraksi
    extracted = PDFExtractor.extract_text_with_metadata(pdf_path)
    if not extracted or not extracted.get("text", "").strip():
        return {"error": "Gagal membaca isi PDF / PDF kosong"}

    if not doc_id:
        return {"error": "Gagal menghasilkan doc_id dokumen"}

    # 3. Build RAG
    rag_engine = RAGEngine()
    rag_engine.build_index(
        text=extracted["text"],
        doc_id=doc_id,
        metadata={"filename": extracted.get("filename")},
    )

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
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input", help="JSON input string")
    input_group.add_argument("--input-file", help="Path ke file JSON input")
    args = parser.parse_args()

    raw_payload = None
    if args.input:
        raw_payload = args.input
    elif args.input_file:
        try:
            with open(args.input_file, "r", encoding="utf-8") as fh:
                raw_payload = fh.read()
        except Exception as e:
            print(json.dumps({"error": f"Gagal membaca file input: {e}"}))
            exit(1)
    else:
        if sys.stdin is not None and not sys.stdin.isatty():
            raw_payload = sys.stdin.read()

    if not raw_payload:
        parser.error("Berikan payload JSON via --input, --input-file, atau stdin")

    try:
        data = json.loads(raw_payload)
    except Exception as e:
        print(json.dumps({"error": f"Gagal parsing input JSON: {e}"}))
        exit(1)

    integration_config = data.get("integration_config") or {}
    Config.apply_overrides(integration_config)

    # Ambil field (dukung camelCase maupun snake_case)
    user_id = data.get("userId") or data.get("user_id")
    assignment_id = data.get("assignmentId") or data.get("assignment_id")
    assignment_url = (
        data.get("assignmentUrl")
        or data.get("assignment_url")
        or data.get("submission_url")
    )
    rubric_data = data.get("rubric")

    if not (user_id and assignment_id and assignment_url and rubric_data):
        print(json.dumps({"error": "Field wajib (userId/assignmentId/assignmentUrl/rubric) tidak lengkap"}))
        exit(1)

    if rubric_data and "rubric" not in rubric_data and "rubric_criteria" in rubric_data:
        # Normalisasi agar kompatibel dengan struktur yang diharapkan grader
        rubric_data = {**{k: v for k, v in rubric_data.items() if k != "rubric_criteria"}, "rubric": {"rubric_criteria": rubric_data["rubric_criteria"]}}

    result = process_grading(
        assignment_id=str(assignment_id),
        user_id=str(user_id),
        assignment_url=assignment_url,
        rubric_data=rubric_data,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
