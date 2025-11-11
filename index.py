import argparse
import json
import tempfile
import httpx

from rag_grading_improved import PDFExtractor, RAGEngine, GradingEngine, Config


def process_grading(assignment_id: str, user_id: str, assignment_url: str, rubric_data: dict):
    # Tambah token via query parameter
    token = "0f087ba729c44eb31235146842c67101"
    file_url = f"{assignment_url}?token={token}"

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
    grader = GradingEngine(Config)
    grading_result = grader.grade_document(rag_engine, rubric_data)

    return {
        "assignment_id": assignment_id,
        "user_id": user_id,
        "advancedgradingdata": grading_result.get("advancedgradingdata", {}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Auto Grading CLI")
    parser.add_argument("--input", required=True, help="JSON input string")
    args = parser.parse_args()

    try:
        data = json.loads(args.input)
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
