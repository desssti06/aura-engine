from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import json
from rag_grading_improved import PDFExtractor, RAGEngine, GradingEngine, Config

app = FastAPI(title="Auto Grading API v1")


@app.post("/api/v1/grade")
async def grade_assignment(
    assignment_id: str = Form(...),
    user_id: str = Form(...),
    rubric: UploadFile = File(...),
    tugas: UploadFile = File(...)
):
    # Read rubric JSON
    rubric_data = json.loads((await rubric.read()).decode("utf-8"))

    # Save PDF temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await tugas.read())
        pdf_path = temp_pdf.name

    # Extract text from PDF
    extracted = PDFExtractor.extract_text_with_metadata(pdf_path)
    if not extracted:
        return {"error": "Gagal membaca file tugas / format PDF rusak"}

    # Build RAG index
    rag_engine = RAGEngine()
    rag_engine.build_index(extracted["text"])

    # Grade
    grader = GradingEngine(Config)
    grading_result = grader.grade_document(rag_engine, rubric_data)

    # **Tambahkan metadata input ke response**
    final_response = {
        "assignment_id": assignment_id,
        "user_id": user_id,
        "advancedgradingdata": grading_result["advancedgradingdata"]  # Tanpa document_info
    }

    return final_response
