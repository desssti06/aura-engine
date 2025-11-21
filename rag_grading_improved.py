import os
import json
import numpy as np
import faiss
import requests
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ==========================================================
# CONFIGURATION 
# ==========================================================
class Config:
    # API KEYS & ENDPOINTS
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_API_URL = os.getenv("GEMINI_API_URL")

    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
    OPENROUTER_URL = os.getenv("OPENROUTER_URL")

    # MODELS
    MODEL = os.getenv("MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # MOODLE TOKEN
    MOODLE_TOKEN = os.getenv("MOODLE_TOKEN")
    MOODLE_DOWNLOAD_TOKEN = os.getenv("MOODLE_DOWNLOAD_TOKEN") or MOODLE_TOKEN

    # RAG SETTINGS
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))
    TEMPERATURE = float(os.getenv("TEMPERATURE"))

    # LOGGING
    LOG_LEVEL = os.getenv("LOG_LEVEL")
    LOG_FILE = os.getenv("LOG_FILE")

    # FOLDERS
    DATA_FOLDER = os.getenv("DATA_FOLDER")
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
    RUBRIC_FILE = os.getenv("RUBRIC_FILE")

    MIN_CONFIDENCE_THRESHOLD = 0.6

    @classmethod
    def validate(cls):
        errors = []

        required_env = {
            "GEMINI_API_KEY": cls.GEMINI_API_KEY,
            "GEMINI_API_URL": cls.GEMINI_API_URL,
            "MODEL": cls.MODEL,
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL,
            "DATA_FOLDER": cls.DATA_FOLDER,
            "OUTPUT_FOLDER": cls.OUTPUT_FOLDER,
            "RUBRIC_FILE": cls.RUBRIC_FILE,
        }

        for key, value in required_env.items():
            if value is None:
                errors.append(f"❌ Missing ENV variable: {key}")

        if not cls.MOODLE_DOWNLOAD_TOKEN:
            errors.append("❌ Missing ENV variable: MOODLE_TOKEN atau MOODLE_DOWNLOAD_TOKEN.")

        if cls.CHUNK_SIZE <= 0:
            errors.append("❌ CHUNK_SIZE harus > 0.")

        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("❌ CHUNK_OVERLAP tidak boleh lebih besar dari CHUNK_SIZE.")

        return errors


# ==========================================================
# LOGGING SETUP
# ==========================================================
def setup_logging():
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("=" * 60)
    logging.info("RAG AUTO-GRADING SYSTEM — SAFE MODE")
    logging.info("=" * 60)


# ==========================================================
# PDF EXTRACTION
# ==========================================================
class PDFExtractor:
    @staticmethod
    def extract_text_with_metadata(pdf_path: str) -> Dict:
        try:
            reader = PdfReader(pdf_path)
            pages = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page_num": i + 1, "text": text.strip()})

            full_text = "\n".join(p["text"] for p in pages)
            filename = Path(pdf_path).stem

            return {
                "text": full_text,
                "filename": filename,
                "page_count": len(pages),
                "metadata": {"kelompok": filename},
            }

        except Exception as e:
            logging.error(f"❌ PDF Extract Error: {e}")
            return None


# ==========================================================
# RAG ENGINE
# ==========================================================
class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []

    def build_index(self, text: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        self.chunks = splitter.split_text(text)
        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype=np.float32))

    def search(self, query: str, k: int = Config.TOP_K_RETRIEVAL):
        if self.index is None:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(np.array(q_emb, dtype=np.float32), k)

        return [self.chunks[i] for i in I[0] if 0 <= i < len(self.chunks)]

    def search_multi_query(self, queries: List[str], k=3):
        seen, results = set(), []
        for q in queries:
            for chunk in self.search(q, k):
                if chunk not in seen:
                    seen.add(chunk)
                    results.append(chunk)
        return results


# ==========================================================
# GRADING ENGINE
# ==========================================================
class GradingEngine:
    def __init__(self):
        pass

    # BUILD PROMPT
    def _build_prompt(self, rubric_data: Dict, evidence: str) -> str:
        rubric_json = json.dumps(rubric_data, indent=2, ensure_ascii=False)

        return f"""
Anda adalah sistem penilaian otomatis berbasis evidence. Anda hanya boleh memilih level_id dari rubric yang diberikan, Dilarang membuat level_id baru.

--- RUBRIK ---
{rubric_json}

--- EVIDENCE ---
{evidence}

Kembalikan hanya JSON dengan struktur:
{{
  "advancedgradingdata": {{
    "rubric": {{
      "criteria": [
        {{
          "criterionid": <int>,
          "fillings": [
            {{
              "criterionid": <int>,
              "levelid": <int>,
              "remark": "<text>",
              "confidence": <float>
            }}
          ]
        }}
      ]
    }}
  }}
}}
"""

    # CALL LLM — ENV ONLY
    def _call_llm(self, prompt: str) -> str:
        try:
            headers = {"Content-Type": "application/json"}

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
            }

            url = f"{Config.GEMINI_API_URL}?key={Config.GEMINI_API_KEY}"

            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()

            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            logging.error(f"❌ LLM ERROR: {e}")
            return "{}"

    # PARSE JSON
    def _parse(self, text: str) -> Dict:
        try:
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            return json.loads(text)

        except Exception as e:
            logging.error(f"❌ JSON PARSE ERROR: {e}")
            return {
                "advancedgradingdata": {
                    "rubric": {
                        "criteria": [
                            {
                                "criterionid": 0,
                                "fillings": [
                                    {
                                        "criterionid": 0,
                                        "levelid": 0,
                                        "remark": f"JSON tidak valid: {e}",
                                        "confidence": 0.0,
                                    }
                                ],
                            }
                        ]
                    }
                }
            }

    # MAIN GRADING
    def grade_document(self, rag_engine: RAGEngine, rubric: Dict):
        criteria = rubric["rubric"]["rubric_criteria"]
        evidence_map = {}

        for crit in criteria:
            query = [f"Cari evidence tentang {crit['description']}"]
            evidence_map[str(crit["id"])] = rag_engine.search_multi_query(query)

        evidence_text = "\n".join(sum(evidence_map.values(), []))
        prompt = self._build_prompt(rubric, evidence_text)
        response = self._call_llm(prompt)

        return self._parse(response)


# ==========================================================
# BATCH PROCESSOR
# ==========================================================
class BatchProcessor:
    def __init__(self):
        self.extractor = PDFExtractor()
        self.grader = GradingEngine()

    def process_folder(self, folder: str, rubric_file: str):
        with open(rubric_file, "r", encoding="utf-8") as f:
            rubric_data = json.load(f)

        pdfs = list(Path(folder).glob("*.pdf"))
        results = []

        for pdf in tqdm(pdfs, desc="Processing PDFs"):
            extracted = self.extractor.extract_text_with_metadata(str(pdf))
            if not extracted:
                continue

            rag = RAGEngine()
            rag.build_index(extracted["text"])

            result = self.grader.grade_document(rag, rubric_data)
            result["document_info"] = extracted

            results.append(result)

        return results


# ==========================================================
# MAIN
# ==========================================================
def main():
    setup_logging()

    errors = Config.validate()
    if errors:
        for e in errors:
            print(e)
        return

    os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

    processor = BatchProcessor()
    results = processor.process_folder(Config.DATA_FOLDER, Config.RUBRIC_FILE)

    out_path = os.path.join(Config.OUTPUT_FOLDER, "hasil_grading.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"👉 Saved to: {out_path}")


if __name__ == "__main__":
    main()
