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
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent")

    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
    OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

    MODEL = os.getenv("MODEL", "google/gemini-2.0-flash")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

    MIN_CONFIDENCE_THRESHOLD = 0.6
    TEMPERATURE = 0.0

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "rag_system.log")

    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output"
    RUBRIC_FILE = "data/rubrik.json"

    @classmethod
    def validate(cls):
        errors = []
        if not (cls.OPENROUTER_KEY or cls.GEMINI_API_KEY):
            errors.append("Tidak ditemukan API key (OPENROUTER_KEY atau GEMINI_API_KEY).")
        if cls.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE harus > 0.")
        if cls.CHUNK_OVERLAP < 0 or cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP harus antara 0 dan CHUNK_SIZE.")
        return errors


# ==========================================================
# LOGGING SETUP
# ==========================================================
def setup_logging():
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(Config.LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("=" * 60)
    logging.info("RAG AUTO-GRADING SYSTEM v3.0 - Session Started")
    logging.info("=" * 60)


# ==========================================================
# PDF EXTRACTOR
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
            text = "\n".join([p["text"] for p in pages])
            filename = Path(pdf_path).stem
            return {
                "text": text,
                "filename": filename,
                "page_count": len(pages),
                "metadata": {"kelompok": filename},
            }
        except Exception as e:
            logging.error(f"Error extracting PDF {pdf_path}: {e}")
            return None


# ==========================================================
# RAG ENGINE
# ==========================================================
class RAGEngine:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, text: str, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = splitter.split_text(text)
        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype=np.float32))

    def search(self, query: str, k: int = Config.TOP_K_RETRIEVAL) -> List[str]:
        if not self.index:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(np.array(q_emb, dtype=np.float32), k)
        return [self.chunks[i] for i in I[0] if i < len(self.chunks)]

    def search_multi_query(self, queries: List[str], k: int = 3) -> List[str]:
        seen, results = set(), []
        for q in queries:
            for r in self.search(q, k):
                if r not in seen:
                    seen.add(r)
                    results.append(r)
        return results


# ==========================================================
# GRADING ENGINE
# ==========================================================
class GradingEngine:
    def __init__(self, config: Config = Config()):
        self.config = config

    def grade_document(self, rag_engine: RAGEngine, rubric_data: Dict) -> Dict:
        criteria = rubric_data.get("rubric", {}).get("rubric_criteria", [])
        evidence_map = {}
        for crit in criteria:
            queries = [f"Cari bagian terkait {crit.get('description', '')}"]
            evidence = rag_engine.search_multi_query(queries)
            evidence_map[str(crit.get("id"))] = evidence

        evidence_text = "\n".join(sum(evidence_map.values(), []))
        prompt = self._build_prompt(rubric_data, evidence_text)
        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _build_prompt(self, rubric_data: Dict, evidence: str) -> str:
        rubric_json = json.dumps(rubric_data, indent=2, ensure_ascii=False)
        return f"""
Anda adalah sistem penilaian otomatis.
Gunakan hanya evidence berikut untuk menilai dokumen berdasarkan rubrik.

--- RUBRIK ---
{rubric_json}

--- EVIDENCE ---
{evidence}

Keluarkan hasil dalam format JSON persis berikut:

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
              "remark": "<penjelasan hasil dan skor>",
              "confidence": <float antara 0-1>
            }}
          ]
        }}
      ]
    }}
  }}
}}

Hanya keluarkan JSON, tanpa teks tambahan.
"""

    def _call_llm(self, prompt: str) -> str:
        try:
            if self.config.GEMINI_API_KEY:
                headers = {"Content-Type": "application/json"}
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(
                    f"{self.config.GEMINI_API_URL}?key={self.config.GEMINI_API_KEY}",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                # debug logging agar tahu isi respons mentah
                logging.debug(f"Raw Gemini response: {json.dumps(data, indent=2)}")

                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("Response Gemini kosong / tidak berisi kandidat output.")
                return candidates[0]["content"]["parts"][0].get("text", "{}")

            elif self.config.OPENROUTER_KEY:
                headers = {"Authorization": f"Bearer {self.config.OPENROUTER_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": self.config.MODEL,
                    "messages": [
                        {"role": "system", "content": "Kamu hanya menjawab dengan JSON valid."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.config.TEMPERATURE,
                }
                resp = requests.post(self.config.OPENROUTER_URL, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            else:
                raise ValueError("Tidak ada API key yang ditemukan.")

        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            return "{}"

    def _parse_response(self, response: str) -> Dict:
        try:
            if not response or response.strip() == "{}":
                logging.warning("Response kosong dari LLM, menggunakan fallback default.")
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
                                            "remark": "LLM tidak mengembalikan hasil penilaian. Gunakan fallback.",
                                            "confidence": 0.0,
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                }

            if "```" in response:
                response = response.split("```")[1].replace("json", "").strip()


            data = json.loads(response.strip())
            return data

        except json.JSONDecodeError as e:
            logging.error(f"Error parsing LLM response: {e}")
            logging.debug(f"Raw response:\n{response}")
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
                                        "remark": f"Response tidak bisa diparse sebagai JSON. Error: {e}",
                                        "confidence": 0.0,
                                    }
                                ],
                            }
                        ]
                    }
                }
            }


# ==========================================================
# BATCH PROCESSOR
# ==========================================================
class BatchProcessor:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.pdf_extractor = PDFExtractor()
        self.grading_engine = GradingEngine(config)

    def process_folder(self, folder_path: str, rubric_path: str) -> List[Dict]:
        with open(rubric_path, "r", encoding="utf-8") as f:
            rubric_data = json.load(f)
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        results = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            extracted = self.pdf_extractor.extract_text_with_metadata(str(pdf_file))
            if not extracted:
                continue
            rag_engine = RAGEngine()
            rag_engine.build_index(extracted["text"])
            grading_result = self.grading_engine.grade_document(rag_engine, rubric_data)
            grading_result["document_info"] = extracted
            results.append(grading_result)
        return results


# ==========================================================
# MAIN
# ==========================================================
def main():
    setup_logging()
    errors = Config.validate()
    if errors:
        for e in errors:
            print(f"❌ {e}")
        return

    processor = BatchProcessor()
    results = processor.process_folder(Config.DATA_FOLDER, Config.RUBRIC_FILE)

    print("\n" + "=" * 80)
    print("📄 HASIL PENILAIAN (DITAMPILKAN LANGSUNG DI TERMINAL)")
    print("=" * 80)

    for i, result in enumerate(results, start=1):
        print(f"\n📘 Dokumen #{i}: {result.get('document_info', {}).get('filename', 'Unknown')}")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("✅ Semua hasil grading sudah ditampilkan di terminal.")
    print("=" * 80)


if __name__ == "__main__":
    main()
