import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
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
    MODEL_PROVIDER = (os.getenv("MODEL_PROVIDER") or "gemini").lower()

    # MOODLE TOKEN
    MOODLE_BASE_URL = os.getenv("MOODLE_BASE_URL")
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
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", str(Path("vector_store").resolve()))

    MIN_CONFIDENCE_THRESHOLD = 0.6

    @classmethod
    def validate(cls):
        errors = []

        required_env = {
            "MODEL": cls.MODEL,
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL,
            "DATA_FOLDER": cls.DATA_FOLDER,
            "OUTPUT_FOLDER": cls.OUTPUT_FOLDER,
            "RUBRIC_FILE": cls.RUBRIC_FILE,
        }

        provider_env = {}

        if cls.MODEL_PROVIDER == "gemini":
            provider_env = {
                "GEMINI_API_KEY": cls.GEMINI_API_KEY,
                "GEMINI_API_URL": cls.GEMINI_API_URL,
            }
        elif cls.MODEL_PROVIDER == "openrouter":
            provider_env = {
                "OPENROUTER_KEY": cls.OPENROUTER_KEY,
                "OPENROUTER_URL": cls.OPENROUTER_URL,
            }
        else:
            errors.append(f"❌ MODEL_PROVIDER tidak dikenali: {cls.MODEL_PROVIDER}")

        required_env.update(provider_env)

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

    @classmethod
    def apply_overrides(cls, integration_config: Optional[Dict]):
        if not integration_config:
            return

        ai_cfg = integration_config.get("ai") or {}

        service = ai_cfg.get("service")
        if service:
            cls.MODEL_PROVIDER = str(service).lower()

        if ai_cfg.get("model"):
            cls.MODEL = ai_cfg["model"]

        temperature = ai_cfg.get("temperature")
        if temperature is not None:
            try:
                cls.TEMPERATURE = float(temperature)
            except (TypeError, ValueError):
                logging.warning("Lewati temperature override tidak valid: %s", temperature)

        provider = cls.MODEL_PROVIDER

        if provider == "gemini":
            if ai_cfg.get("api_key"):
                cls.GEMINI_API_KEY = ai_cfg["api_key"]
            if ai_cfg.get("api_base_url"):
                cls.GEMINI_API_URL = ai_cfg["api_base_url"]
        elif provider == "openrouter":
            if ai_cfg.get("api_key"):
                cls.OPENROUTER_KEY = ai_cfg["api_key"]
            if ai_cfg.get("api_base_url"):
                cls.OPENROUTER_URL = ai_cfg["api_base_url"]

        moodle_cfg = integration_config.get("moodle") or {}
        base_url = moodle_cfg.get("ws_base_url")
        download_token = moodle_cfg.get("download_token")
        ws_token = moodle_cfg.get("ws_token")

        if base_url:
            cls.MOODLE_BASE_URL = base_url
        if ws_token:
            cls.MOODLE_TOKEN = ws_token
        if download_token or ws_token:
            cls.MOODLE_DOWNLOAD_TOKEN = download_token or ws_token


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
def _classify_chunk(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "narrative"

    code_like = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith((
            "for ",
            "if ",
            "while ",
            "def ",
            "class ",
            "public ",
            "private ",
            "int ",
            "float ",
        )):
            code_like += 1
            continue
        if stripped.startswith(("#", "//", "/*")):
            code_like += 1
            continue
        if stripped.endswith((";", "{", "}")):
            code_like += 1

    symbol_ratio = sum(1 for ch in text if ch in "{}();[]=<>" ) / max(len(text), 1)
    if code_like >= max(2, len(lines) // 2) or symbol_ratio > 0.04:
        return "code"
    return "narrative"


class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        persist_dir = Path(Config.VECTOR_STORE_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name="auto-grading-chunks",
            metadata={"source": "pdf-chunks"},
        )
        self._current_doc_id: Optional[str] = None
        self._doc_chunk_count = 0

    def build_index(self, text: str, doc_id: str, metadata: Optional[Dict] = None):
        if not doc_id:
            raise ValueError("doc_id diperlukan untuk indexing")

        self._current_doc_id = doc_id
        self._doc_chunk_count = 0

        existing = self._collection.get(where={"doc_id": {"$eq": doc_id}}, include=["metadatas", "documents"])
        ids = existing.get("ids") if existing else []
        if ids:
            metas = existing.get("metadatas") or []
            if metas and metas[0].get("chunk_type"):
                self._doc_chunk_count = len(ids)
                logging.debug("Reuse Chroma cache untuk %s (%d chunk)", doc_id, self._doc_chunk_count)
                return
            self._collection.delete(where={"doc_id": {"$eq": doc_id}})
            logging.debug("Refresh Chroma cache untuk %s", doc_id)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=[
                "\nclass ",
                "\ndef ",
                "\nfunction ",
                "\npublic ",
                "\nprivate ",
                "\nfor ",
                "\nwhile ",
                "\nif ",
                "\n",
                " ",
            ],
        )
        chunks = splitter.split_text(text)
        if not chunks:
            logging.warning("Tidak ada chunk yang dihasilkan untuk %s", doc_id)
            return

        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)

        base_meta = {k: v for k, v in (metadata or {}).items() if v is not None}
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                **base_meta,
                "doc_id": doc_id,
                "chunk_idx": i,
                "chunk_type": _classify_chunk(chunk),
            })

        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        self._doc_chunk_count = len(chunks)
        logging.debug("Simpan %d chunk untuk %s", self._doc_chunk_count, doc_id)

    def search(self, query: str, k: int = Config.TOP_K_RETRIEVAL, filters: Optional[Dict] = None):
        if not self._current_doc_id or self._doc_chunk_count == 0:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        k = max(1, min(k, self._doc_chunk_count))

        conditions = [{"doc_id": {"$eq": self._current_doc_id}}]
        if filters:
            for key, value in filters.items():
                conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            where=where,
            include=["documents"],
        )

        documents = results.get("documents", [[]])
        if not documents or not documents[0]:
            return []

        return documents[0]

    def search_multi_query(self, queries: List[str], k: int = 3, filters: Optional[Dict] = None):
        seen, results = set(), []
        for q in queries:
            for chunk in self.search(q, k, filters=filters):
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

    _CRITERION_HINTS = {
        6: {
            "filters": {"chunk_type": "code"},
            "extra_queries": [
                "Ambil potongan kode program yang relevan",
                "Cuplikan fungsi atau algoritma",
            ],
        },
        9: {
            "filters": {"chunk_type": "code"},
            "extra_queries": [
                "Cari kode beserta penjelasan baris",
            ],
        },
    }

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
            provider = Config.MODEL_PROVIDER

            if provider == "openrouter":
                if not Config.OPENROUTER_URL or not Config.OPENROUTER_KEY:
                    raise ValueError("OpenRouter credentials are not configured")

                headers = {
                    "Authorization": f"Bearer {Config.OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": Config.MODEL,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": Config.TEMPERATURE,
                }

                resp = requests.post(Config.OPENROUTER_URL, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()

                data = resp.json()
                choices = data.get("choices")
                if not choices:
                    raise ValueError("OpenRouter response missing choices")

                message = choices[0].get("message", {})
                content = message.get("content")
                if not content:
                    raise ValueError("OpenRouter response missing content")

                return content

            if not Config.GEMINI_API_URL or not Config.GEMINI_API_KEY:
                raise ValueError("Gemini credentials are not configured")

            headers = {"Content-Type": "application/json"}

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
            }

            url = f"{Config.GEMINI_API_URL}?key={Config.GEMINI_API_KEY}"

            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()

            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        except requests.HTTPError as http_err:
            detail = http_err.response.text if http_err.response is not None else "<no response body>"
            logging.error(f"❌ LLM ERROR ({Config.MODEL_PROVIDER}): {http_err} | response={detail}")
            return "{}"
        except Exception as e:
            logging.error(f"❌ LLM ERROR ({Config.MODEL_PROVIDER}): {e}")
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
            cid = int(crit["id"])
            hints = self._CRITERION_HINTS.get(cid, {})

            query = [f"Cari evidence tentang {crit['description']}"]
            query.extend(hints.get("extra_queries", []))

            evidence_map[str(cid)] = rag_engine.search_multi_query(
                query,
                filters=hints.get("filters"),
            )

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

            doc_bytes = Path(pdf).read_bytes()
            doc_id = hashlib.md5(doc_bytes).hexdigest()

            rag = RAGEngine()
            rag.build_index(
                text=extracted["text"],
                doc_id=doc_id,
                metadata={"filename": extracted.get("filename")},
            )

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
