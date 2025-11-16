# 🤖 Aura Engine — AI Auto Grading (RAG)

This repository berisi engine penilaian otomatis berbasis Retrieval-Augmented Generation (RAG) untuk menilai tugas mahasiswa (PDF) berdasarkan rubrik JSON.

README ini diperbarui agar sesuai dengan isi repositori: entrypoint CLI `index.py` dan engine `rag_grading_improved.py`.

---

## Ringkasan

- Input: file PDF tugas dan rubrik (JSON).
- Proses: ekstraksi teks dari PDF → pembuatan index embedding (FAISS) → retrieval → prompt ke LLM untuk penilaian per kriteria.
- Output: struktur JSON `advancedgradingdata` berisi hasil penilaian dan catatan.

---

## Struktur Proyek (ringkas)

```
.
├── index.py                   # Entrypoint CLI untuk memproses satu tugas lewat JSON input
├── rag_grading_improved.py    # Engine RAG, ekstraktor PDF, dan grading engine
├── evaluation_metrics.py      # Utility evaluasi/metric
├── data/                      # Contoh rubrik dan (opsional) PDF untuk batch
│   └── rubrik.json
├── output/                    # Hasil run batch akan disimpan di sini
├── requirements.txt
└── README.md
```

---

## Prasyarat

- Python 3.8+
- Sistem operasi: Windows (instruksi PowerShell disediakan), Linux/Mac juga kompatibel dengan penyesuaian aktivasi venv.

---

## Instalasi (Windows PowerShell)

### Opsi A — Astral uv (disarankan)

1. Instal uv (sekali saja): `pip install uv`
2. Buat virtual environment: `uv venv`
3. Aktifkan: `.venv\Scripts\Activate.ps1`
4. Instal dependensi: `uv pip install -r requirements.txt`

### Opsi B — Python venv + pip

1. Buat dan aktifkan virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Instal dependensi:

```powershell
pip install -r requirements.txt
```

3. Salin `.env.example` ke `.env` dan isi nilai API/konfigurasi.

Contoh variabel di `.env`:

```text
GEMINI_API_KEY=
OPENROUTER_KEY=
MOODLE_DOWNLOAD_TOKEN=
MODEL=google/gemini-2.0-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
```

Catatan: `index.py` kini membaca token Moodle dari `.env` (variabel `MOODLE_DOWNLOAD_TOKEN`).

---

## Cara Menjalankan

1. Mode CLI (satu tugas)

`index.py` menerima argumen `--input` berupa string JSON. Contoh (PowerShell):

```powershell
$json = '{"userId":"U123","assignmentId":"A001","assignmentUrl":"https://example.com/tugas.pdf","rubric": {"criteria": []}}'
python index.py --input $json
```

Respons JSON akan dicetak ke stdout.

2. Mode Batch (semua file di folder `data/`)

```powershell
python rag_grading_improved.py
```

Hasil run batch akan disimpan di `output/` (jika script menyimpan hasil) dan/atau dicetak ke terminal.

---

## Format Rubrik

File contoh `data/rubrik.json` berisi struktur kriteria yang digunakan oleh `GradingEngine`. Pastikan rubrik dikirim ke `index.py` sebagai objek JSON pada field `rubric`.

---
