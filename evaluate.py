import argparse
import json
import re
from pathlib import Path

import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util


###############################################
# GROUND TRUTH (penilaian manual)
###############################################
ground_truth = {
    "LP 2 & 3_Kelompok 8.pdf": {
        3: 5,    # Dasar Teori
        6: 42,   # Kode Program
        9: 13,   # Keterangan Baris
        10: 5    # Kesimpulan
    },
    "Laprak _Minggu 2 dan 3_Kelompok 6.pdf": {
        3: 5,
        6: 48,
        9: 20,
        10: 20
    },
    "LP 2 & 3_Kelompok 4.pdf": {
        3: 10,
        6: 50,
        9: 19,
        10: 20
    },
    "_LP 12 & 13_Kelompok 6.docx.pdf": {
        3: 8,
        6: 50,
        9: 16,
        10: 18
    },
    "LP 4_Kelompok 7.pdf": {
        3: 10,
        6: 50,
        9: 14,
        10: 20
    }
}


###############################################
# RUBRIC DAN PREDIKSI MODEL
###############################################
rubric = {
    "rubric": {
        "rubric_criteria": [
            {"id": 3, "description": "Dasar Teori", "levels": [{"id":7,"score":0},{"id":9,"score":6},{"id":19,"score":10}]},
            {"id": 6, "description": "Kode Program", "levels": [{"id":16,"score":0},{"id":17,"score":30},{"id":18,"score":50}]},
            {"id": 9, "description": "Keterangan Baris Program", "levels": [{"id":25,"score":0},{"id":26,"score":12},{"id":27,"score":20}]},
            {"id": 10, "description": "Kesimpulan", "levels": [{"id":28,"score":0},{"id":29,"score":12},{"id":30,"score":20}]},
        ]
    }
}

predictions = [
    {
        "assignment_id": "01",
        "user_id": "1121033",
        "document_info": {"filename": "_LP 12 & 13_Kelompok 6.docx.pdf"},
        "advancedgradingdata": {
    "rubric": {
      "criteria": [
        {
          "criterionid": 3,
          "fillings": [
            {
              "criterionid": 3,
              "levelid": 19,
              "remark": "Penjelasan mengenai penggunaan modul sys, PyQt5, dan struktur kelas BarangApp dikemukakan dengan jelas dan dapat dimengerti. Penjelasan tentang logika validasi juga disajikan dengan baik.",  
              "confidence": 0.9
            }
          ]
        },
        {
          "criterionid": 6,
          "fillings": [
            {
              "criterionid": 6,
              "levelid": 18,
              "remark": "Dari potongan kode dan deskripsi yang diberikan, penamaan variabel seperti `nama_barang`, `harga_barang`, `input_nama`, `selected_row`, dan `formatted_harga` informatif. Tidak ada indikasi program tidak efisien dari evidence yang tersedia.",
              "confidence": 0.8
            }
          ]
        },
        {
          "criterionid": 9,
          "fillings": [
            {
              "criterionid": 9,
              "levelid": 27,
              "remark": "Terdapat bagian 'Pembahasan' yang menjelaskan fungsi kode per bagian berdasarkan nomor baris (misalnya '1-4', '6-10', '69-78'). Penjelasan ini sangat informatif dan mencakup bagian-bagian penting dari kode, memenuhi kriteria bahwa setiap baris code diberikan penjelasan dalam konteks blok logis.",
              "confidence": 0.9
            }
          ]
        },
        {
          "criterionid": 10,
          "fillings": [
            {
              "criterionid": 10,
              "levelid": 28,
              "remark": "Tidak ditemukan bagian kesimpulan yang merangkum materi ataupun program yang telah dibuat oleh siswa. Terdapat bagian 'Saran' tetapi itu bukan kesimpulan dari hasil kerja.",
              "confidence": 0.95
            }
          ]
        }
      ]
    }       
        }  
    }    
]


###############################################
# PDF UTIL
###############################################
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t + "\n"
    return text

def split_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return re.split(r'(?<=[\.\?\!])\s+', text)


###############################################
# METRIK
###############################################
def compute_accuracy_metrics(pred_levels, true_levels):
    pred = np.array(pred_levels)
    true = np.array(true_levels)
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true)**2)
    rmse = np.sqrt(mse)
    acc = np.mean(pred == true)
    return mae, mse, rmse, acc


def compute_completeness(evidence_sents, remark):
  """Nilai rata-rata tumpang tindih token antara bukti dan remark."""

  if not evidence_sents:
    return 0.0

  remark_tokens = set(re.findall(r"\w+", remark.lower()))
  if not remark_tokens:
    return 0.0

  coverages = []
  for sent in evidence_sents:
    sent_tokens = set(re.findall(r"\w+", sent.lower()))
    if not sent_tokens:
      continue

    overlap = remark_tokens & sent_tokens
    coverages.append(len(overlap) / len(sent_tokens))

  if not coverages:
    return 0.0

  return float(np.mean(coverages))


def compute_faithfulness(remark, pdf_text):
    remark_tokens = {w.lower() for w in re.findall(r'\w+', remark)}
    pdf_tokens    = {w.lower() for w in re.findall(r'\w+', pdf_text)}

    overlap = remark_tokens & pdf_tokens
    if not remark_tokens:
        return 0.0

    return len(overlap) / len(remark_tokens)



###############################################
# EVALUASI UTAMA
###############################################
def evaluate_full(pdf_path, predictions, rubric):

    pdf_text = extract_text_from_pdf(pdf_path)
    sentences = split_sentences(pdf_text)
    embedder = SentenceTransformer("intfloat/multilingual-e5-large")

    results = []
    pred_scores = []
    true_scores = []

    rubric_map = {
        c["id"]: {lvl["id"]: lvl["score"] for lvl in c["levels"]}
        for c in rubric["rubric"]["rubric_criteria"]
    }

    for item in predictions:
        filename = item["document_info"]["filename"]

        if filename not in ground_truth:
            raise ValueError(f"Ground truth missing for: {filename}")

        for crit in item["advancedgradingdata"]["rubric"]["criteria"]:
            cid = crit["criterionid"]
            fill = crit["fillings"][0]

            remark = fill["remark"]
            levelid = fill["levelid"]

            score_pred = rubric_map[cid][levelid]
            score_true = ground_truth[filename][cid]

            pred_scores.append(score_pred)
            true_scores.append(score_true)

            # similarity evidence
            emb_remark = embedder.encode(remark, convert_to_tensor=True)
            sims = []
            for s in sentences:
                sims.append(float(util.cos_sim(emb_remark, embedder.encode(s)).item()))

            top_evidence_idx = np.argsort(sims)[-5:]
            evidence_sents = [sentences[i] for i in top_evidence_idx]

            relevance = float(np.mean([sims[i] for i in top_evidence_idx]))

            completeness = compute_completeness(evidence_sents, remark)
            faith = compute_faithfulness(remark, pdf_text)

            results.append({
                "criterion_id": cid,
                "remark": remark,
                "pred_level": levelid,
                "pred_score": score_pred,
                "true_score": score_true,
                "evidence": evidence_sents,
                "relevance_similarity": relevance,
                "completeness": completeness,
                "faithfulness": faith,
            })

    mae, mse, rmse, acc = compute_accuracy_metrics(pred_scores, true_scores)

    return {
        "summary": {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Accuracy_Level": acc,
        },
        "items": results
    }


###############################################
# JALANKAN
###############################################
pdf_path = "data\\_LP 12 & 13_Kelompok 6.docx.pdf"

report = evaluate_full(pdf_path, predictions, rubric)

print("\n===== SUMMARY =====")
print(report["summary"])

print("\n===== DETAIL PER KRITERIA =====")
for item in report["items"]:
    print("\n--- Criterion", item["criterion_id"], "---")
    print("Pred:", item["pred_score"], "| True:", item["true_score"])
    print("Relevance:", item["relevance_similarity"])
    print("Completeness:", item["completeness"])
    print("Faithfulness:", item["faithfulness"])
