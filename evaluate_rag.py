import re
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

###############################################
# HARD CODE RUBRIC + PREDICTIONS
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
            "rubric": {"criteria": [
                {"criterionid":3,"fillings":[{"criterionid":3,"levelid":19,
                    "remark":"Dasar teori mengenai output, operasi aritmatika, eksekusi kondisional, dan konversi tipe data dijelaskan dengan cukup jelas.","confidence":0.8}]},

                {"criterionid":6,"fillings":[{"criterionid":6,"levelid":16,
                    "remark":"Kode program yang diberikan kurang efisien dan penamaan variabel kurang informatif","confidence":0.6}]},

                {"criterionid":9,"fillings":[{"criterionid":9,"levelid":27,
                    "remark":"Penjelasan baris kode kurang informatif","confidence":0.7}]},

                {"criterionid":10,"fillings":[{"criterionid":10,"levelid":28,
                    "remark":"Kesimpulan kurang jelas.","confidence":0.6}]}
            ]}
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
# METRIK EVALUASI TAMBAHAN
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
    """
    completeness = berapa banyak kalimat evidence yang dibahas di remark
    """
    if not evidence_sents:
        return 0.0

    used = 0
    for s in evidence_sents:
        if s.strip() and s.lower()[:20] in remark.lower():
            used += 1

    return used / len(evidence_sents)

def compute_faithfulness(remark, pdf_text):
    """
    faithfulness = apakah remark hanya berisi informasi yang benar-benar ada di PDF
    jika banyak token remark tidak ditemukan di PDF → kemungkinan halusinasi
    """
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
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    results = []
    pred_scores = []
    true_scores = []

    # rubrik mapping (levelid → score)
    rubric_map = {
        c["id"]: {lvl["id"]: lvl["score"] for lvl in c["levels"]}
        for c in rubric["rubric"]["rubric_criteria"]
    }

    for item in predictions:
        for crit in item["advancedgradingdata"]["rubric"]["criteria"]:
            cid = crit["criterionid"]
            fill = crit["fillings"][0]

            remark = fill["remark"]
            levelid = fill["levelid"]
            score_pred = rubric_map[cid][levelid]

            # ground truth (ambil score tertinggi rubrik -> simulasi manual)  
            score_true = max(rubric_map[cid].values())  

            pred_scores.append(score_pred)
            true_scores.append(score_true)

            # relevance / similarity
            emb_remark = embedder.encode(remark, convert_to_tensor=True)
            sims = []
            for s in sentences:
                sims.append(float(util.cos_sim(emb_remark, embedder.encode(s)).item()))
            top_evidence_idx = np.argsort(sims)[-5:]
            evidence_sents = [sentences[i] for i in top_evidence_idx]

            relevance = float(np.mean([sims[i] for i in top_evidence_idx]))

            # completeness
            completeness = compute_completeness(evidence_sents, remark)

            # faithfulness
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

    # accuracy metrics
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

pdf_path = "data\\LP 2 & 3_Kelompok 4.pdf"   # GANTI DENGAN FILE PDF YANG BENAR

report = evaluate_full(pdf_path, predictions, rubric)

print("\n===== SUMMARY =====")
print(report["summary"])

print("\n===== DETAIL PER KRITERIA =====")
for item in report["items"]:
    print("\n--- Criterion", item["criterion_id"], "---")
    print("Score Pred:", item["pred_score"], " | Score True:", item["true_score"])
    print("Relevance:", item["relevance_similarity"])
    print("Completeness:", item["completeness"])
    print("Faithfulness:", item["faithfulness"])
