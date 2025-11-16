import json
import os
import math
from typing import List, Dict, Any
from statistics import mean

# =====================================================================
# RUBRIC MAP (menyesuaikan rubrik manualmu)
# =====================================================================
RUBRIC_MAP = {
    3: {7: 5, 9: 5, 19: 10},  # Dasar teori
    6: {16: 42, 17: 48, 18: 50},  # Kode program
    9: {25: 13, 26: 14, 27: 20},  # Keterangan baris
    10: {28: 5, 29: 18, 30: 20}   # Kesimpulan
}

# =====================================================================
# FILE → FILENAME MANUAL MAPPING
# =====================================================================
FILE_MAPPING = {
    0: "LP 2 & 3_Kelompok 8.pdf"
}

# =====================================================================
# LOAD JSON – aman untuk semua format
# =====================================================================
def load_json_flexible(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    cleaned = []
    for item in data:
        if isinstance(item, str):
            try:
                cleaned.append(json.loads(item))
            except:
                print("⚠️ Gagal parse string JSON:", item[:80])
                continue
        elif isinstance(item, dict):
            cleaned.append(item)
        else:
            print("⚠️ Format item tidak valid:", item)

    return cleaned

# =====================================================================
# EXTRACT AI SCORES — tidak wajib punya document_info
# =====================================================================
def extract_ai_scores(results: List[Dict]):
    ai_scores = []

    for idx, item in enumerate(results):
        
        filename = FILE_MAPPING.get(idx, f"Unknown_{idx}")

        criteria = (
            item.get("advancedgradingdata", {})
               .get("rubric", {})
               .get("criteria", [])
        )

        if len(criteria) < 4:
            print(f"⚠️ Rubrik tidak lengkap untuk {filename}")
            continue

        dasar = criteria[0]["fillings"][0]["levelid"]
        kode = criteria[1]["fillings"][0]["levelid"]
        baris = criteria[2]["fillings"][0]["levelid"]
        simpul = criteria[3]["fillings"][0]["levelid"]

        skor_dasar = RUBRIC_MAP[3].get(dasar, 0)
        skor_kode = RUBRIC_MAP[6].get(kode, 0)
        skor_baris = RUBRIC_MAP[9].get(baris, 0)
        skor_simpul = RUBRIC_MAP[10].get(simpul, 0)

        total = skor_dasar + skor_kode + skor_baris + skor_simpul

        ai_scores.append({
            "submission": filename,
            "dasar_teori": skor_dasar,
            "kode_program": skor_kode,
            "keterangan_baris": skor_baris,
            "kesimpulan": skor_simpul,
            "total": total
        })

    return ai_scores

# =====================================================================
# LOAD GROUND TRUTH
# =====================================================================
def load_ground_truth(path: str) -> Dict[str, int]:
    data = load_json_flexible(path)
    truth = {}

    for item in data:
        truth[item["submission"]] = item["final_score"]

    return truth

# =====================================================================
# METRIC CALCULATION
# =====================================================================
def compute_metrics(pred: List[float], target: List[float]):
    errors = [abs(p - t) for p, t in zip(pred, target)]

    mae = mean(errors)
    mse = mean([(p - t) ** 2 for p, t in zip(pred, target)])
    rmse = math.sqrt(mse)
    acc = sum(1 for p, t in zip(pred, target) if abs(p - t) <= 0.1 * t) / len(pred)

    return mae, mse, rmse, acc

# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 80)
    print("📊 RAG METRICS EVALUATION (with Ground Truth)")
    print("=" * 80)

    base = os.path.dirname(os.path.abspath(__file__))
    grading_path = os.path.join(base, "hasil_grading.json")
    truth_path = os.path.join(base, "ground_truth.json")

    print("📥 Membaca hasil_grading.json ...")
    grading_results = load_json_flexible(grading_path)

    print("📥 Membaca ground_truth.json ...")
    ground_truth = load_ground_truth(truth_path)

    print("\n🔄 Mengambil skor AI ...")
    ai_scores = extract_ai_scores(grading_results)

    print("\n=== SCORE COMPARISON ===")
    pred, target = [], []

    for item in ai_scores:
        fname = item["submission"]
        ai_total = item["total"]

        if fname not in ground_truth:
            print(f"⚠️ Ground truth tidak ditemukan untuk {fname}")
            continue

        manual = ground_truth[fname]
        print(f"- {fname} → AI = {ai_total}, Manual = {manual}")

        pred.append(ai_total)
        target.append(manual)

    mae, mse, rmse, acc = compute_metrics(pred, target)

    print("\n=== 📈 METRICS RESULT ===")
    print(f"MAE   : {mae:.4f}")
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"ACC (error ≤10%) : {acc*100:.2f}%")

    out_path = os.path.join(base, "metric_results.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Accuracy": acc
        }, f, indent=2)

    print("\n💾 metric_results.json berhasil disimpan.")
    print("=" * 80)

if __name__ == "__main__":
    main()
