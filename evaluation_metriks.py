import json
import math
from statistics import mean

# ============================================
# RAGAS-LIKE AUTO GRADE EVALUATION (FINAL FIX)
# ============================================

def ragas_autograde_matrix(ai_data, gt_data):
    """
    ai_data → hasil auto grading AI (levelid + confidence)
    gt_data → ground truth rubrik (levelid dari human)
    """

    ai_criteria = ai_data["advancedgradingdata"]["rubric"]["criteria"]
    gt_criteria = gt_data["advancedgradingdata"]["rubric"]["criteria"]

    # -------------------------------------------
    # Rubric Comparison
    # -------------------------------------------
    rubric_matches = []
    criteria_result = {}

    for ai, gt in zip(ai_criteria, gt_criteria):

        cid = str(ai["criterionid"])  # gunakan string agar tidak duplikat key
        ai_level = ai["fillings"][0]["levelid"]
        gt_level = gt["fillings"][0]["levelid"]

        match = 1 if ai_level == gt_level else 0
        rubric_matches.append(match)

        criteria_result[cid] = {
            "ai": ai_level,
            "manual": gt_level,
            "match": match
        }

    rubric_consistency = sum(rubric_matches) / len(rubric_matches)

    # -------------------------------------------
    # SCORE Mapping (Manual Rubric)
    # -------------------------------------------

    RUBRIC_MAP = {
        3: {7: 5, 9: 5, 19: 10},       # Dasar teori
        6: {16: 42, 17: 48, 18: 50},   # Kode program
        9: {25: 13, 26: 14, 27: 20},   # Keterangan baris
        10: {28: 5, 29: 18, 30: 20}    # Kesimpulan
    }

    def calc_score(criteria):
        total = 0
        for c in criteria:
            cid = c["criterionid"]
            levelid = c["fillings"][0]["levelid"]
            total += RUBRIC_MAP.get(cid, {}).get(levelid, 0)
        return total

    ai_score = calc_score(ai_criteria)
    gt_score = calc_score(gt_criteria)

    # -------------------------------------------
    # Score-based metrics
    # -------------------------------------------
    score_deviation = abs(ai_score - gt_score)
    score_accuracy = 1 - (score_deviation / max(gt_score, 1))

    # -------------------------------------------
    # Faithfulness — Confidence AI
    # -------------------------------------------
    def compute_faithfulness(criteria):
        return mean([c["fillings"][0]["confidence"] for c in criteria])

    faithfulness = compute_faithfulness(ai_criteria)

    # -------------------------------------------
    # Explanation Alignment (AI reasoning vs GT)
    # -------------------------------------------
    def explanation_alignment(ai_criteria, gt_criteria):
        aligns = []
        for ai, gt in zip(ai_criteria, gt_criteria):
            if ai["fillings"][0]["levelid"] == gt["fillings"][0]["levelid"]:
                aligns.append(1)      # benar full
            else:
                aligns.append(0.5)    # sebagian benar
        return mean(aligns)

    exp_align = explanation_alignment(ai_criteria, gt_criteria)

    # -------------------------------------------
    # Overall Score (RAGAS style aggregation)
    # -------------------------------------------
    overall = mean([
        rubric_consistency,
        score_accuracy,
        faithfulness,
        exp_align
    ])

    # -------------------------------------------
    # Output JSON (Clean & Valid)
    # -------------------------------------------
    return {
        "rubric_consistency": rubric_consistency,
        "score_accuracy": score_accuracy,
        "score_deviation": score_deviation,
        "criteria_match": criteria_result,
        "faithfulness": faithfulness,
        "explanation_alignment": exp_align,
        "overall_score": overall
    }


# ============================================
# TEST MENGGUNAKAN DATA KAMU
# ============================================

ai_json = {
  "advancedgradingdata": {
    "rubric": {
      "criteria": [
        {"criterionid": 3, "fillings": [{"criterionid": 3, "levelid": 9, "confidence": 0.7}]},
        {"criterionid": 6, "fillings": [{"criterionid": 6, "levelid": 17, "confidence": 0.6}]},
        {"criterionid": 9, "fillings": [{"criterionid": 9, "levelid": 26, "confidence": 0.8}]},
        {"criterionid": 10,"fillings": [{"criterionid": 10, "levelid": 20, "confidence": 0.9}]}
      ]
    }
  }
}

gt_json = {
  "advancedgradingdata": {
    "rubric": {
      "criteria": [
        {"criterionid": 3, "fillings": [{"criterionid": 3, "levelid": 9}]},
        {"criterionid": 6, "fillings": [{"criterionid": 6, "levelid": 18}]},
        {"criterionid": 9, "fillings": [{"criterionid": 9, "levelid": 17}]},
        {"criterionid": 10,"fillings": [{"criterionid": 10, "levelid": 30}]}
      ]
    }
  }
}

result = ragas_autograde_matrix(ai_json, gt_json)
print(json.dumps(result, indent=2))
