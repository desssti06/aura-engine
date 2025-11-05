import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
import json

class RAGEvaluationMetrics:
    """
    Metrik evaluasi untuk sistem RAG Auto-Grading

    Mengukur performa sistem dari 3 aspek:
    1. Retrieval Quality (RAG component)
    2. Generation Quality (LLM component)
    3. Overall Grading Accuracy (end-to-end)
    """

    @staticmethod
    def mean_absolute_error(predicted: List[float], actual: List[float]) -> float:
        """
        MAE: Rata-rata error absolut antara prediksi dan actual

        Range: 0 to infinity (0 = perfect, semakin kecil semakin baik)
        """
        return mean_absolute_error(actual, predicted)

    @staticmethod
    def root_mean_squared_error(predicted: List[float], actual: List[float]) -> float:
        """
        RMSE: Root mean squared error
        Lebih sensitif terhadap outlier dibanding MAE

        Range: 0 to infinity (0 = perfect, semakin kecil semakin baik)
        """
        return np.sqrt(mean_squared_error(actual, predicted))

    @staticmethod
    def pearson_correlation(predicted: List[float], actual: List[float]) -> Tuple[float, float]:
        """
        Pearson Correlation Coefficient
        Mengukur linear relationship antara AI grading vs human grading

        Returns:
            (correlation, p-value)
            correlation range: -1 to 1 (1 = perfect positive correlation)
        """
        return stats.pearsonr(predicted, actual)

    @staticmethod
    def spearman_correlation(predicted: List[float], actual: List[float]) -> Tuple[float, float]:
        """
        Spearman Rank Correlation
        Mengukur monotonic relationship (lebih robust untuk outlier)

        Returns:
            (correlation, p-value)
        """
        return stats.spearmanr(predicted, actual)

    @staticmethod
    def exact_match_rate(predicted: List[float], actual: List[float], tolerance: float = 0) -> float:
        """
        Exact Match Rate: Proporsi prediksi yang sama persis dengan actual

        Args:
            tolerance: Toleransi error (default 0 = exact match)

        Returns:
            Rate: 0 to 1
        """
        matches = sum(abs(p - a) <= tolerance for p, a in zip(predicted, actual))
        return matches / len(predicted)

    @staticmethod
    def within_range_accuracy(predicted: List[float], actual: List[float], range_threshold: float = 5) -> float:
        """
        Within-Range Accuracy: Proporsi prediksi dalam range threshold

        Args:
            range_threshold: Batas error yang diterima (default ±5)

        Returns:
            Accuracy: 0 to 1
        """
        within_range = sum(abs(p - a) <= range_threshold for p, a in zip(predicted, actual))
        return within_range / len(predicted)

    @staticmethod
    def cohen_kappa(predicted_labels: List[str], actual_labels: List[str]) -> float:
        """
        Cohen's Kappa: Agreement antara AI vs Human rater
        Mengukur agreement beyond chance

        Args:
            labels: Grade level (e.g., ['A', 'B', 'C'])

        Returns:
            Kappa: -1 to 1
            < 0: worse than random
            0: random agreement
            0.01-0.20: slight agreement
            0.21-0.40: fair agreement
            0.41-0.60: moderate agreement
            0.61-0.80: substantial agreement
            0.81-1.00: almost perfect agreement
        """
        return cohen_kappa_score(actual_labels, predicted_labels)

    @staticmethod
    def confidence_accuracy_correlation(confidences: List[float], errors: List[float]) -> float:
        """
        Korelasi antara confidence score dan accuracy
        Confidence yang baik harus berkorelasi negatif dengan error
        (high confidence → low error)

        Returns:
            Correlation: -1 to 1 (idealnya negatif)
        """
        corr, _ = stats.pearsonr(confidences, errors)
        return corr

    @staticmethod
    def expected_calibration_error(confidences: List[float], errors: List[float], n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE)
        Mengukur seberapa well-calibrated confidence scores

        Confidence yang well-calibrated: jika model 80% confident,
        seharusnya benar 80% of the time

        Returns:
            ECE: 0 to 1 (0 = perfectly calibrated)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        accuracies = [1 - e for e in errors]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [(c >= bin_lower) and (c < bin_upper) for c in confidences]
            prop_in_bin = sum(in_bin) / len(in_bin) if sum(in_bin) > 0 else 0

            if prop_in_bin > 0:
                accuracy_in_bin = sum([a for a, b in zip(accuracies, in_bin) if b]) / sum(in_bin)
                avg_confidence_in_bin = sum([c for c, b in zip(confidences, in_bin) if b]) / sum(in_bin)
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    @staticmethod
    def hit_rate_at_k(retrieved_chunks: List[List[str]], relevant_chunks: List[List[str]], k: int = 5) -> float:
        """
        Hit Rate@K: Proporsi queries yang menemukan dokumen relevan di Top-K

        Args:
            retrieved_chunks: List of retrieved chunks per query
            relevant_chunks: List of known relevant chunks per query
            k: Top-K to consider

        Returns:
            Hit Rate: 0 to 1
        """
        hits = 0
        for retrieved, relevant in zip(retrieved_chunks, relevant_chunks):
            top_k = retrieved[:k]
            if any(chunk in relevant for chunk in top_k):
                hits += 1

        return hits / len(retrieved_chunks)

    @staticmethod
    def mean_reciprocal_rank(retrieved_chunks: List[List[str]], relevant_chunks: List[List[str]]) -> float:
        """
        Mean Reciprocal Rank (MRR)
        Mengukur ranking quality: semakin tinggi posisi dokumen relevan, semakin baik

        Returns:
            MRR: 0 to 1 (1 = relevant doc always at rank 1)
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_chunks, relevant_chunks):
            for rank, chunk in enumerate(retrieved, 1):
                if chunk in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

    @staticmethod
    def context_precision(retrieved_chunks: List[str], used_chunks: List[str]) -> float:
        """
        Context Precision: Proporsi retrieved chunks yang benar-benar digunakan

        High precision = sistem tidak retrieve banyak irrelevant chunks

        Returns:
            Precision: 0 to 1
        """
        if not retrieved_chunks:
            return 0.0

        used_count = sum(1 for chunk in retrieved_chunks if chunk in used_chunks)
        return used_count / len(retrieved_chunks)

    @staticmethod
    def context_recall(relevant_chunks: List[str], retrieved_chunks: List[str]) -> float:
        """
        Context Recall: Proporsi relevant chunks yang berhasil di-retrieve

        High recall = sistem tidak miss informasi penting

        Returns:
            Recall: 0 to 1
        """
        if not relevant_chunks:
            return 0.0

        retrieved_count = sum(1 for chunk in relevant_chunks if chunk in retrieved_chunks)
        return retrieved_count / len(relevant_chunks)

    @staticmethod
    def faithfulness_score(response: str, evidence_chunks: List[str]) -> float:
        """
        Faithfulness Score: Proporsi claims dalam response yang didukung evidence

        Simplified version: Check overlap between response and evidence
        (Implementasi lengkap butuh NLI model atau LLM verification)

        Returns:
            Score: 0 to 1 (1 = fully faithful to evidence)
        """
        response_words = set(response.lower().split())
        evidence_words = set(' '.join(evidence_chunks).lower().split())

        if not response_words:
            return 0.0

        overlap = response_words.intersection(evidence_words)
        return len(overlap) / len(response_words)

    @staticmethod
    def grade_distribution_similarity(predicted_dist: Dict[str, int], actual_dist: Dict[str, int]) -> float:
        """
        Kullback-Leibler Divergence untuk distribusi grades
        Mengukur seberapa mirip distribusi nilai AI vs Human

        Returns:
            KL Divergence: 0 to infinity (0 = identical distributions)
        """
        all_grades = set(predicted_dist.keys()).union(actual_dist.keys())

        p_total = sum(predicted_dist.values())
        q_total = sum(actual_dist.values())

        p_dist = np.array([predicted_dist.get(g, 0) / p_total for g in sorted(all_grades)])
        q_dist = np.array([actual_dist.get(g, 0) / q_total for g in sorted(all_grades)])

        p_dist = p_dist + 1e-10
        q_dist = q_dist + 1e-10

        return stats.entropy(p_dist, q_dist)

    @classmethod
    def comprehensive_evaluation(cls,
                                  ai_results: List[Dict],
                                  human_results: List[Dict]) -> Dict:
        """
        Comprehensive evaluation: Hitung semua metrik sekaligus

        Args:
            ai_results: List of AI grading results
            human_results: List of human grading results

        Returns:
            Dict dengan semua metrik
        """
        ai_scores = [r['final_score'] for r in ai_results]
        human_scores = [r['final_score'] for r in human_results]

        ai_confidences = [r.get('overall_confidence', 0) for r in ai_results]
        errors = [abs(a - h) for a, h in zip(ai_scores, human_scores)]

        ai_labels = []
        human_labels = []
        for ai_r, human_r in zip(ai_results, human_results):
            for ai_grade, human_grade in zip(ai_r['grading_result'], human_r['grading_result']):
                ai_labels.append(ai_grade['selected_level'])
                human_labels.append(human_grade['selected_level'])

        pearson_corr, pearson_p = cls.pearson_correlation(ai_scores, human_scores)
        spearman_corr, spearman_p = cls.spearman_correlation(ai_scores, human_scores)

        metrics = {
            "accuracy_metrics": {
                "mae": cls.mean_absolute_error(ai_scores, human_scores),
                "rmse": cls.root_mean_squared_error(ai_scores, human_scores),
                "exact_match_rate": cls.exact_match_rate(ai_scores, human_scores, tolerance=0),
                "within_5_points": cls.within_range_accuracy(ai_scores, human_scores, range_threshold=5),
                "within_10_points": cls.within_range_accuracy(ai_scores, human_scores, range_threshold=10),
            },
            "correlation_metrics": {
                "pearson_r": pearson_corr,
                "pearson_p_value": pearson_p,
                "spearman_rho": spearman_corr,
                "spearman_p_value": spearman_p,
            },
            "agreement_metrics": {
                "cohen_kappa": cls.cohen_kappa(ai_labels, human_labels),
            },
            "confidence_metrics": {
                "mean_confidence": np.mean(ai_confidences),
                "confidence_accuracy_correlation": cls.confidence_accuracy_correlation(ai_confidences, errors),
                "ece": cls.expected_calibration_error(ai_confidences, [e / 100 for e in errors]),
            },
            "summary": {
                "total_documents": len(ai_results),
                "mean_error": np.mean(errors),
                "median_error": np.median(errors),
                "max_error": np.max(errors),
            }
        }

        return metrics

    @staticmethod
    def print_evaluation_report(metrics: Dict):
        """Pretty print evaluation metrics"""
        print("\n" + "="*60)
        print("RAG AUTO-GRADING EVALUATION REPORT")
        print("="*60)

        print("\n📊 ACCURACY METRICS:")
        acc = metrics['accuracy_metrics']
        print(f"  MAE (Mean Absolute Error):     {acc['mae']:.2f} points")
        print(f"  RMSE (Root Mean Squared):      {acc['rmse']:.2f} points")
        print(f"  Exact Match Rate:              {acc['exact_match_rate']*100:.1f}%")
        print(f"  Within ±5 points:              {acc['within_5_points']*100:.1f}%")
        print(f"  Within ±10 points:             {acc['within_10_points']*100:.1f}%")

        print("\n📈 CORRELATION METRICS:")
        corr = metrics['correlation_metrics']
        print(f"  Pearson Correlation:           {corr['pearson_r']:.3f} (p={corr['pearson_p_value']:.4f})")
        print(f"  Spearman Correlation:          {corr['spearman_rho']:.3f} (p={corr['spearman_p_value']:.4f})")

        print("\n🤝 AGREEMENT METRICS:")
        agree = metrics['agreement_metrics']
        kappa = agree['cohen_kappa']
        print(f"  Cohen's Kappa:                 {kappa:.3f}", end="")
        if kappa < 0:
            print(" (worse than random)")
        elif kappa < 0.20:
            print(" (slight agreement)")
        elif kappa < 0.40:
            print(" (fair agreement)")
        elif kappa < 0.60:
            print(" (moderate agreement)")
        elif kappa < 0.80:
            print(" (substantial agreement)")
        else:
            print(" (almost perfect agreement)")

        print("\n🎯 CONFIDENCE METRICS:")
        conf = metrics['confidence_metrics']
        print(f"  Mean Confidence:               {conf['mean_confidence']:.3f}")
        print(f"  Confidence-Error Correlation:  {conf['confidence_accuracy_correlation']:.3f}")
        print(f"  Expected Calibration Error:    {conf['ece']:.3f}")

        print("\n📋 SUMMARY:")
        summ = metrics['summary']
        print(f"  Total Documents:               {summ['total_documents']}")
        print(f"  Mean Error:                    {summ['mean_error']:.2f} points")
        print(f"  Median Error:                  {summ['median_error']:.2f} points")
        print(f"  Max Error:                     {summ['max_error']:.2f} points")

        print("\n" + "="*60)

        if acc['mae'] < 5:
            print("✅ EXCELLENT: MAE < 5 points")
        elif acc['mae'] < 10:
            print("✅ GOOD: MAE < 10 points")
        elif acc['mae'] < 15:
            print("⚠️ ACCEPTABLE: MAE < 15 points")
        else:
            print("❌ NEEDS IMPROVEMENT: MAE > 15 points")

        print("="*60)


if __name__ == "__main__":
    print("RAG Evaluation Metrics Module")
    print("Import this module to evaluate your RAG auto-grading system")

    print("\nExample usage:")
    print("""
    from evaluation_metrics import RAGEvaluationMetrics

    # Load AI results and human results
    ai_results = [...]  # Your AI grading results
    human_results = [...] # Ground truth from human graders

    # Calculate all metrics
    metrics = RAGEvaluationMetrics.comprehensive_evaluation(ai_results, human_results)

    # Print report
    RAGEvaluationMetrics.print_evaluation_report(metrics)
    """)
