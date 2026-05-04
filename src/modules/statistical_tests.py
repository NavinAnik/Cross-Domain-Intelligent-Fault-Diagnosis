"""
Statistical tests and confidence intervals for evaluation (e.g. cross-validation).

Provides:
- 95% and 99% confidence intervals for metric means (e.g. across folds).
- Bootstrap confidence intervals for accuracy.
- Paired t-test for comparing two methods across folds.
- Wilcoxon signed-rank test for non-parametric comparison.
- Chi-square test for contingency tables.
- Friedman test for comparing multiple methods across blocks.
"""

import numpy as np
from scipy import stats


def confidence_interval(scores, confidence=0.95):
    """
    Compute confidence interval for the mean of a sample (e.g. metric across folds).

    Args:
        scores: Array-like of metric values (e.g. one value per fold).
        confidence: Confidence level in (0, 1), default 0.95 for 95% CI.

    Returns:
        tuple: (lower_bound, upper_bound, mean, std_error).
    """
    scores = np.asarray(scores).ravel()
    n = len(scores)
    if n < 2:
        mean = float(np.mean(scores))
        return (mean, mean, mean, 0.0)

    mean = np.mean(scores)
    sem = stats.sem(scores)
    half_width = sem * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    lower = mean - half_width
    upper = mean + half_width
    return (float(lower), float(upper), float(mean), float(sem))


def confidence_interval_str(scores, confidence=0.95):
    """
    Format mean ± half-width and [lower, upper] as a string (e.g. for tables).

    Returns:
        str: e.g. "0.85 ± 0.02 [0.81, 0.89]"
    """
    lower, upper, mean, sem = confidence_interval(scores, confidence)
    half = (upper - lower) / 2
    return f"{mean:.4f} ± {half:.4f} [{lower:.4f}, {upper:.4f}]"


def paired_ttest(scores_a, scores_b):
    """
    Paired t-test for comparing two methods across the same folds/splits.

    Args:
        scores_a: Metric values for method A (one per fold).
        scores_b: Metric values for method B (one per fold).

    Returns:
        dict: {"statistic": float, "pvalue": float, "mean_diff": float}
    """
    scores_a = np.asarray(scores_a).ravel()
    scores_b = np.asarray(scores_b).ravel()
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")
    stat, pval = stats.ttest_rel(scores_a, scores_b)
    return {
        "statistic": float(stat),
        "pvalue": float(pval),
        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
    }


def wilcoxon_signed_rank(scores_a, scores_b):
    """
    Wilcoxon signed-rank test (non-parametric) for comparing two methods.

    Args:
        scores_a: Metric values for method A (one per fold).
        scores_b: Metric values for method B (one per fold).

    Returns:
        dict: {"statistic": float, "pvalue": float}
    """
    scores_a = np.asarray(scores_a).ravel()
    scores_b = np.asarray(scores_b).ravel()
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")
    stat, pval = stats.wilcoxon(scores_a, scores_b)
    return {"statistic": float(stat), "pvalue": float(pval)}


def aggregate_cv_metrics(list_of_metric_dicts, confidence=0.95):
    """
    Aggregate metrics from multiple folds/runs with confidence intervals.

    Args:
        list_of_metric_dicts: List of dicts, each from compute_classification_metrics
            for one fold/run. Each dict should have the same keys (e.g. "accuracy", "f1").
        confidence: Confidence level for CIs.

    Returns:
        dict: For each metric key, value is dict with "mean", "std", "ci_lower", "ci_upper",
              "ci_str" (formatted string).
    """
    if not list_of_metric_dicts:
        return {}

    keys = list(list_of_metric_dicts[0].keys())
    result = {}
    for k in keys:
        values = [d[k] for d in list_of_metric_dicts if k in d and d[k] is not None]
        if not values:
            result[k] = {"mean": None, "std": None, "ci_lower": None, "ci_upper": None, "ci_str": None}
            continue
        ci_lower, ci_upper, mean, _ = confidence_interval(values, confidence)
        result[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_str": confidence_interval_str(values, confidence),
        }
    return result


def significance_matrix(
    all_models_scores,
    metric_name="accuracy",
    tests=("ttest", "wilcoxon"),
):
    """
    Compute pairwise statistical significance (p-values) between all models across folds.

    Used for Table 5: Statistical Significance Matrix in the plan. Each model must have
    the same number of fold scores (paired comparison).

    Args:
        all_models_scores: Dict mapping model name -> list of metric values (one per fold),
            e.g. {"DINOv2": [0.85, 0.87, 0.84, 0.86, 0.88], "MAE": [0.82, 0.84, ...]}.
        metric_name: Label for the metric (used in returned dict only).
        tests: Which tests to run: "ttest", "wilcoxon", or both.

    Returns:
        dict with:
            "model_names": list of model names (same order as matrix rows/cols),
            "metric_name": metric_name,
            "pvalue_ttest": 2D array of p-values (paired t-test), shape (n_models, n_models).
                Entry (i, j) = p-value for model_i vs model_j (H0: no difference).
            "pvalue_wilcoxon": 2D array of p-values (Wilcoxon signed-rank), same shape.
            "mean_scores": 1D array of mean score per model (same order as model_names).
    """
    model_names = list(all_models_scores.keys())
    n = len(model_names)
    if n == 0:
        return {
            "model_names": [],
            "metric_name": metric_name,
            "pvalue_ttest": np.array([]),
            "pvalue_wilcoxon": np.array([]),
            "mean_scores": np.array([]),
        }

    # Extract score arrays and ensure same length (same folds)
    score_arrays = []
    for name in model_names:
        arr = np.asarray(all_models_scores[name]).ravel()
        score_arrays.append(arr)
    fold_len = len(score_arrays[0])
    for i, arr in enumerate(score_arrays):
        if len(arr) != fold_len:
            raise ValueError(
                f"All models must have the same number of fold scores; "
                f"{model_names[i]} has {len(arr)}, expected {fold_len}"
            )

    mean_scores = np.array([float(np.mean(a)) for a in score_arrays])

    pvalue_ttest = np.full((n, n), np.nan)
    pvalue_wilcoxon = np.full((n, n), np.nan)
    for i in range(n):
        pvalue_ttest[i, i] = 1.0
        pvalue_wilcoxon[i, i] = 1.0
        for j in range(i + 1, n):
            if "ttest" in tests:
                res = paired_ttest(score_arrays[i], score_arrays[j])
                pvalue_ttest[i, j] = res["pvalue"]
                pvalue_ttest[j, i] = res["pvalue"]
            if "wilcoxon" in tests:
                res = wilcoxon_signed_rank(score_arrays[i], score_arrays[j])
                pvalue_wilcoxon[i, j] = res["pvalue"]
                pvalue_wilcoxon[j, i] = res["pvalue"]

    out = {
        "model_names": model_names,
        "metric_name": metric_name,
        "mean_scores": mean_scores,
    }
    if "ttest" in tests:
        out["pvalue_ttest"] = pvalue_ttest
    if "wilcoxon" in tests:
        out["pvalue_wilcoxon"] = pvalue_wilcoxon
    return out


def chi_square_test(observed):
    """
    Chi-square test for independence on a contingency table.

    Args:
        observed: 2D array, contingency table (e.g. confusion matrix or counts).

    Returns:
        dict: {"statistic": float, "pvalue": float, "dof": int}
    """
    observed = np.asarray(observed)
    stat, pval, dof, _ = stats.chi2_contingency(observed)
    return {"statistic": float(stat), "pvalue": float(pval), "dof": int(dof)}


def friedman_test(*samples):
    """
    Friedman test for comparing multiple methods across the same blocks (e.g. folds).

    Args:
        *samples: One array per method, each of length n (number of blocks/folds).
            e.g. friedman_test(scores_model_a, scores_model_b, scores_model_c)

    Returns:
        dict: {"statistic": float, "pvalue": float}
    """
    data = np.column_stack([np.asarray(s).ravel() for s in samples])
    stat, pval = stats.friedmanchisquare(*data.T)
    return {"statistic": float(stat), "pvalue": float(pval)}


def bootstrap_accuracy(y_true, y_pred, n_bootstrap=1000, confidence=0.95, random_state=None):
    """
    Bootstrap confidence interval for accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g. 0.95 for 95% CI, 0.99 for 99% CI).
        random_state: Random seed.

    Returns:
        dict: {"accuracy_mean": float, "ci_lower": float, "ci_upper": float,
               "accuracy_std": float, "ci_95_lower", "ci_95_upper", "ci_99_lower", "ci_99_upper"}
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        acc = np.mean(y_true[idx] == y_pred[idx])
        accs.append(acc)
    accs = np.array(accs)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(accs, 100 * alpha / 2))
    ci_upper = float(np.percentile(accs, 100 * (1 - alpha / 2)))
    ci_95_lo = float(np.percentile(accs, 2.5))
    ci_95_hi = float(np.percentile(accs, 97.5))
    ci_99_lo = float(np.percentile(accs, 0.5))
    ci_99_hi = float(np.percentile(accs, 99.5))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_95_lower": ci_95_lo,
        "ci_95_upper": ci_95_hi,
        "ci_99_lower": ci_99_lo,
        "ci_99_upper": ci_99_hi,
    }


def cohens_d(scores_a, scores_b):
    """
    Compute Cohen's d effect size for comparing two groups.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        scores_a: Metric values for method A.
        scores_b: Metric values for method B.

    Returns:
        dict: {"d": float, "interpretation": str}
    """
    scores_a = np.asarray(scores_a).ravel()
    scores_b = np.asarray(scores_b).ravel()
    
    n1, n2 = len(scores_a), len(scores_b)
    mean1, mean2 = np.mean(scores_a), np.mean(scores_b)
    var1, var2 = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        d = 0.0
    else:
        d = (mean1 - mean2) / pooled_std
    
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {"d": float(d), "interpretation": interpretation}


def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List or array of p-values from multiple tests.
        alpha: Original significance level (default 0.05).

    Returns:
        dict: {
            "corrected_alpha": float,
            "significant": list of bool (one per p-value),
            "adjusted_p_values": list of adjusted p-values (min(p*n, 1))
        }
    """
    p_values = np.asarray(p_values).ravel()
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests if n_tests > 0 else alpha
    adjusted = np.minimum(p_values * n_tests, 1.0)
    significant = p_values < corrected_alpha
    
    return {
        "corrected_alpha": float(corrected_alpha),
        "significant": significant.tolist(),
        "adjusted_p_values": adjusted.tolist(),
        "n_tests": n_tests,
    }


def holm_bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Holm-Bonferroni (step-down) correction for multiple comparisons.
    More powerful than standard Bonferroni while controlling family-wise error rate.

    Args:
        p_values: List or array of p-values from multiple tests.
        alpha: Original significance level (default 0.05).

    Returns:
        dict: {
            "significant": list of bool,
            "adjusted_p_values": list of adjusted p-values,
            "rejection_indices": indices of rejected null hypotheses
        }
    """
    p_values = np.asarray(p_values).ravel()
    n = len(p_values)
    if n == 0:
        return {"significant": [], "adjusted_p_values": [], "rejection_indices": []}
    
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    adjusted = np.zeros(n)
    significant = np.zeros(n, dtype=bool)
    
    running_max = 0.0
    for i in range(n):
        adj_p = sorted_p[i] * (n - i)
        running_max = max(running_max, adj_p)
        adjusted[sorted_idx[i]] = min(running_max, 1.0)
    
    for i in range(n):
        threshold = alpha / (n - i)
        if sorted_p[i] <= threshold:
            significant[sorted_idx[i]] = True
        else:
            break
    
    rejection_indices = np.where(significant)[0].tolist()
    
    return {
        "significant": significant.tolist(),
        "adjusted_p_values": adjusted.tolist(),
        "rejection_indices": rejection_indices,
    }


def significance_matrix_with_correction(
    all_models_scores,
    metric_name="accuracy",
    tests=("ttest", "wilcoxon"),
    correction="bonferroni",
    alpha=0.05,
):
    """
    Compute pairwise statistical significance with multiple comparison correction.
    
    Extends significance_matrix() with:
    - Bonferroni or Holm-Bonferroni correction
    - Cohen's d effect sizes
    - Clear significance indicators

    Args:
        all_models_scores: Dict mapping model name -> list of metric values.
        metric_name: Label for the metric.
        tests: Which tests to run: "ttest", "wilcoxon", or both.
        correction: "bonferroni", "holm", or None.
        alpha: Significance level before correction.

    Returns:
        dict with pvalue matrices, effect sizes, and corrected significance.
    """
    model_names = list(all_models_scores.keys())
    n = len(model_names)
    
    if n < 2:
        return {
            "model_names": model_names,
            "metric_name": metric_name,
            "n_comparisons": 0,
        }
    
    score_arrays = [np.asarray(all_models_scores[name]).ravel() for name in model_names]
    fold_len = len(score_arrays[0])
    
    for i, arr in enumerate(score_arrays):
        if len(arr) != fold_len:
            raise ValueError(
                f"All models must have the same number of fold scores; "
                f"{model_names[i]} has {len(arr)}, expected {fold_len}"
            )
    
    mean_scores = np.array([float(np.mean(a)) for a in score_arrays])
    
    pvalue_ttest = np.full((n, n), np.nan)
    pvalue_wilcoxon = np.full((n, n), np.nan)
    effect_size = np.full((n, n), np.nan)
    effect_interp = [['' for _ in range(n)] for _ in range(n)]
    
    p_values_list = []
    pair_indices = []
    
    for i in range(n):
        pvalue_ttest[i, i] = 1.0
        pvalue_wilcoxon[i, i] = 1.0
        effect_size[i, i] = 0.0
        effect_interp[i][i] = 'same'
        
        for j in range(i + 1, n):
            if "ttest" in tests:
                res = paired_ttest(score_arrays[i], score_arrays[j])
                pvalue_ttest[i, j] = res["pvalue"]
                pvalue_ttest[j, i] = res["pvalue"]
                p_values_list.append(res["pvalue"])
                pair_indices.append((i, j))
            
            if "wilcoxon" in tests:
                try:
                    res = wilcoxon_signed_rank(score_arrays[i], score_arrays[j])
                    pvalue_wilcoxon[i, j] = res["pvalue"]
                    pvalue_wilcoxon[j, i] = res["pvalue"]
                except ValueError:
                    pvalue_wilcoxon[i, j] = 1.0
                    pvalue_wilcoxon[j, i] = 1.0
            
            cd = cohens_d(score_arrays[i], score_arrays[j])
            effect_size[i, j] = cd["d"]
            effect_size[j, i] = -cd["d"]
            effect_interp[i][j] = cd["interpretation"]
            effect_interp[j][i] = cd["interpretation"]
    
    if correction == "bonferroni" and p_values_list:
        corr_result = bonferroni_correction(p_values_list, alpha=alpha)
    elif correction == "holm" and p_values_list:
        corr_result = holm_bonferroni_correction(p_values_list, alpha=alpha)
    else:
        corr_result = {
            "significant": [p < alpha for p in p_values_list],
            "adjusted_p_values": p_values_list,
            "corrected_alpha": alpha,
        }
    
    significant_pairs = []
    adjusted_pvalue_ttest = np.full((n, n), np.nan)
    for idx, ((i, j), adj_p, is_sig) in enumerate(zip(
        pair_indices, corr_result["adjusted_p_values"], corr_result["significant"]
    )):
        adjusted_pvalue_ttest[i, j] = adj_p
        adjusted_pvalue_ttest[j, i] = adj_p
        if is_sig:
            significant_pairs.append((model_names[i], model_names[j], adj_p))
    
    return {
        "model_names": model_names,
        "metric_name": metric_name,
        "mean_scores": mean_scores,
        "pvalue_ttest": pvalue_ttest,
        "pvalue_wilcoxon": pvalue_wilcoxon,
        "adjusted_pvalue_ttest": adjusted_pvalue_ttest,
        "effect_size": effect_size,
        "effect_interpretation": effect_interp,
        "n_comparisons": len(pair_indices),
        "corrected_alpha": corr_result.get("corrected_alpha", alpha),
        "significant_pairs": significant_pairs,
        "correction_method": correction,
    }
