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
