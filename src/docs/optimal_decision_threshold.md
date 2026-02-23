# Optimal Decision Threshold Selection

For binary classification with class imbalance (e.g., normal vs. abnormal), the default decision threshold of 0.5 may not yield the best F1, precision, or recall. This document describes how to optimize the threshold on the validation set and apply it at inference.

## Procedure

1. **Collect validation probabilities**  
   Run the trained model on the validation set and collect predicted probabilities for the positive class (abnormal):

   ```python
   from sklearn.metrics import precision_recall_curve, f1_score

   y_true = ...  # Ground truth (0=normal, 1=abnormal)
   y_proba = ...  # Probability of class 1 from model (e.g. softmax[:, 1])
   ```

2. **Compute precision–recall curve**  
   Use the validation labels and probabilities to get precision and recall at each threshold:

   ```python
   precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
   ```

3. **Choose threshold by F1**  
   For each threshold, compute F1 and pick the one that maximizes F1:

   ```python
   f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
   best_idx = np.argmax(f1_scores)
   optimal_threshold = thresholds[best_idx]
   best_f1 = f1_scores[best_idx]
   ```

4. **Optional: optimize for precision or recall**  
   - To favor **precision**: use the threshold where precision ≥ target (e.g. 0.9).
   - To favor **recall**: use the threshold where recall ≥ target.
   - To balance: use F1 as above or choose a point on the PR curve.

5. **Apply at inference**  
   Use the chosen threshold instead of 0.5:

   ```python
   y_pred = (y_proba >= optimal_threshold).astype(int)
   ```

## Implementation

See [`evaluate_models.py`](../evaluate_models.py): `compute_classification_metrics` and `precision_recall_curve` from `sklearn.metrics` can be used to compute optimal threshold on the validation set. Store `optimal_threshold` in the checkpoint or config for reproducibility.

## References

- Plan: [best_image_classification_models_for_q1_journal](../../.cursor/plans/best_image_classification_models_for_q1_journal_708a366a.plan.md)
- sklearn: [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
