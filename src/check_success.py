"""
Success Validation Script

Run this after training completes to verify that 90%+ target was achieved.
"""

import os
import sys
import pandas as pd


def check_results(results_dir="results/dinov2"):
    """Check if training results meet 90%+ target."""
    
    print("="*60)
    print(" DINOv2 Enhanced - Results Validation")
    print("="*60)
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"\n❌ ERROR: Results directory not found: {results_dir}")
        print("   Have you run the training yet?")
        return False
    
    # Check if classification report exists
    report_path = os.path.join(results_dir, "classification_report_epoch_best.csv")
    if not os.path.exists(report_path):
        print(f"\n❌ ERROR: Classification report not found: {report_path}")
        print("   Training may not have completed successfully.")
        return False
    
    # Load and parse report
    try:
        report = pd.read_csv(report_path, index_col=0)
    except Exception as e:
        print(f"\n❌ ERROR: Could not read report: {e}")
        return False
    
    # Extract metrics
    try:
        overall_acc = float(report.loc["accuracy", "precision"])
        normal_precision = float(report.loc["0", "precision"])
        normal_recall = float(report.loc["0", "recall"])
        normal_f1 = float(report.loc["0", "f1-score"])
        abnormal_precision = float(report.loc["1", "precision"])
        abnormal_recall = float(report.loc["1", "recall"])
        abnormal_f1 = float(report.loc["1", "f1-score"])
        
        # Calculate macro metrics
        macro_recall = (normal_recall + abnormal_recall) / 2
        macro_f1 = (normal_f1 + abnormal_f1) / 2
        
    except Exception as e:
        print(f"\n❌ ERROR: Could not extract metrics: {e}")
        print("   Report format may be incorrect.")
        return False
    
    # Print current metrics
    print("\n" + "="*60)
    print(" Current Performance")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Overall Accuracy:  {overall_acc:.1%}")
    print(f"  Macro Recall:      {macro_recall:.1%}")
    print(f"  Macro F1:          {macro_f1:.1%}")
    
    print(f"\nClass 0 (Normal) - Support: {int(report.loc['0', 'support'])}")
    print(f"  Precision: {normal_precision:.1%}")
    print(f"  Recall:    {normal_recall:.1%}")
    print(f"  F1 Score:  {normal_f1:.1%}")
    
    print(f"\nClass 1 (Abnormal) - Support: {int(report.loc['1', 'support'])}")
    print(f"  Precision: {abnormal_precision:.1%}")
    print(f"  Recall:    {abnormal_recall:.1%}")
    print(f"  F1 Score:  {abnormal_f1:.1%}")
    
    # Define success criteria
    criteria = {
        "Overall Accuracy >= 90%": (overall_acc, 0.90),
        "Abnormal Recall >= 88%": (abnormal_recall, 0.88),
        "Abnormal F1 >= 82%": (abnormal_f1, 0.82),
        "Normal Recall >= 90%": (normal_recall, 0.90),
        "Macro F1 >= 85%": (macro_f1, 0.85),
    }
    
    # Check each criterion
    print("\n" + "="*60)
    print(" Success Criteria Validation")
    print("="*60)
    
    all_passed = True
    for criterion_name, (value, threshold) in criteria.items():
        passed = value >= threshold
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {criterion_name}: {value:.1%} (threshold: {threshold:.1%})")
        if not passed:
            all_passed = False
    
    # Final verdict
    print("\n" + "="*60)
    if all_passed:
        print("🎉 SUCCESS! ALL CRITERIA MET - 90%+ TARGET ACHIEVED!")
        print("="*60)
        print("\nYour enhanced DINOv2 model is production-ready!")
        print("Key improvement: Abnormal recall increased from 16.6% to {:.1%}".format(abnormal_recall))
    else:
        print("⚠️  PARTIAL SUCCESS - Some criteria not met")
        print("="*60)
        print("\nSuggestions:")
        if abnormal_recall < 0.88:
            print("  - Abnormal recall still low: Try FOCAL_ALPHA=0.1,0.9")
        if overall_acc < 0.90:
            print("  - Overall accuracy low: Train 20 more epochs")
        if normal_recall < 0.90:
            print("  - Normal recall low: Reduce focal loss weight")
        print("\nSee COMPLETE_IMPLEMENTATION_SUMMARY.md for troubleshooting.")
    
    print("\n" + "="*60)
    print(" Detailed Metrics Available In:")
    print("="*60)
    print(f"  - {results_dir}/classification_report_epoch_best.csv")
    print(f"  - {results_dir}/conf_matrix_epoch_best.png")
    print(f"  - {results_dir}/recall_vs_epoch.png")
    print(f"  - {results_dir}/precision_vs_epoch.png")
    print(f"  - {results_dir}/f1_vs_epoch.png")
    print(f"  - {results_dir}/roc_epoch_best.png")
    print(f"  - {results_dir}/pr_curve_epoch_best.png")
    
    return all_passed


if __name__ == "__main__":
    # Allow command-line argument for results directory
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results/dinov2"
    
    success = check_results(results_dir)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
