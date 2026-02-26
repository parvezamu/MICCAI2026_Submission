# check_results.py
import json

def main():
    results_file = "/home/pahm409/isles_test_results_FINAL/fold_0_test_results.json"

    with open(results_file, "r") as f:
        results = json.load(f)

    print("Previous evaluation results:")
    print(f"  Fold: {results['fold']}")
    print(f"  Mean DSC: {results['mean_dsc']:.4f}")
    print(f"  Num cases: {results['num_test_cases']}")

    print("\nCase IDs evaluated:")
    for i, case_result in enumerate(results["per_case_results"][:10], 1):
        print(
            f"  {i}. {case_result['case_id']} - DSC: {case_result['dsc_original']:.4f}"
        )

    if len(results["per_case_results"]) > 10:
        remaining = len(results["per_case_results"]) - 10
        print(f"  ... and {remaining} more")


if __name__ == "__main__":
    main()

