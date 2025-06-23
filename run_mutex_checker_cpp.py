#!/usr/bin/env python3
"""
Run simple_mutex_checker.py on Monero implementation (.cpp) files.
"""
import json
import os
import subprocess
from pathlib import Path


def find_cpp_files(directory, extensions=(".cpp", ".cxx", ".cc", ".c")):
    """Find all C++ source files in the given directory."""
    cpp_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                cpp_files.append(os.path.join(root, file))
    return cpp_files


def run_mutex_checker(file_paths, output_file):
    """Run mutex checker on multiple files and collect results."""
    results = {
        "files_analyzed": [],
        "files_with_threading": [],
        "files_with_issues": [],
        "total_issues": 0,
        "details": {},
    }

    for file_path in file_paths:
        try:
            print(f"Analyzing {file_path}...")
            result = subprocess.run(
                ["python", "simple_mutex_checker.py", file_path],
                capture_output=True,
                text=True,
            )

            # Store the result
            results["files_analyzed"].append(file_path)

            # Check if threading was detected
            if "Threading: ✅ Detected" in result.stdout:
                results["files_with_threading"].append(file_path)

                # Check if there were any issues
                if (
                    "WARNING:" in result.stdout
                    or "potential issue" in result.stdout.lower()
                ):
                    results["files_with_issues"].append(file_path)
                    results["total_issues"] += 1

            # Store the full output
            results["details"][file_path] = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            results["details"][file_path] = {"error": str(e)}

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    # Directory containing Monero source code
    # Update this path to point to your Monero source directory
    monero_src_dir = "[MONERO_SRC]/src"

    # Find all C++ source files
    print(f"Searching for C++ files in {monero_src_dir}...")
    cpp_files = find_cpp_files(monero_src_dir)

    if not cpp_files:
        print("No C++ files found. Please check the source directory.")
        return

    print(f"Found {len(cpp_files)} C++ files. Starting analysis...")

    output_file = "mutex_checker_cpp_results.json"
    results = run_mutex_checker(
        cpp_files[:50], output_file
    )  # Limit to first 50 files for initial run

    # Print summary
    print("\n=== Analysis Complete ===")
    print(f"Files analyzed: {len(results['files_analyzed'])}")
    print(f"Files with threading: {len(results['files_with_threading'])}")
    print(f"Files with potential issues: {len(results['files_with_issues'])}")
    print(f"Total issues found: {results['total_issues']}")
    print(f"Detailed results saved to: {output_file}")

    if results["files_with_threading"]:
        print("\nFiles with threading detected:")
        for file in results["files_with_threading"]:
            print(f"  - {file}")

    if results["files_with_issues"]:
        print("\nFiles with potential issues:")
        for file in results["files_with_issues"]:
            print(f"  - {file}")


if __name__ == "__main__":
    main()
