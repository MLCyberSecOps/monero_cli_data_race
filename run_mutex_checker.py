#!/usr/bin/env python3
"""
Run simple_mutex_checker.py on all Monero source files.
"""
import json
import subprocess
from pathlib import Path


def run_mutex_checker(file_paths, output_file):
    """Run mutex checker on multiple files and collect results."""
    results = {
        "files_analyzed": [],
        "files_with_issues": [],
        "total_issues": 0,
        "details": {},
    }

    for file_path in file_paths:
        try:
            print(f"Analyzing {file_path}...")
            result = subprocess.run(
                ["python", "simple_mutex_checker.py", str(file_path)],
                capture_output=True,
                text=True,
            )

            # Store the result
            results["files_analyzed"].append(str(file_path))

            # Check if there were any issues
            if (
                "WARNING:" in result.stdout
                or "potential issue" in result.stdout.lower()
            ):
                results["files_with_issues"].append(str(file_path))
                results["total_issues"] += 1

            # Store the full output
            results["details"][str(file_path)] = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            results["details"][str(file_path)] = {"error": str(e)}

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    # List of files to analyze
    files_to_analyze = [
        "[MONERO_SRC]/contrib/epee/include/misc_language.h",
        "[MONERO_SRC]/contrib/epee/include/console_handler.h",
        "[MONERO_SRC]/contrib/epee/include/string_coding.h",
        "[MONERO_SRC]/contrib/epee/include/misc_log_ex.h",
        "[MONERO_SRC]/contrib/epee/include/string_tools.h",
        "[MONERO_SRC]/contrib/epee/include/syncobj.h",
        "[MONERO_SRC]/contrib/epee/include/net/http_client.h",
        "[MONERO_SRC]/contrib/epee/include/net/abstract_tcp_server2.h",
        "[MONERO_SRC]/contrib/epee/include/net/levin_base.h",
        "[MONERO_SRC]/contrib/epee/src/net_parse_helpers.cpp",
        "[MONERO_SRC]/contrib/epee/src/string_tools.cpp",
        "[MONERO_SRC]/contrib/epee/src/portable_storage.cpp",
        "[MONERO_SRC]/contrib/epee/src/misc_language.cpp",
        "[MONERO_SRC]/src/cryptonote_basic/connection_context.h",
    ]

    output_file = "mutex_checker_results.json"

    print(f"Starting mutex checker analysis on {len(files_to_analyze)} files...")
    results = run_mutex_checker(files_to_analyze, output_file)

    # Print summary
    print("\n=== Analysis Complete ===")
    print(f"Files analyzed: {len(results['files_analyzed'])}")
    print(f"Files with potential issues: {len(results['files_with_issues'])}")
    print(f"Total issues found: {results['total_issues']}")
    print(f"Detailed results saved to: {output_file}")

    if results["files_with_issues"]:
        print("\nFiles with potential issues:")
        for file in results["files_with_issues"]:
            print(f"  - {file}")


if __name__ == "__main__":
    main()
