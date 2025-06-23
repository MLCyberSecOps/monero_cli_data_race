#!/usr/bin/env python3
"""
Script to analyze multiple Monero source files using ThreadGuard.
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, List

def analyze_files(file_paths: List[str], output_file: str) -> Dict:
    """Analyze multiple files and collect results."""
    all_results = {
        "files_analyzed": [],
        "total_races": 0,
        "total_issues": 0,
        "file_results": {}
    }
    
    for file_path in file_paths:
        try:
            print(f"\nAnalyzing {file_path}...")
            # Run the analyzer on a single file
            result = subprocess.run(
                ["python", "-m", "threadguard_enhanced", "--json", output_file, file_path],
                capture_output=True,
                text=True
            )
            
            # Print the output for visibility
            print(result.stdout)
            
            # Store the result
            all_results["files_analyzed"].append(file_path)
            all_results["total_issues"] += result.returncode
            
            # Try to parse the JSON output if it exists
            try:
                with open(output_file, 'r') as f:
                    file_result = json.load(f)
                    all_results["file_results"][file_path] = file_result
                    all_results["total_races"] += len(file_result.get("races", []))
            except (FileNotFoundError, json.JSONDecodeError):
                all_results["file_results"][file_path] = {"error": "Failed to parse results"}
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            all_results["file_results"][file_path] = {"error": str(e)}
    
    return all_results

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
        "[MONERO_SRC]/src/cryptonote_basic/connection_context.h"
    ]
    
    output_file = "monero_analysis_results.json"
    
    print(f"Starting analysis of {len(files_to_analyze)} files...")
    results = analyze_files(files_to_analyze, output_file)
    
    # Save summary
    summary = {
        "total_files_analyzed": len(results["files_analyzed"]),
        "total_races_found": results["total_races"],
        "total_issues_found": results["total_issues"],
        "analyzed_files": results["files_analyzed"]
    }
    
    with open("monero_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nAnalysis complete!")
    print(f"Summary saved to monero_analysis_summary.json")
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()
