"""Mutation testing framework for concurrency patterns."""
import ast
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional


class CodeMutator:
    """Base class for code mutators that introduce concurrency issues."""

    def __init__(self, min_mutations: int = 1, max_mutations: int = 3):
        self.min_mutations = min_mutations
        self.max_mutations = max_mutations

    def mutate(self, code: str) -> str:
        """Apply mutations to the code."""
        raise NotImplementedError

    def get_mutation_description(self) -> str:
        """Return a description of the mutation applied."""
        return self.__class__.__name__


class RemoveLockMutator(CodeMutator):
    """Remove lock operations to test missing synchronization."""

    def mutate(self, code: str) -> str:
        lines = code.split("\n")
        lock_lines = [
            i
            for i, line in enumerate(lines)
            if re.search(r"\block\s*\(|lock\.lock\s*\(", line)
        ]

        if not lock_lines:
            return code

        # Randomly select lock lines to remove
        num_mutations = random.randint(
            self.min_mutations, min(self.max_mutations, len(lock_lines))
        )
        to_remove = random.sample(lock_lines, num_mutations)

        # Remove the lock operations
        for i in sorted(to_remove, reverse=True):
            lines.pop(i)

        return "\n".join(lines)

    def get_mutation_description(self) -> str:
        return "Removed one or more lock operations"


class SwapLockOrderMutator(CodeMutator):
    """Swap the order of lock acquisitions to test for potential deadlocks."""

    def mutate(self, code: str) -> str:
        # Find all lock acquisition patterns
        lock_blocks = []
        lines = code.split("\n")

        # Simple pattern matching for lock acquisitions
        for i, line in enumerate(lines):
            if re.search(
                r"std::(mutex|lock_guard|unique_lock|shared_lock|scoped_lock)", line
            ):
                lock_blocks.append((i, line))

        # Need at least two locks to swap
        if len(lock_blocks) < 2:
            return code

        # Select a random pair of locks to swap
        idx1, idx2 = random.sample(range(len(lock_blocks)), 2)
        i1, line1 = lock_blocks[idx1]
        i2, line2 = lock_blocks[idx2]

        # Swap the lines
        lines[i1], lines[i2] = lines[i2], lines[i1]

        return "\n".join(lines)

    def get_mutation_description(self) -> str:
        return "Swapped order of lock acquisitions"


class AddDataRaceMutator(CodeMutator):
    """Add a data race by removing synchronization around shared variable access."""

    def mutate(self, code: str) -> str:
        lines = code.split("\n")

        # Find all critical sections
        critical_sections = []
        in_critical = False
        start_line = 0

        for i, line in enumerate(lines):
            if "lock" in line and ("lock(" in line or "lock_guard" in line):
                in_critical = True
                start_line = i
            elif "unlock" in line or "}" in line and in_critical:
                if start_line < i:  # Ensure we have a valid range
                    critical_sections.append((start_line, i))
                in_critical = False

        if not critical_sections:
            return code

        # Select a critical section to modify
        start, end = random.choice(critical_sections)

        # Remove the lock/unlock
        lines.pop(start)  # Remove lock

        # Find and remove matching unlock
        for i in range(start, min(start + 20, len(lines))):  # Look ahead up to 20 lines
            if "unlock" in lines[i] or "}" in lines[i]:
                lines.pop(i)
                break

        return "\n".join(lines)

    def get_mutation_description(self) -> str:
        return "Removed synchronization around shared variable access"


def run_mutation_test(
    original_code: str,
    test_func: Callable,
    mutators: List[CodeMutator],
    num_iterations: int = 10,
) -> Dict[str, List[str]]:
    """
    Run mutation testing on the given code.

    Args:
        original_code: The original, correct code
        test_func: A function that takes code as input and returns True if the test passes
        mutators: List of mutators to apply
        num_iterations: Number of mutation iterations to run

    Returns:
        Dictionary mapping mutation types to lists of test results
    """
    results = {}

    for _ in range(num_iterations):
        # Select a random mutator
        mutator = random.choice(mutators)

        # Apply mutation
        mutated_code = mutator.mutate(original_code)

        # Skip if no mutation was applied
        if mutated_code == original_code:
            continue

        # Run the test
        try:
            test_passed = test_func(mutated_code)
            status = "PASSED" if test_passed else "FAILED"
        except Exception as e:
            status = f"ERROR: {str(e)}"

        # Record results
        mutator_name = mutator.get_mutation_description()
        if mutator_name not in results:
            results[mutator_name] = []
        results[mutator_name].append(status)

    return results


def print_mutation_test_results(results: Dict[str, List[str]]):
    """Print a summary of mutation test results."""
    print("\nMutation Test Results:")
    print("=" * 50)

    for mutator, outcomes in results.items():
        passed = outcomes.count("PASSED")
        failed = outcomes.count("FAILED")
        errors = len(outcomes) - passed - failed

        print(f"{mutator}:")
        print(f"  Total runs: {len(outcomes)}")
        print(f"  Passed: {passed} ({(passed/len(outcomes))*100:.1f}%)")
        print(f"  Failed: {failed} ({(failed/len(outcomes))*100:.1f}%)")
        if errors > 0:
            print(f"  Errors: {errors}")
        print()

    # Calculate overall mutation score
    total = sum(len(v) for v in results.values())
    if total > 0:
        passed = sum(1 for v in results.values() for s in v if s == "PASSED")
        print(f"Overall Mutation Score: {(passed/total)*100:.1f}%")
    else:
        print("No mutations were applied.")
    print("=" * 50)
