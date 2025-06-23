#!/usr/bin/env python3
"""
ThreadGuard Enhanced - Fixes for Test Failures

This module contains fixes for the specific test failures:
1. Missing unlock detection with early returns
2. Complex deadlock detection in nested method calls
3. Vector race detection for STL containers

Authors: Pradeep Kumar
Version: 1.3.0
Date: 2025-06-23
"""

import ast
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ControlFlowPath:
    """Represents a control flow path through a function"""

    function_name: str
    entry_line: int
    exit_line: int
    path_type: str  # "normal", "early_return", "exception", "break", "continue"
    locks_acquired: List[str]
    locks_released: List[str]
    has_early_exit: bool = False


@dataclass
class STLContainerAccess:
    """Represents access to STL containers"""

    container_name: str
    container_type: str  # "vector", "map", "set", etc.
    access_method: str  # "push_back", "insert", "erase", "[]", etc.
    is_thread_safe: bool
    line_number: int
    function: str


class EnhancedThreadGuardAnalyzer:
    """
    Enhanced analyzer with fixes for test failures
    """

    def __init__(self):
        # Initialize base analyzer components
        self.shared_vars = set()
        self.global_vars = set()
        self.memory_accesses = defaultdict(list)
        self.locking_patterns = defaultdict(list)

        # Enhanced tracking for fixes
        self.control_flow_paths = defaultdict(list)
        self.stl_container_accesses = defaultdict(list)
        self.method_call_graph = defaultdict(set)
        self.recursive_locks = defaultdict(set)

        # STL containers that are NOT thread-safe
        self.non_thread_safe_containers = {
            "std::vector",
            "std::deque",
            "std::list",
            "std::forward_list",
            "std::array",
            "std::map",
            "std::multimap",
            "std::set",
            "std::multiset",
            "std::unordered_map",
            "std::unordered_multimap",
            "std::unordered_set",
            "std::unordered_multiset",
            "std::string",
            "std::wstring",
        }

        # Thread-safe containers (C++11+)
        self.thread_safe_containers = {
            "std::atomic",
            "std::shared_ptr",  # Limited thread safety
        }

        # Compile enhanced patterns
        self._compile_enhanced_patterns()

    def _compile_enhanced_patterns(self):
        """Compile enhanced regex patterns for better detection"""
        self.enhanced_patterns = {
            # Control flow patterns
            "early_return": re.compile(r"^\s*return\s*[^;]*;"),
            "if_statement": re.compile(r"^\s*if\s*\([^)]+\)\s*{?"),
            "else_statement": re.compile(r"^\s*else\s*{?"),
            "while_loop": re.compile(r"^\s*while\s*\([^)]+\)\s*{?"),
            "for_loop": re.compile(r"^\s*for\s*\([^)]*\)\s*{?"),
            "switch_statement": re.compile(r"^\s*switch\s*\([^)]+\)\s*{?"),
            "break_statement": re.compile(r"^\s*break\s*;"),
            "continue_statement": re.compile(r"^\s*continue\s*;"),
            "throw_statement": re.compile(r"^\s*throw\s+[^;]*;"),
            # Lock patterns with more context
            "explicit_lock": re.compile(r"(\w+)\.lock\s*\(\s*\)"),
            "explicit_unlock": re.compile(r"(\w+)\.unlock\s*\(\s*\)"),
            "try_lock": re.compile(r"(\w+)\.try_lock\s*\(\s*\)"),
            # RAII patterns
            "lock_guard_full": re.compile(
                r"std\s*::\s*(lock_guard|unique_lock|shared_lock|scoped_lock)\s*<[^>]+>\s+"
                r"(\w+)\s*\(\s*([^)]+)\s*\)"
            ),
            # STL container patterns
            "vector_declaration": re.compile(r"std\s*::\s*vector\s*<[^>]+>\s+(\w+)"),
            "map_declaration": re.compile(
                r"std\s*::\s*(?:unordered_)?map\s*<[^>]+>\s+(\w+)"
            ),
            "set_declaration": re.compile(
                r"std\s*::\s*(?:unordered_)?set\s*<[^>]+>\s+(\w+)"
            ),
            "string_declaration": re.compile(r"std\s*::\s*(?:w)?string\s+(\w+)"),
            # Container method calls
            "container_method": re.compile(
                r"(\w+)\.(push_back|insert|erase|clear|resize|at|operator\[\]|\[\])\s*\("
            ),
            "container_iterator": re.compile(
                r"(\w+)\.(begin|end|rbegin|rend|find|lower_bound|upper_bound)\s*\("
            ),
            # Method calls and nested calls
            "method_call": re.compile(r"(\w+)\s*\.\s*(\w+)\s*\([^)]*\)"),
            "function_call": re.compile(r"(\w+)\s*\([^)]*\)"),
            # Class and method definitions
            "class_definition": re.compile(r"class\s+(\w+)(?:\s*:\s*[^{]*)?{"),
            "method_definition": re.compile(
                r"(?:(?:inline|static|virtual|explicit|constexpr)\s+)*"
                r"(?:\w+(?:\s*::\s*\w+)*\s+)?"
                r"(?:\w+\s*::\s*)?(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?[{;]"
            ),
        }

    def analyze_control_flow_paths(self, content: str, filename: str):
        """
        FIX 1: Enhanced control flow analysis to detect missing unlocks
        """
        lines = content.split("\n")
        current_function = ""
        current_class = ""

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track class context
            class_match = self.enhanced_patterns["class_definition"].search(line)
            if class_match:
                current_class = class_match.group(1)
                continue

            # Track method definitions
            method_match = self.enhanced_patterns["method_definition"].search(line)
            if method_match:
                current_function = method_match.group(1)
                # Analyze this function for control flow issues
                self._analyze_function_control_flow(
                    lines, i, current_function, current_class
                )
                continue

    def _analyze_function_control_flow(
        self, lines: List[str], start_line: int, function_name: str, class_name: str
    ):
        """
        Analyze a single function for control flow and lock management issues
        """
        function_lines = []
        brace_count = 0
        in_function = False

        # Extract function body
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            function_lines.append((i + 1, line))

            if "{" in line:
                in_function = True
                brace_count += line.count("{")

            if in_function:
                brace_count += line.count("{") - line.count("}")

                if brace_count <= 0:
                    break

        if not function_lines:
            return

        # Analyze paths through this function
        paths = self._extract_control_flow_paths(function_lines, function_name)

        for path in paths:
            self._check_path_for_lock_issues(path, function_name, class_name)

    def _extract_control_flow_paths(
        self, function_lines: List[Tuple[int, str]], function_name: str
    ) -> List[ControlFlowPath]:
        """
        Extract all possible execution paths through a function
        """
        paths = []
        locks_in_scope = []
        current_path = ControlFlowPath(
            function_name=function_name,
            entry_line=function_lines[0][0] if function_lines else 0,
            exit_line=function_lines[-1][0] if function_lines else 0,
            path_type="normal",
            locks_acquired=[],
            locks_released=[],
        )

        for line_num, line in function_lines:
            stripped = line.strip()

            # Check for lock operations
            lock_match = self.enhanced_patterns["explicit_lock"].search(line)
            if lock_match:
                mutex_name = lock_match.group(1)
                current_path.locks_acquired.append(mutex_name)
                locks_in_scope.append(mutex_name)

            unlock_match = self.enhanced_patterns["explicit_unlock"].search(line)
            if unlock_match:
                mutex_name = unlock_match.group(1)
                current_path.locks_released.append(mutex_name)
                if mutex_name in locks_in_scope:
                    locks_in_scope.remove(mutex_name)

            # Check for RAII locks
            raii_match = self.enhanced_patterns["lock_guard_full"].search(line)
            if raii_match:
                lock_type = raii_match.group(1)
                var_name = raii_match.group(2)
                mutex_name = raii_match.group(3)
                # RAII locks are automatically released at scope end
                current_path.locks_acquired.append(mutex_name)
                # Don't add to locks_in_scope as RAII handles it

            # Check for early returns
            if self.enhanced_patterns["early_return"].search(stripped):
                # Found early return - check if locks are properly released
                early_return_path = ControlFlowPath(
                    function_name=function_name,
                    entry_line=current_path.entry_line,
                    exit_line=line_num,
                    path_type="early_return",
                    locks_acquired=current_path.locks_acquired.copy(),
                    locks_released=current_path.locks_released.copy(),
                    has_early_exit=True,
                )

                # Check if there are unreleased locks
                unreleased = set(current_path.locks_acquired) - set(
                    current_path.locks_released
                )
                if unreleased:
                    early_return_path.path_type = "early_return_with_leaked_locks"

                paths.append(early_return_path)

            # Check for throw statements
            if self.enhanced_patterns["throw_statement"].search(stripped):
                throw_path = ControlFlowPath(
                    function_name=function_name,
                    entry_line=current_path.entry_line,
                    exit_line=line_num,
                    path_type="exception",
                    locks_acquired=current_path.locks_acquired.copy(),
                    locks_released=current_path.locks_released.copy(),
                    has_early_exit=True,
                )
                paths.append(throw_path)

        # Add normal completion path
        if current_path.path_type == "normal":
            paths.append(current_path)

        return paths

    def _check_path_for_lock_issues(
        self, path: ControlFlowPath, function_name: str, class_name: str
    ):
        """
        Check a specific execution path for lock-related issues
        """
        acquired = set(path.locks_acquired)
        released = set(path.locks_released)
        unreleased = acquired - released

        if unreleased and path.has_early_exit:
            for mutex in unreleased:
                issue = (
                    f"Missing unlock for '{mutex}' in {function_name}() "
                    f"on {path.path_type} path at line {path.exit_line}"
                )
                if hasattr(self, "result"):
                    self.result.deadlock_risks.append(issue)

    def analyze_stl_container_usage(self, content: str, filename: str):
        """
        FIX 3: Analyze STL container usage for thread safety issues
        """
        lines = content.split("\n")
        current_function = ""
        current_class = ""
        container_vars = {}  # var_name -> container_type

        for i, line in enumerate(lines, 1):
            # Track class and function context
            class_match = self.enhanced_patterns["class_definition"].search(line)
            if class_match:
                current_class = class_match.group(1)
                continue

            method_match = self.enhanced_patterns["method_definition"].search(line)
            if method_match:
                current_function = method_match.group(1)
                continue

            # Find container declarations
            for pattern_name, pattern in [
                ("vector", self.enhanced_patterns["vector_declaration"]),
                ("map", self.enhanced_patterns["map_declaration"]),
                ("set", self.enhanced_patterns["set_declaration"]),
                ("string", self.enhanced_patterns["string_declaration"]),
            ]:
                match = pattern.search(line)
                if match:
                    var_name = match.group(1)
                    container_vars[var_name] = f"std::{pattern_name}"
                    # Add to shared vars if it's a member variable
                    if var_name.startswith("m_") or current_class:
                        self.shared_vars.add(var_name)

            # Find container method calls
            container_method_match = self.enhanced_patterns["container_method"].search(
                line
            )
            if container_method_match:
                var_name = container_method_match.group(1)
                method_name = container_method_match.group(2)

                if var_name in container_vars:
                    container_type = container_vars[var_name]

                    # Check if this is a thread-unsafe operation
                    is_thread_safe = self._is_container_operation_thread_safe(
                        container_type, method_name
                    )

                    access = STLContainerAccess(
                        container_name=var_name,
                        container_type=container_type,
                        access_method=method_name,
                        is_thread_safe=is_thread_safe,
                        line_number=i,
                        function=current_function or "<global>",
                    )

                    self.stl_container_accesses[var_name].append(access)

                    # If it's not thread safe and it's a shared variable, flag it
                    if not is_thread_safe and var_name in self.shared_vars:
                        self._check_container_synchronization(access, line, lines, i)

    def _is_container_operation_thread_safe(
        self, container_type: str, method: str
    ) -> bool:
        """
        Determine if a container operation is thread-safe
        """
        # Most STL containers are NOT thread-safe for write operations
        write_operations = {
            "push_back",
            "insert",
            "erase",
            "clear",
            "resize",
            "pop_back",
            "pop_front",
            "assign",
            "swap",
            "operator[]",
        }

        read_operations = {
            "size",
            "empty",
            "at",
            "front",
            "back",
            "begin",
            "end",
            "find",
            "count",
            "lower_bound",
            "upper_bound",
        }

        # Even read operations can be unsafe during concurrent writes
        if container_type in self.non_thread_safe_containers:
            return False

        if container_type in self.thread_safe_containers:
            return True

        return False

    def _check_container_synchronization(
        self, access: STLContainerAccess, line: str, lines: List[str], line_num: int
    ):
        """
        Check if container access is properly synchronized
        """
        # Look for synchronization in surrounding context
        context_start = max(0, line_num - 5)
        context_end = min(len(lines), line_num + 3)
        context = "\n".join(lines[context_start:context_end])

        # Check for various synchronization mechanisms
        sync_patterns = [
            r"lock_guard",
            r"unique_lock",
            r"shared_lock",
            r"scoped_lock",
            r"\.lock\s*\(",
            r"\.unlock\s*\(",
            r"std::atomic",
            r"mutex",
            r"critical_section",
        ]

        has_synchronization = any(
            re.search(pattern, context, re.IGNORECASE) for pattern in sync_patterns
        )

        if not has_synchronization:
            issue = (
                f"Unsynchronized access to STL container '{access.container_name}' "
                f"({access.container_type}) in {access.function}() at line {line_num}. "
                f"Method '{access.access_method}' is not thread-safe."
            )

            if hasattr(self, "result"):
                self.result.inconsistent_locking.append(issue)

    def analyze_complex_deadlock_scenarios(self, content: str, filename: str):
        """
        FIX 2: Enhanced deadlock detection for complex scenarios with nested calls
        """
        lines = content.split("\n")
        current_class = ""
        current_method = ""

        # Build method call graph with lock information
        class_methods = defaultdict(dict)  # class -> {method -> {locks, calls}}

        for i, line in enumerate(lines, 1):
            # Track class context
            class_match = self.enhanced_patterns["class_definition"].search(line)
            if class_match:
                current_class = class_match.group(1)
                continue

            # Track method definitions
            method_match = self.enhanced_patterns["method_definition"].search(line)
            if method_match:
                current_method = method_match.group(1)
                if current_class:
                    class_methods[current_class][current_method] = {
                        "locks": [],
                        "calls": [],
                        "line": i,
                    }
                continue

            if not current_class or not current_method:
                continue

            # Track lock operations in this method
            lock_match = self.enhanced_patterns["explicit_lock"].search(line)
            if lock_match:
                mutex_name = lock_match.group(1)
                class_methods[current_class][current_method]["locks"].append(mutex_name)

            raii_match = self.enhanced_patterns["lock_guard_full"].search(line)
            if raii_match:
                mutex_name = raii_match.group(3)
                class_methods[current_class][current_method]["locks"].append(mutex_name)

            # Track method calls
            method_call_match = self.enhanced_patterns["method_call"].search(line)
            if method_call_match:
                obj_or_class = method_call_match.group(1)
                called_method = method_call_match.group(2)
                class_methods[current_class][current_method]["calls"].append(
                    (called_method, i, obj_or_class)
                )

        # Analyze for complex deadlock scenarios
        self._detect_nested_deadlock_risks(class_methods)

    def _detect_nested_deadlock_risks(self, class_methods: Dict):
        """
        Detect deadlock risks in nested method calls
        """
        for class_name, methods in class_methods.items():
            for method_name, method_info in methods.items():
                locks_held = method_info["locks"]
                calls_made = method_info["calls"]

                if not locks_held:
                    continue

                # Check each method call made while holding locks
                for called_method, call_line, obj_name in calls_made:
                    # Check if the called method might acquire locks
                    potential_locks = self._get_potential_locks_in_call_chain(
                        class_name, called_method, class_methods, visited=set()
                    )

                    if potential_locks:
                        # Check for potential deadlock scenarios
                        for held_lock in locks_held:
                            for potential_lock in potential_locks:
                                if held_lock != potential_lock:
                                    # Potential deadlock if another thread does the reverse
                                    issue = (
                                        f"Potential deadlock in {class_name}::{method_name}() "
                                        f"at line {call_line}: holds '{held_lock}' while calling "
                                        f"{called_method}() which may acquire '{potential_lock}'"
                                    )

                                    if hasattr(self, "result"):
                                        self.result.deadlock_risks.append(issue)

    def _get_potential_locks_in_call_chain(
        self,
        class_name: str,
        method_name: str,
        class_methods: Dict,
        visited: Set[str],
        depth: int = 0,
    ) -> Set[str]:
        """
        Recursively find all locks that might be acquired in a call chain
        """
        if depth > 5:  # Prevent infinite recursion
            return set()

        call_key = f"{class_name}::{method_name}"
        if call_key in visited:
            return set()

        visited.add(call_key)
        potential_locks = set()

        # Check if this method exists in our analysis
        if class_name in class_methods and method_name in class_methods[class_name]:
            method_info = class_methods[class_name][method_name]

            # Add direct locks
            potential_locks.update(method_info["locks"])

            # Recursively check called methods
            for called_method, _, _ in method_info["calls"]:
                nested_locks = self._get_potential_locks_in_call_chain(
                    class_name, called_method, class_methods, visited.copy(), depth + 1
                )
                potential_locks.update(nested_locks)

        return potential_locks


# Integration function to apply fixes to existing analyzer
def apply_enhanced_fixes(analyzer_instance, content: str, filename: str):
    """
    Apply the enhanced fixes to an existing ThreadGuardAnalyzer instance
    """
    enhancer = EnhancedThreadGuardAnalyzer()

    # Copy shared state
    enhancer.shared_vars = analyzer_instance.shared_vars
    enhancer.global_vars = analyzer_instance.global_vars
    enhancer.memory_accesses = analyzer_instance.memory_accesses
    enhancer.locking_patterns = analyzer_instance.locking_patterns
    enhancer.result = analyzer_instance.result

    # Apply enhanced analyses
    enhancer.analyze_control_flow_paths(content, filename)
    enhancer.analyze_stl_container_usage(content, filename)
    enhancer.analyze_complex_deadlock_scenarios(content, filename)

    # Copy results back
    analyzer_instance.stl_container_accesses = enhancer.stl_container_accesses
    analyzer_instance.control_flow_paths = enhancer.control_flow_paths

    return analyzer_instance


# Example usage and test cases
def create_test_cases():
    """
    Create test cases that should pass with the enhanced analyzer
    """
    test_cases = {
        "missing_unlock_early_return.cpp": """
#include <mutex>
#include <iostream>

class TestClass {
private:
    std::mutex m_mutex;
    int m_data;

public:
    bool process_data(bool condition) {
        m_mutex.lock();  // Lock acquired

        if (condition) {
            std::cout << "Early return" << std::endl;
            return false;  // Missing unlock here - should be detected
        }

        m_data = 42;
        m_mutex.unlock();
        return true;
    }
};
""",
        "vector_race_condition.cpp": """
#include <vector>
#include <thread>

class DataManager {
private:
    std::vector<int> m_data;  // Shared container

public:
    void add_data(int value) {
        m_data.push_back(value);  // Unsynchronized access - should be detected
    }

    size_t get_size() {
        return m_data.size();  // Also unsynchronized - should be detected
    }
};

void worker_thread(DataManager& manager, int value) {
    manager.add_data(value);
}
""",
        "complex_deadlock_scenario.cpp": """
#include <mutex>

class ResourceManager {
private:
    std::mutex m_resource_mutex;
    std::mutex m_log_mutex;

public:
    void acquire_resource() {
        std::lock_guard<std::mutex> lock(m_resource_mutex);
        log_action("Resource acquired");  // Calls method that locks m_log_mutex
    }

    void log_and_acquire() {
        std::lock_guard<std::mutex> lock(m_log_mutex);
        acquire_resource();  // Potential deadlock - different lock order
    }

private:
    void log_action(const std::string& message) {
        std::lock_guard<std::mutex> lock(m_log_mutex);
        // Log the message
    }
};
""",
    }

    return test_cases


if __name__ == "__main__":
    # Test the enhanced fixes
    test_cases = create_test_cases()

    for filename, content in test_cases.items():
        print(f"\n=== Testing {filename} ===")
        enhancer = EnhancedThreadGuardAnalyzer()

        # Simulate result object
        from types import SimpleNamespace

        enhancer.result = SimpleNamespace()
        enhancer.result.deadlock_risks = []
        enhancer.result.inconsistent_locking = []

        # Run enhanced analyses
        enhancer.analyze_control_flow_paths(content, filename)
        enhancer.analyze_stl_container_usage(content, filename)
        enhancer.analyze_complex_deadlock_scenarios(content, filename)

        # Print results
        if enhancer.result.deadlock_risks:
            print("Deadlock Risks Found:")
            for risk in enhancer.result.deadlock_risks:
                print(f"  - {risk}")

        if enhancer.result.inconsistent_locking:
            print("Locking Issues Found:")
            for issue in enhancer.result.inconsistent_locking:
                print(f"  - {issue}")

        if (
            not enhancer.result.deadlock_risks
            and not enhancer.result.inconsistent_locking
        ):
            print("No issues detected (may need integration with main analyzer)")
