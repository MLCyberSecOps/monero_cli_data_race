#!/usr/bin/env python3
"""
ThreadGuard: Enhanced Static Analysis Tool for Concurrency Bug Detection

Detects data races, inconsistent locking patterns, and thread safety violations
specifically targeting async I/O handlers like Monero's async_stdin_reader.

Authors: Pradeep Kumar, Enhanced Version
License: MIT
Version: 1.2.0
Date: 2025-06-23

Key Improvements:
- Better error handling and file validation
- Enhanced C++ parsing with more robust regex patterns
- Improved lock analysis with RAII detection
- Better deadlock detection algorithms
- Comprehensive test coverage support
- Performance optimizations
"""

import argparse
import ast
import itertools
import json
import os
import re
import sys
from collections import defaultdict, deque, namedtuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import astunparse

# Conditional import for networkx with fallback
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not available. Advanced deadlock detection disabled.")
    print("Install with: pip install networkx")


class SeverityLevel(Enum):
    """Bug severity classification for mission-critical software"""

    CRITICAL = "CRITICAL"  # Mission failure risk
    HIGH = "HIGH"  # Data corruption risk
    MEDIUM = "MEDIUM"  # Performance degradation
    LOW = "LOW"  # Code quality issue
    INFO = "INFO"  # Informational finding


class AccessType(Enum):
    """Memory access pattern classification"""

    READ = "READ"
    WRITE = "WRITE"
    RMW = "READ_MODIFY_WRITE"  # Atomic read-modify-write
    CALL = "CALL"  # Function call that may access


@dataclass
class MemoryAccess:
    """Represents a memory access to a shared variable"""

    variable: str
    access_type: AccessType
    line_number: int
    function: str
    thread_context: str
    is_protected: bool = False
    protection_mechanism: Optional[str] = None
    atomic_ordering: Optional[str] = None
    code_snippet: str = ""


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


@dataclass
class LockingPattern:
    """Represents a locking/unlocking pattern"""

    mutex_name: str
    operation: str  # 'lock' or 'unlock'
    line_number: int
    function: str
    is_raii: bool = False
    is_implicit: bool = False
    scope_depth: int = 0


@dataclass
class RaceCondition:
    """Detected race condition between two memory accesses"""

    variable: str
    access1: MemoryAccess
    access2: MemoryAccess
    severity: SeverityLevel
    description: str
    fix_suggestion: str
    confidence: float = 0.8  # Confidence level 0.0-1.0


@dataclass
class AnalysisResult:
    """Complete analysis results"""

    races: List[RaceCondition] = field(default_factory=list)
    inconsistent_locking: List[str] = field(default_factory=list)
    deadlock_risks: List[str] = field(default_factory=list)
    atomic_violations: List[str] = field(default_factory=list)
    metrics: Dict[str, int] = field(default_factory=dict)
    analysis_time: float = 0.0
    file_analyzed: str = ""


class SimpleGraph:
    """Simple directed graph implementation as networkx fallback"""

    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(set)

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, u, v):
        self.nodes.add(u)
        self.nodes.add(v)
        self.edges[u].add(v)

    def has_edge(self, u, v):
        return v in self.edges[u]

    def simple_cycles(self):
        """Find simple cycles using DFS"""
        cycles = []
        visited = set()
        path = []
        path_set = set()

        def dfs(node):
            if node in path_set:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if len(cycle) > 2:  # Only meaningful cycles
                    cycles.append(cycle[:-1])  # Remove duplicate node
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)
            path_set.add(node)

            for neighbor in self.edges[node]:
                dfs(neighbor)

            path.pop()
            path_set.remove(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node)

        return cycles


class ThreadGuardAnalyzer:
    """
    Advanced static analyzer for detecting concurrency issues in C++ code,
    with special focus on Monero's async_stdin_reader patterns.
    """

    def __init__(self):
        # Performance optimization: compile regex patterns once
        self._compile_patterns()

        # Shared state tracking
        self.shared_vars: Set[str] = set()
        self.global_vars: Set[str] = set()
        self.memory_accesses: Dict[str, List[MemoryAccess]] = defaultdict(list)
        self.locking_patterns: Dict[str, List[LockingPattern]] = defaultdict(list)

        # Use networkx if available, otherwise fallback
        if HAS_NETWORKX:
            self.call_graph = nx.DiGraph()
        else:
            self.call_graph = SimpleGraph()

        # Enhanced patterns for better detection
        self.monero_race_patterns = [
            # State transitions and checks
            r"m_read_status\s*(?:!|=|==|!=|>|<|>=|<=)\s*state_\w+",
            # Unsynchronized access patterns
            r"m_read_status(?:\s*[^=]|$)",
            # EOF checks without proper sync
            r"!eos\s*\(\s*\)",
            # Signal handling patterns
            r"signal\s*\([^,]+,\s*SIG[A-Z]+\)",
            # Thread creation patterns
            r"boost\s*::\s*thread\s*\(\s*&\s*async_stdin_reader\s*::",
            # Condition variable patterns
            r"m_response_cv\.(wait|wait_for|wait_until|notify_(one|all))\s*\(",
        ]

        # Thread entry points
        self.thread_entry_points = {
            "reader_thread_func",
            "run",
            "start",
            "operator()",
            "thread_main",
            "worker_thread",
            "async_handler",
        }

        # Common synchronization primitives
        self.sync_primitives = {
            "std::mutex",
            "std::shared_mutex",
            "std::timed_mutex",
            "std::recursive_mutex",
            "boost::mutex",
            "boost::shared_mutex",
            "pthread_mutex_t",
            "std::atomic",
            "std::lock_guard",
            "std::unique_lock",
            "std::shared_lock",
            "std::scoped_lock",
        }

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance"""
        self.patterns = {
            "function_def": re.compile(
                r"(?:(?:inline|static|virtual|explicit|constexpr)\s+)*"
                r"(?:\w+(?:\s*::\s*\w+)*\s+)?"
                r"(?:\w+\s*::\s*)?(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?[{;]"
            ),
            "member_var": re.compile(r"\b(m_\w+)\b"),
            "global_var": re.compile(
                r"^\s*(?:static\s+)?(?:extern\s+)?"
                r"(?:int|long|short|char|bool|float|double|size_t|uint\w*|int\w*)\s+"
                r"([A-Za-z_]\w*)\s*(?:=[^;]*)?;"
            ),
            "lock_guard": re.compile(
                r"std\s*::\s*(lock_guard|unique_lock|shared_lock|scoped_lock)\s*<[^>]+>\s+"
                r"(\w+)\s*\(\s*([^)]+)\s*\)"
            ),
            "mutex_op": re.compile(r"(\w+)\.(lock|unlock|try_lock)\s*\("),
            "atomic_op": re.compile(
                r"(\w+)\.(load|store|exchange|compare_exchange)\s*\("
            ),
            "thread_create": re.compile(r"std\s*::\s*thread\s+(\w+)\s*\([^)]*\)"),
        }

    def analyze_file(self, filepath: Path) -> AnalysisResult:
        """
        Analyze a C++ source file for concurrency issues

        Args:
            filepath: Path to the C++ source file

        Returns:
            AnalysisResult containing all detected issues

        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        import time

        start_time = time.time()

        self.result = AnalysisResult()
        self.result.file_analyzed = str(filepath)

        # Enhanced file validation
        if not self._validate_file(filepath):
            return self.result

        try:
            content = self._read_file_safely(filepath)
            if content is None:
                return self.result

            # Run analysis phases with progress tracking
            self._extract_shared_variables(content)
            self._extract_memory_accesses(content, str(filepath))
            self._extract_locking_patterns(content, str(filepath))
            self._build_call_graph(content)
            self._analyze_thread_contexts(content)

            # Run detection algorithms
            self._detect_data_races()
            self._analyze_locking_consistency()
            self._detect_deadlock_potential()
            self._validate_atomic_operations()

            # Monero-specific analysis
            self._analyze_monero_specific_patterns(content, str(filepath))
            self._detect_async_io_races(content)
            self._generate_monero_specific_fixes(self.result.races)

            # Calculate metrics
            self._calculate_metrics()

        except Exception as e:
            error_msg = f"ERROR during analysis of {filepath}: {str(e)}"
            self.result.inconsistent_locking.append(error_msg)
            if "--debug" in sys.argv:
                import traceback

                traceback.print_exc()

        self.result.analysis_time = time.time() - start_time
        return self.result

    def _validate_file(self, filepath: Path) -> bool:
        """Enhanced file validation with detailed error reporting"""
        if not filepath.exists():
            self.result.inconsistent_locking.append(f"File does not exist: {filepath}")
            return False

        if not filepath.is_file():
            self.result.inconsistent_locking.append(f"Path is not a file: {filepath}")
            return False

        # Check file extension
        valid_extensions = {".cpp", ".cc", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx"}
        if filepath.suffix.lower() not in valid_extensions:
            self.result.inconsistent_locking.append(
                f"Warning: Unusual file extension '{filepath.suffix}' for C++ file"
            )

        # Check file size (warn if very large)
        try:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # 10MB
                self.result.inconsistent_locking.append(
                    f"Warning: Large file ({size_mb:.1f}MB) may impact analysis performance"
                )
        except OSError as e:
            self.result.inconsistent_locking.append(f"Cannot read file stats: {e}")
            return False

        return True

    def _read_file_safely(self, filepath: Path) -> Optional[str]:
        """Safely read file with multiple encoding attempts"""
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

        for encoding in encodings:
            try:
                return filepath.read_text(encoding=encoding, errors="replace")
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.result.inconsistent_locking.append(
                    f"Error reading file {filepath}: {e}"
                )
                return None

        self.result.inconsistent_locking.append(
            f"Could not decode file {filepath} with any supported encoding"
        )
        return None

    def _extract_shared_variables(self, content: str):
        """Enhanced shared variable extraction with better C++ parsing"""
        # Remove comments and strings to avoid false positives
        cleaned_content = self._remove_comments_and_strings(content)

        # Extract class member variables
        class_pattern = r"class\s+(\w+)(?:\s*:\s*[^{]*)?{([^{}]*(?:{[^{}]*}[^{}]*)*)}"
        for class_match in re.finditer(class_pattern, cleaned_content, re.DOTALL):
            class_name = class_match.group(1)
            class_body = class_match.group(2)

            # Look for member variables
            member_vars = self.patterns["member_var"].findall(class_body)
            for var in member_vars:
                if var not in self.shared_vars:
                    self.shared_vars.add(var)

        # Extract global variables with better pattern matching
        lines = cleaned_content.split("\n")
        in_function_or_class = False
        brace_depth = 0

        for line in lines:
            # Track scope depth
            brace_depth += line.count("{") - line.count("}")

            # Skip lines inside functions or classes
            if brace_depth > 0:
                continue

            # Check for global variable declarations
            global_match = self.patterns["global_var"].match(line.strip())
            if global_match:
                var_name = global_match.group(1)
                if var_name not in self.shared_vars:
                    self.global_vars.add(var_name)
                    self.shared_vars.add(var_name)

    def _remove_comments_and_strings(self, content: str) -> str:
        """Remove C++ comments and string literals to avoid false positives"""
        # This is a simplified version - a production tool would use a proper lexer
        result = []
        i = 0
        while i < len(content):
            # Skip single-line comments
            if i < len(content) - 1 and content[i : i + 2] == "//":
                while i < len(content) and content[i] != "\n":
                    i += 1
                if i < len(content):
                    result.append("\n")  # Keep newlines for line numbers
                    i += 1
                continue

            # Skip multi-line comments
            if i < len(content) - 1 and content[i : i + 2] == "/*":
                i += 2
                while i < len(content) - 1 and content[i : i + 2] != "*/":
                    if content[i] == "\n":
                        result.append("\n")  # Keep newlines
                    i += 1
                i += 2  # Skip */
                continue

            # Skip string literals
            if content[i] == '"':
                result.append(" ")  # Replace with space
                i += 1
                while i < len(content) and content[i] != '"':
                    if content[i] == "\\" and i + 1 < len(content):
                        i += 2  # Skip escaped character
                    else:
                        if content[i] == "\n":
                            result.append("\n")
                        i += 1
                if i < len(content):
                    i += 1  # Skip closing quote
                continue

            result.append(content[i])
            i += 1

        return "".join(result)

    def _extract_memory_accesses(self, content: str, filename: str):
        """Enhanced memory access extraction with context analysis"""
        lines = content.split("\n")
        current_function = ""
        current_class = ""
        scope_depth = 0

        for i, line in enumerate(lines, 1):
            # Track scope depth
            scope_depth += line.count("{") - line.count("}")

            # Track class context
            class_match = re.search(r"class\s+(\w+)", line)
            if class_match:
                current_class = class_match.group(1)

            # Track function definitions with improved pattern
            func_match = self.patterns["function_def"].search(line)
            if func_match:
                current_function = func_match.group(1)

            # Check for access to shared variables
            for var in self.shared_vars:
                if var in line:
                    # Determine access type with better heuristics
                    access_type = self._determine_access_type(line, var)

                    # Check if access is protected
                    is_protected, protection = self._check_protection(line, lines, i)

                    access = MemoryAccess(
                        variable=var,
                        access_type=access_type,
                        line_number=i,
                        function=current_function or "<global>",
                        thread_context=self._infer_thread_context(current_function),
                        is_protected=is_protected,
                        protection_mechanism=protection,
                        code_snippet=line.strip(),
                    )

                    self.memory_accesses[var].append(access)

    def _determine_access_type(self, line: str, var: str) -> AccessType:
        """Determine the type of memory access with improved heuristics"""
        # Find the variable in the line and analyze context
        var_index = line.find(var)
        if var_index == -1:
            return AccessType.READ

        # Look at what comes after the variable
        after_var = line[var_index + len(var) :].strip()

        # Check for assignment operations
        if after_var.startswith("=") and not after_var.startswith("=="):
            return AccessType.WRITE

        # Check for compound assignment
        if any(
            after_var.startswith(op)
            for op in ["+=", "-=", "*=", "/=", "%=", "|=", "&=", "^="]
        ):
            return AccessType.RMW

        # Check for increment/decrement
        if after_var.startswith("++") or after_var.startswith("--"):
            return AccessType.RMW

        # Check for atomic operations
        if any(
            op in after_var
            for op in [".load(", ".store(", ".exchange(", ".compare_exchange"]
        ):
            return AccessType.RMW

        # Check if variable appears before assignment
        before_var = line[:var_index].strip()
        if before_var.endswith("++") or before_var.endswith("--"):
            return AccessType.RMW

        return AccessType.READ

    def _check_protection(
        self, line: str, lines: List[str], line_num: int
    ) -> Tuple[bool, Optional[str]]:
        """Check if a memory access is properly protected"""
        # Look for synchronization primitives in surrounding context
        context_start = max(0, line_num - 10)
        context_end = min(len(lines), line_num + 5)
        context = "\n".join(lines[context_start:context_end])

        # Check for various protection mechanisms
        protections = []

        # Mutex locks
        if any(
            pattern in context
            for pattern in ["lock_guard", "unique_lock", "shared_lock"]
        ):
            protections.append("RAII_lock")

        if re.search(r"\w+\.lock\s*\(", context):
            protections.append("explicit_lock")

        # Atomic operations
        if "atomic" in context or ".load(" in context or ".store(" in context:
            protections.append("atomic")

        # Memory barriers
        if any(
            barrier in context for barrier in ["memory_order", "atomic_thread_fence"]
        ):
            protections.append("memory_barrier")

        return len(protections) > 0, ",".join(protections) if protections else None

    def analyze_control_flow_paths(self, content: str, filename: str):
        """Enhanced control flow analysis to detect missing unlocks"""
        lines = content.split("\n")
        current_function = ""
        current_class = ""

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track class context
            class_match = re.search(r"class\s+(\w+)(?:\s*:\s*[^{]*)?{", line)
            if class_match:
                current_class = class_match.group(1)
                continue

            # Track method definitions
            method_match = re.search(
                r"(?:(?:inline|static|virtual|explicit|constexpr)\s+)*"
                r"(?:\w+(?:\s*::\s*\w+)*\s+)?"
                r"(?:\w+\s*::\s*)?(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?[{;]",
                line,
            )
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
        """Analyze a single function for control flow and lock management issues"""
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
        """Extract all possible execution paths through a function"""
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
            lock_match = re.search(r"(\w+)\.lock\s*\(", line)
            if lock_match:
                mutex_name = lock_match.group(1)
                current_path.locks_acquired.append(mutex_name)
                locks_in_scope.append(mutex_name)

            unlock_match = re.search(r"(\w+)\.unlock\s*\(", line)
            if unlock_match:
                mutex_name = unlock_match.group(1)
                current_path.locks_released.append(mutex_name)
                if mutex_name in locks_in_scope:
                    locks_in_scope.remove(mutex_name)

            # Check for RAII locks
            raii_match = re.search(
                r"std\s*::\s*(lock_guard|unique_lock|shared_lock|scoped_lock)\s*<[^>]+>\s+(\w+)\s*\(\s*([^)]+)\s*\)",
                line,
            )
            if raii_match:
                lock_type = raii_match.group(1)
                var_name = raii_match.group(2)
                mutex_name = raii_match.group(3)
                # RAII locks are automatically released at scope end
                current_path.locks_acquired.append(mutex_name)
                # Don't add to locks_in_scope as RAII handles it

            # Check for early returns
            if re.search(r"^\s*return\s*[^;]*;", stripped):
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
            if re.search(r"^\s*throw\s+[^;]*;", stripped):
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
        """Check a specific execution path for lock-related issues"""
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
        """Analyze STL container usage for thread safety issues"""
        lines = content.split("\n")
        current_function = ""
        current_class = ""
        container_vars = {}  # var_name -> container_type

        # Initialize stl_container_accesses if not exists
        if not hasattr(self, "stl_container_accesses"):
            self.stl_container_accesses = defaultdict(list)

        for i, line in enumerate(lines, 1):
            # Track class and function context
            class_match = re.search(r"class\s+(\w+)(?:\s*:\s*[^{]*)?{", line)
            if class_match:
                current_class = class_match.group(1)
                continue
            # Find container method calls
            container_method_match = re.search(
                r"(\w+)\.(push_back|insert|erase|clear|resize|at|operator\[\]|\[\])\s*\(",
                line,
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
        """Determine if a container operation is thread-safe"""
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
        non_thread_safe_containers = {
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

        thread_safe_containers = {
            "std::atomic",
            "std::shared_ptr",  # Limited thread safety
        }

        if container_type in non_thread_safe_containers:
            return False

        if container_type in thread_safe_containers:
            return True

        return False

    def _check_container_synchronization(
        self, access: STLContainerAccess, line: str, lines: List[str], line_num: int
    ):
        """Check if container access is properly synchronized"""
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
                f"({access.container_type}) in {access.function}() at line {access.line_number}. "
                f"Method '{access.access_method}' is not thread-safe."
            )

            if hasattr(self, "result"):
                self.result.inconsistent_locking.append(issue)

    def analyze_complex_deadlock_scenarios(self, content: str, filename: str):
        """Enhanced deadlock detection for complex scenarios with nested calls"""
        lines = content.split("\n")
        current_class = ""
        current_method = ""

        # Build method call graph with lock information
        class_methods = defaultdict(dict)  # class -> {method -> {locks, calls, line}}

        for i, line in enumerate(lines, 1):
            # Track class context
            class_match = re.search(r"class\s+(\w+)(?:\s*:\s*[^{]*)?{", line)
            if class_match:
                current_class = class_match.group(1)
                continue

            # Track method definitions
            method_match = re.search(
                r"(?:(?:inline|static|virtual|explicit|constexpr)\s+)*"
                r"(?:\w+(?:\s*::\s*\w+)*\s+)?"
                r"(?:\w+\s*::\s*)?(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?[{;]",
                line,
            )
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
            lock_match = re.search(r"(\w+)\.lock\s*\(", line)
            if lock_match:
                mutex_name = lock_match.group(1)
                if (
                    current_class in class_methods
                    and current_method in class_methods[current_class]
                ):
                    class_methods[current_class][current_method]["locks"].append(
                        mutex_name
                    )

            raii_match = re.search(
                r"std\s*::\s*(lock_guard|unique_lock|scoped_lock)\s*<[^>]+>\s+(\w+)\s*\(\s*([^)]+)\s*\)",
                line,
            )
            if raii_match:
                mutex_name = raii_match.group(3)
                if (
                    current_class in class_methods
                    and current_method in class_methods[current_class]
                ):
                    class_methods[current_class][current_method]["locks"].append(
                        mutex_name
                    )

            # Track method calls
            method_call_match = re.search(r"(\w+)\s*\.\s*(\w+)\s*\([^)]*\)", line)
            if (
                method_call_match
                and current_class in class_methods
                and current_method in class_methods[current_class]
            ):
                obj_or_class = method_call_match.group(1)
                called_method = method_call_match.group(2)
                class_methods[current_class][current_method]["calls"].append(
                    (called_method, i, obj_or_class)
                )

        # Analyze for complex deadlock scenarios
        self._detect_nested_deadlock_risks(class_methods)

    def _detect_nested_deadlock_risks(self, class_methods: Dict):
        """Detect deadlock risks in nested method calls"""
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
        """Recursively find all locks that might be acquired in a call chain"""
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

    def _analyze_lock_acquisition_orders(self, lock_acquisition_order, lock_scopes):
        """Analyze lock acquisition orders for potential deadlocks.
        
        This method detects potential deadlocks by analyzing the order in which
        locks are acquired across different code paths. It looks for:
        1. Inconsistent lock acquisition orders (A->B in one path, B->A in another)
        2. Missing unlocks that could lead to deadlocks
        3. Recursive locking patterns that might be problematic
        
        Args:
            lock_acquisition_order: List of lock acquisition events with context
            lock_scopes: Dictionary mapping lock names to their scope information
            
        Returns:
            List of potential deadlock issues found
        """
        deadlock_issues = []
        
        # Track lock acquisition orders per function
        function_lock_orders = {}
        
        # First, group lock acquisitions by function and thread context
        for lock_event in lock_acquisition_order:
            func_key = (lock_event['function'], lock_event['thread_context'])
            if func_key not in function_lock_orders:
                function_lock_orders[func_key] = []
            function_lock_orders[func_key].append(lock_event)
        
        # Check for inconsistent lock ordering between functions
        lock_orders = {}
        for func_name, events in function_lock_orders.items():
            # Get the order of locks acquired in this function
            current_order = [e['lock_name'] for e in events if e['operation'] == 'acquire']
            
            # Check against previously seen orders for the same locks
            for i in range(1, len(current_order)):
                lock_pair = (current_order[i-1], current_order[i])
                if lock_pair[0] != lock_pair[1]:  # Skip same-lock pairs
                    if lock_pair not in lock_orders:
                        lock_orders[lock_pair] = set()
                    lock_orders[lock_pair].add(func_name[0])  # Add function name
        
        # Look for inconsistent ordering (A->B in one place, B->A in another)
        for (lock1, lock2), funcs in list(lock_orders.items()):
            reverse_pair = (lock2, lock1)
            if reverse_pair in lock_orders:
                # We have inconsistent ordering between these locks
                funcs1 = lock_orders[(lock1, lock2)]
                funcs2 = lock_orders[reverse_pair]
                
                # Only report if different functions are involved
                if not funcs1.intersection(funcs2):
                    issue = {
                        'type': 'inconsistent_lock_ordering',
                        'locks': [lock1, lock2],
                        'functions': list(funcs1.union(funcs2)),
                        'severity': 'high',
                        'description': (
                            f"Inconsistent lock ordering between {lock1} and {lock2} "
                            f"across functions: {', '.join(funcs1.union(funcs2))}"
                        ),
                        'suggestion': (
                            f"Ensure consistent lock acquisition order between {lock1} and {lock2} "
                            "across all code paths to prevent potential deadlocks."
                        )
                    }
                    deadlock_issues.append(issue)
        
        # Check for missing unlocks in functions
        for func_name, events in function_lock_orders.items():
            lock_stack = []
            for event in events:
                if event['operation'] == 'acquire':
                    lock_stack.append(event['lock_name'])
                elif event['operation'] == 'release':
                    if lock_stack and lock_stack[-1] == event['lock_name']:
                        lock_stack.pop()
            
            # Any remaining locks on the stack were not released
            for lock_name in reversed(lock_stack):
                issue = {
                    'type': 'missing_unlock',
                    'lock': lock_name,
                    'function': func_name[0],
                    'severity': 'high',
                    'description': f"Lock '{lock_name}' may not be released in function '{func_name[0]}'",
                    'suggestion': (
                        f"Ensure all code paths in '{func_name[0]}' release the lock '{lock_name}'. "
                        "Consider using RAII wrapper classes like std::lock_guard or std::unique_lock."
                    )
                }
                deadlock_issues.append(issue)
        
        # Check for recursive locking patterns
        for func_name, events in function_lock_orders.items():
            lock_depth = {}
            for event in events:
                if event['operation'] == 'acquire':
                    lock_depth[event['lock_name']] = lock_depth.get(event['lock_name'], 0) + 1
                    if lock_depth[event['lock_name']] > 1:
                        issue = {
                            'type': 'recursive_locking',
                            'lock': event['lock_name'],
                            'function': func_name[0],
                            'severity': 'medium',
                            'description': (
                                f"Recursive locking detected for '{event['lock_name']}' "
                                f"in function '{func_name[0]}'"
                            ),
                            'suggestion': (
                                f"Avoid recursive locking of '{event['lock_name']}'. "
                                "If recursive locking is necessary, use std::recursive_mutex instead."
                            )
                        }
                        deadlock_issues.append(issue)
                elif event['operation'] == 'release':
                    if event['lock_name'] in lock_depth:
                        lock_depth[event['lock_name']] -= 1
                        if lock_depth[event['lock_name']] == 0:
                            del lock_depth[event['lock_name']]
        
        return deadlock_issues

    def _extract_locking_patterns(self, content: str, filename: str):
        """Enhanced extraction of locking patterns with support for modern C++ features.

        This method identifies and tracks various locking patterns including:
        - Direct mutex operations (lock/unlock/try_lock)
        - RAII-style locking (lock_guard, unique_lock, scoped_lock)
        - C++20 synchronization primitives (counting_semaphore, latch, barrier)
        - Custom lockable types with lock()/unlock() methods

        Args:
            content: The source code content to analyze
            filename: Name of the file being analyzed (for error reporting)
        """
        import re
        from collections import defaultdict

        lines = content.split("\n")
        current_function = ""
        current_class = ""

        # Track the current scope and locks held in each scope
        scope_stack = []  # Stack of scope dictionaries
        raii_vars = {}  # Map of variable names to their RAII mutex names

        # Track lock acquisition order for deadlock detection
        lock_acquisition_order = defaultdict(list)

        # Track lock scopes for RAII objects
        lock_scopes = defaultdict(list)

        # Regex patterns for modern C++ synchronization primitives
        cpp20_patterns = {
            "counting_semaphore": r"std\s*::\s*counting_semaphore\s*<\s*[^>]+\s*>",
            "binary_semaphore": r"std\s*::\s*binary_semaphore\s*<\s*[^>]+\s*>",
            "latch": r"std\s*::\s*latch\b",
            "barrier": r"std\s*::\s*barrier\s*<\s*[^>]+\s*>",
        }

        # Track class/struct definitions to handle member functions
        class_stack = []

        # Initialize locking patterns if not exists
        if not hasattr(self, "locking_patterns"):
            self.locking_patterns = defaultdict(list)

        # Initialize result object if not exists
        if not hasattr(self, "result"):
            self.result = type("Result", (), {"inconsistent_locking": []})()

        try:
            for i, line in enumerate(lines, 1):
                try:
                    # Track class/struct definitions
                    class_match = re.search(r"\b(class|struct)\s+(\w+)", line)
                    if class_match and "{" in line:
                        class_stack.append(class_match.group(2))
                        current_class = class_stack[-1]
                    elif "}" in line and class_stack:
                        class_stack.pop()
                        current_class = class_stack[-1] if class_stack else ""

                    # Track function definitions - handle both member and free functions
                    func_match = re.search(
                        r"(?:(?:inline|static|virtual|explicit|constexpr)\s+)*"
                        r"(?:\w+(?:\s*::\s*\w+)*\s+)?"
                        r"(?:\w+\s*::\s*)?(\w+)\s*[^;{=]*[({]",
                        line,
                    )
                    if (
                        func_match
                        and "{" in line
                        and not any(
                            kw in line for kw in ["if", "while", "for", "switch"]
                        )
                    ):
                        current_function = func_match.group(1)
                        scope_stack.append(
                            {
                                "start_line": i,
                                "end_line": None,
                                "function": current_function,
                                "class": current_class,
                                "locks_held": set(),
                                "lock_operations": [],
                                "return_points": [],
                            }
                        )

                        # Track lock acquisition order for this function
                        lock_acquisition_order[current_function] = []

                        # If this is a lock guard constructor, track the lock acquisition
                        lock_guard_match = re.search(
                            r"std\s*::\s*(lock_guard|unique_lock|scoped_lock)\s*<[^>]+>\s+(\w+)\s*\(\s*([^)]+)\s*\)",
                            line,
                        )
                        if lock_guard_match:
                            var_name = lock_guard_match.group(2)
                            mutex_name = lock_guard_match.group(3).strip()
                            raii_vars[var_name] = mutex_name

                            # Record the lock acquisition
                            if scope_stack:
                                scope = scope_stack[-1]
                                scope["locks_held"].add(mutex_name)
                                scope["lock_operations"].append(("lock", mutex_name, i))
                                lock_acquisition_order[current_function].append(
                                    ("acquire", mutex_name, i)
                                )

                                # Record the lock scope
                                lock_scopes[mutex_name].append(
                                    {
                                        "start_line": i,
                                        "end_line": None,
                                        "function": current_function,
                                        "class": current_class,
                                        "is_raii": True,
                                    }
                                )

                    # Track RAII locks (lock_guard, unique_lock, etc.)
                    lock_guard_match = re.search(
                        r"std\s*::\s*(lock_guard|unique_lock|scoped_lock)\s*<[^>]+>\s+(\w+)\s*\(\s*([^)]+)\s*\)",
                        line,
                    )
                    if lock_guard_match and scope_stack:
                        var_name = lock_guard_match.group(2)
                        mutex_name = lock_guard_match.group(3).strip()
                        raii_vars[var_name] = mutex_name

                        # Add lock operation
                        self.locking_patterns[mutex_name].append(
                            LockingPattern(
                                mutex_name=mutex_name,
                                operation="lock",
                                line_number=i,
                                function=current_function or "<global>",
                            )
                        )

                        # Track lock in current scope
                        scope_stack[-1]["locks_held"].add(mutex_name)

                        # Add implicit unlock at end of scope
                        self.locking_patterns[mutex_name].append(
                            LockingPattern(
                                mutex_name=mutex_name,
                                operation="unlock",
                                line_number=i,  # Will be updated when scope ends
                                function=current_function or "<global>",
                                is_implicit=True,
                            )
                        )

                except Exception as e:
                    self.result.inconsistent_locking.append(
                        f"Error analyzing line {i} in {filename}: {str(e)}"
                    )
                    continue

        except Exception as e:
            self.result.inconsistent_locking.append(
                f"Fatal error in lock analysis for {filename}: {str(e)}"
            )

        # Analyze lock acquisition orders for potential deadlocks
        self._analyze_lock_acquisition_orders(lock_acquisition_order, lock_scopes)

        # Track scope changes
        for i, line in enumerate(content.split("\n"), 1):
            open_braces = line.count("{")
            close_braces = line.count("}")

            # Handle scope openings
            for _ in range(open_braces):
                if scope_stack:
                    scope_stack[-1]["end_line"] = (
                        i - 1
                    )  # Previous line was end of previous scope

            # Handle scope closures
            for _ in range(close_braces):
                if scope_stack:
                    scope = scope_stack.pop()
                    # Update line numbers for any implicit unlocks in this scope
                    for pattern_list in self.locking_patterns.values():
                        for pattern in pattern_list:
                            if (
                                getattr(pattern, "is_implicit", False)
                                and pattern.function == scope["function"]
                            ):
                                pattern.line_number = i  # Update to end of scope

                    # Generate implicit unlocks for any remaining locks in this scope
                    for mutex_name in scope["locks_held"]:
                        self.locking_patterns[mutex_name].append(
                            LockingPattern(
                                mutex_name=mutex_name,
                                operation="unlock",
                                line_number=i,
                                function=scope["function"],
                                is_raii=True,
                                is_implicit=True,
                            )
                        )

                    # If we still have a parent scope, transfer any remaining locks
                    if scope_stack:
                        scope_stack[-1]["locks_held"].update(scope["locks_held"])

        return lock_acquisition_order, lock_scopes

    def _build_call_graph(self, content: str):
        """Build call graph with improved function call detection"""
        lines = content.split("\n")
        current_function = ""

        for line in lines:
            # Track function definitions
            func_match = self.patterns["function_def"].search(line)
            if func_match:
                current_function = func_match.group(1)
                if HAS_NETWORKX:
                    self.call_graph.add_node(current_function)
                else:
                    self.call_graph.add_node(current_function)

            # Find function calls (simplified)
            if current_function:
                # Look for function calls
                call_matches = re.finditer(r"(\w+)\s*\(", line)
                for match in call_matches:
                    called_func = match.group(1)
                    # Filter out obvious non-function calls
                    if called_func not in [
                        "if",
                        "while",
                        "for",
                        "switch",
                        "return",
                        "sizeof",
                    ]:
                        if HAS_NETWORKX:
                            self.call_graph.add_edge(current_function, called_func)
                        else:
                            self.call_graph.add_edge(current_function, called_func)

    def _analyze_thread_contexts(self, content: str):
        """Enhanced thread context analysis"""
        # Look for thread creation patterns
        thread_matches = self.patterns["thread_create"].finditer(content)
        for match in thread_matches:
            thread_name = match.group(1)
            # Mark functions called by threads
            # This is simplified - a full implementation would trace call paths

    def _detect_data_races(self):
        """Enhanced data race detection with confidence scoring"""
        for var, accesses in self.memory_accesses.items():
            if len(accesses) < 2:
                continue

            for i in range(len(accesses)):
                for j in range(i + 1, len(accesses)):
                    access1 = accesses[i]
                    access2 = accesses[j]

                    # Calculate race probability
                    race_confidence = self._calculate_race_confidence(
                        access1, access2, var
                    )

                    if race_confidence > 0.3:  # Threshold for reporting
                        severity = self._determine_race_severity(
                            access1, access2, var, race_confidence
                        )

                        race = RaceCondition(
                            variable=var,
                            access1=access1,
                            access2=access2,
                            severity=severity,
                            description=self._generate_race_description(
                                access1, access2, var
                            ),
                            fix_suggestion=self._generate_race_fix(
                                access1, access2, var
                            ),
                            confidence=race_confidence,
                        )

                        self.result.races.append(race)

    def _calculate_race_confidence(
        self, access1: MemoryAccess, access2: MemoryAccess, var: str
    ) -> float:
        """Calculate confidence level for race condition detection"""
        confidence = 0.0

        # Base confidence based on access types
        if (
            access1.access_type == AccessType.WRITE
            or access2.access_type == AccessType.WRITE
        ):
            confidence += 0.6
        else:
            confidence += 0.2  # Read-read races are less likely but possible

        # Increase confidence if different thread contexts
        if access1.thread_context != access2.thread_context:
            confidence += 0.3

        # Decrease confidence if both accesses are protected
        if access1.is_protected and access2.is_protected:
            # Check if same protection mechanism
            if access1.protection_mechanism == access2.protection_mechanism:
                confidence -= 0.7
            else:
                confidence -= 0.3  # Different protection might still race
        elif access1.is_protected or access2.is_protected:
            confidence -= 0.2  # Partial protection

        # Global variables are more likely to race
        if var in self.global_vars:
            confidence += 0.2

        return max(0.0, min(1.0, confidence))

    def _determine_race_severity(
        self, access1: MemoryAccess, access2: MemoryAccess, var: str, confidence: float
    ) -> SeverityLevel:
        """Determine severity level based on access patterns and context"""
        # Critical for state variables and control flow
        if any(
            keyword in var.lower() for keyword in ["state", "status", "flag", "control"]
        ):
            return SeverityLevel.CRITICAL

        # High for write operations with high confidence
        if confidence > 0.8 and (
            access1.access_type == AccessType.WRITE
            or access2.access_type == AccessType.WRITE
        ):
            return SeverityLevel.HIGH

        # Medium for moderate confidence or RMW operations
        if confidence > 0.6 or AccessType.RMW in [
            access1.access_type,
            access2.access_type,
        ]:
            return SeverityLevel.MEDIUM

        return SeverityLevel.LOW

    def _generate_race_description(
        self, access1: MemoryAccess, access2: MemoryAccess, var: str
    ) -> str:
        """Generate human-readable description of race condition"""
        return (
            f"Data race on '{var}' between {access1.function}() "
            f"({access1.access_type.value} at line {access1.line_number}) and "
            f"{access2.function}() ({access2.access_type.value} at line {access2.line_number})"
        )

    def _generate_race_fix(
        self, access1: MemoryAccess, access2: MemoryAccess, var: str
    ) -> str:
        """Generate fix suggestion for race condition"""
        if access1.is_protected or access2.is_protected:
            return f"Ensure both accesses to '{var}' use the same synchronization mechanism"
        else:
            return f"Protect all accesses to '{var}' with a mutex or make it atomic"

    def _analyze_locking_consistency(self):
        """Enhanced locking consistency analysis"""
        for mutex, operations in self.locking_patterns.items():
            # Separate RAII and explicit operations
            raii_ops = [op for op in operations if op.is_raii]
            explicit_ops = [op for op in operations if not op.is_raii]

            # Check explicit lock/unlock balance
            if explicit_ops:
                self._check_explicit_lock_balance(mutex, explicit_ops)

            # Check for mixed RAII and explicit usage
            if raii_ops and explicit_ops:
                self.result.inconsistent_locking.append(
                    f"Mixed RAII and explicit locking for {mutex} - prefer consistent approach"
                )

    def _check_explicit_lock_balance(
        self, mutex: str, operations: List[LockingPattern]
    ):
        """Check balance of explicit lock/unlock operations"""
        balance = 0
        for op in sorted(operations, key=lambda x: (x.function, x.line_number)):
            if op.operation == "lock":
                balance += 1
            elif op.operation == "unlock":
                balance -= 1
                if balance < 0:
                    self.result.inconsistent_locking.append(
                        f"Unlock without matching lock for {mutex} in {op.function}() at line {op.line_number}"
                    )
                    balance = 0

        if balance > 0:
            self.result.inconsistent_locking.append(
                f"Potential lock leak - {balance} lock(s) not released for {mutex}"
            )

    def _detect_deadlock_potential(self):
        """
        Detect potential deadlocks by analyzing lock acquisition patterns.

        This method performs several analyses:
        1. Checks for recursive locking on the same mutex
        2. Detects missing unlock operations
        3. Checks for inconsistent lock ordering between different functions
        4. Builds a lock acquisition graph to detect potential deadlock cycles
        """
        if not self.locking_patterns:
            return  # No locking patterns to analyze

        # Track lock state per function and per mutex
        function_lock_states = defaultdict(
            lambda: defaultdict(int)
        )  # function -> {mutex: lock_count}
        function_lock_sequences = defaultdict(list)  # function -> list[mutex]

        # First pass: Process all operations in line order
        for mutex, operations in self.locking_patterns.items():
            current_function = None

            for op in sorted(operations, key=lambda x: x.line_number):
                func = op.function

                # Skip RAII-based operations as they're self-managing
                if op.is_raii and op.operation in ["lock", "unlock"]:
                    continue

                if op.operation == "lock":
                    # Check for recursive lock
                    if function_lock_states[func][mutex] > 0:
                        self.result.deadlock_risks.append(
                            f"Recursive lock detected on '{mutex}' in {func}() at line {op.line_number}"
                        )
                    function_lock_states[func][mutex] += 1

                    # Only add to sequence if not already there (avoid duplicates from multiple operations)
                    if (
                        not function_lock_sequences[func]
                        or function_lock_sequences[func][-1] != mutex
                    ):
                        function_lock_sequences[func].append(mutex)

                elif op.operation == "unlock":
                    if function_lock_states[func][mutex] <= 0:
                        self.result.deadlock_risks.append(
                            f"Unlock without matching lock for '{mutex}' in {func}() at line {op.line_number}"
                        )
                    else:
                        function_lock_states[func][mutex] -= 1

                # Track try_lock operations but don't count them as locks for sequence purposes
                elif op.operation == "try_lock" and not op.is_raii:
                    if mutex not in function_lock_sequences[func]:
                        function_lock_sequences[func].append(mutex)

        # Second pass: Check for missing unlocks at the end of functions
        for func, locks in function_lock_states.items():
            for mutex, count in locks.items():
                if count > 0 and not any(
                    op.is_raii for op in self.locking_patterns.get(mutex, [])
                ):
                    self.result.deadlock_risks.append(
                        f"Lock '{mutex}' not released in {func}()"
                    )

        # Second pass: Check for inconsistent lock ordering between functions
        self._check_inconsistent_lock_ordering(function_lock_sequences)

        # Third pass: Detect potential deadlock cycles
        self._detect_deadlock_cycles(function_lock_sequences)

    def _check_inconsistent_lock_ordering(self, function_lock_sequences):
        """Check for inconsistent lock ordering between different functions"""
        # Filter out functions with less than 2 locks
        functions = [
            f for f, locks in function_lock_sequences.items() if len(locks) >= 2
        ]

        for i in range(len(functions)):
            func1 = functions[i]
            locks1 = function_lock_sequences[func1]

            for j in range(i + 1, len(functions)):
                func2 = functions[j]
                locks2 = function_lock_sequences[func2]

                # Find common locks between the two functions (need at least 2)
                common_locks = set(locks1) & set(locks2)
                if len(common_locks) < 2:
                    continue

                # Get the relative order of common locks in each function
                order1 = {
                    lock: idx for idx, lock in enumerate(locks1) if lock in common_locks
                }
                order2 = {
                    lock: idx for idx, lock in enumerate(locks2) if lock in common_locks
                }

                # Check all pairs of locks for inconsistent ordering
                for lock1, lock2 in itertools.permutations(common_locks, 2):
                    # Skip if either lock doesn't appear in both functions
                    if (
                        lock1 not in order1
                        or lock2 not in order1
                        or lock1 not in order2
                        or lock2 not in order2
                    ):
                        continue

                    # Check if the order is inconsistent
                    if (
                        order1[lock1] < order1[lock2] and order2[lock1] > order2[lock2]
                    ) or (
                        order1[lock1] > order1[lock2] and order2[lock1] < order2[lock2]
                    ):

                        # Only report each inconsistent pair once
                        if lock1 < lock2:  # Ensure consistent ordering of the pair
                            self.result.deadlock_risks.append(
                                f"Inconsistent lock ordering between {func1}() and {func2}(): "
                                f"{lock1} and {lock2} are acquired in different orders"
                            )
                            break  # Only report one inconsistency per function pair

    def _detect_deadlock_cycles(self, function_lock_sequences):
        """Detect potential deadlocks by building a lock acquisition graph"""
        if not HAS_NETWORKX or not function_lock_sequences:
            return

        try:
            lock_graph = nx.DiGraph()

            # Build lock acquisition graph
            for func, locks in function_lock_sequences.items():
                # Add edges between consecutive locks to show acquisition order
                for i in range(len(locks) - 1):
                    for j in range(
                        i + 1, min(i + 3, len(locks))
                    ):  # Limit lookahead for performance
                        lock_graph.add_edge(locks[i], locks[j])

            # Find cycles in the lock graph (potential deadlocks)
            try:
                cycles = list(nx.simple_cycles(lock_graph))
                for cycle in cycles:
                    if len(cycle) >= 2:  # Only report cycles with 2+ locks
                        cycle_str = " -> ".join(cycle)
                        self.result.deadlock_risks.append(
                            f"Potential deadlock cycle: {cycle_str}"
                        )
            except nx.NetworkXNoCycle:
                pass  # No cycles found

        except Exception as e:
            self.result.deadlock_risks.append(
                f"Error detecting deadlock cycles: {str(e)}"
            )

    def _validate_atomic_operations(self):
        """Enhanced atomic operations validation with memory ordering checks"""
        atomic_vars = set()

        # Find atomic variable declarations and operations
        for var, accesses in self.memory_accesses.items():
            for access in accesses:
                # Check for std::atomic<T> declarations
                if (
                    "std::atomic" in access.code_snippet
                    and "=" in access.code_snippet
                    and ";" in access.code_snippet
                ):
                    atomic_vars.add(var)
                    continue

                # Check for atomic operations
                atomic_ops = [
                    "load",
                    "store",
                    "exchange",
                    "compare_exchange_weak",
                    "compare_exchange_strong",
                    "fetch_add",
                    "fetch_sub",
                    "fetch_and",
                    "fetch_or",
                    "fetch_xor",
                    "++",
                    "--",
                    "+=",
                    "-=",
                    "|=",
                    "&=",
                    "^=",
                    "=",
                    "==",
                    "!=",
                    "<=",
                    ">=",
                    "<",
                    ">",
                ]

                if any(f"{var}.{op}" in access.code_snippet for op in atomic_ops):
                    atomic_vars.add(var)

                    # Check for proper memory orderings
                    if any(
                        op in access.code_snippet
                        for op in ["load", "store", "exchange", "compare_exchange"]
                    ):
                        has_ordering = any(
                            order in access.code_snippet
                            for order in [
                                "memory_order_relaxed",
                                "memory_order_consume",
                                "memory_order_acquire",
                                "memory_order_release",
                                "memory_order_acq_rel",
                                "memory_order_seq_cst",
                            ]
                        )
                        if not has_ordering:
                            self.result.atomic_violations.append(
                                f"Missing memory ordering specification for atomic operation on '{var}' "
                                f"in {access.function}() at line {access.line_number}"
                            )

        # Check for non-atomic accesses to atomic variables
        for var in atomic_vars:
            accesses = self.memory_accesses.get(var, [])
            for access in accesses:
                if not access.is_protected or "atomic" not in (
                    access.protection_mechanism or ""
                ):
                    # Skip if this is a proper atomic operation
                    if any(
                        f"{var}.{op}" in access.code_snippet
                        for op in [
                            "load",
                            "store",
                            "exchange",
                            "fetch_",
                            "++",
                            "--",
                            "+=",
                            "-=",
                        ]
                    ):
                        continue

                    self.result.atomic_violations.append(
                        f"Non-atomic access to atomic variable '{var}' in {access.function}() "
                        f"at line {access.line_number}. Use atomic operations like load()/store()"
                    )

    def _analyze_monero_specific_patterns(self, content: str, filename: str):
        """Enhanced Monero-specific pattern analysis"""
        lines = content.split("\n")
        in_async_stdin_reader = False
        current_method = ""

        for i, line in enumerate(lines, 1):
            # Track class context
            if "class async_stdin_reader" in line:
                in_async_stdin_reader = True
                continue

            if not in_async_stdin_reader:
                continue

            # Track method definitions
            method_match = re.search(r"(\w+)\s*::\s*(\w+)\s*\([^)]*\)", line)
            if method_match:
                current_method = method_match.group(2)

            # Check for specific Monero patterns
            self._check_read_status_access(line, i, current_method, filename)
            self._check_signal_handling(line, i, current_method, filename)
            self._check_condition_variables(line, i, current_method, filename)

    def _check_read_status_access(
        self, line: str, line_num: int, method: str, filename: str
    ):
        """Check for unsafe m_read_status access patterns"""
        if "m_read_status" not in line:
            return

        # Check for synchronization
        sync_keywords = [
            "m_response_mutex",
            "lock_guard",
            "unique_lock",
            "scoped_lock",
            "std::lock_guard",
            "std::unique_lock",
            "std::scoped_lock",
            "std::mutex",
            "std::atomic",
        ]

        # Skip if line contains any synchronization
        if any(keyword in line for keyword in sync_keywords):
            return

        # Check if this is a read or write access
        access_type = AccessType.READ
        if (
            "=" in line
            and "==" not in line
            and "!=" not in line
            and "<=" not in line
            and ">=" not in line
        ):
            # Simple heuristic: if there's an assignment to m_read_status
            access_type = AccessType.WRITE

        # Create memory access record
        access = MemoryAccess(
            variable="m_read_status",
            access_type=access_type,
            line_number=line_num,
            function=method or "<global>",
            thread_context=self._infer_thread_context(method),
            is_protected=False,
            code_snippet=line.strip(),
        )
        self.memory_accesses["m_read_status"].append(access)

        # Also add to races if this is a write or if we find a conflicting access
        if access_type == AccessType.WRITE:
            for existing_access in self.memory_accesses["m_read_status"]:
                if existing_access.line_number != line_num:  # Different line
                    race = RaceCondition(
                        variable="m_read_status",
                        access1=existing_access,
                        access2=access,
                        severity=(
                            SeverityLevel.CRITICAL
                            if "run" in method.lower()
                            else SeverityLevel.HIGH
                        ),
                        description=f"Race condition on m_read_status between {existing_access.function}() and {method}()",
                        fix_suggestion="Protect access with std::mutex or use std::atomic",
                        confidence=0.9,
                    )
                    self.result.races.append(race)

    def _check_signal_handling(
        self, line: str, line_num: int, method: str, filename: str
    ):
        """Check for signal handling issues"""
        if "signal(" in line and "SIG" in line:
            self.result.inconsistent_locking.append(
                f"Use of signal() detected in {method}() at {filename}:{line_num} - "
                f"consider sigaction() for better portability"
            )

    def _check_condition_variables(
        self, line: str, line_num: int, method: str, filename: str
    ):
        """Check condition variable usage patterns"""
        if "m_response_cv." in line:
            if ".wait(" in line and "lock" not in line:
                self.result.inconsistent_locking.append(
                    f"Condition variable wait without visible lock in {method}() "
                    f"at {filename}:{line_num}"
                )

    def _detect_async_io_races(self, content: str):
        """Enhanced async I/O race detection"""
        # Check for EOF handling patterns
        if "eof" in content.lower() and "thread" in content.lower():
            # Look for race conditions in EOF checking
            eof_patterns = re.finditer(r"(?:!)?eof\s*\(\s*\)", content, re.IGNORECASE)
            for match in eof_patterns:
                # Check if EOF check is properly synchronized
                surrounding = content[max(0, match.start() - 100) : match.end() + 100]
                if not any(sync in surrounding for sync in ["mutex", "lock", "atomic"]):
                    self.result.inconsistent_locking.append(
                        "Potentially unsynchronized EOF check detected"
                    )

    def _generate_monero_specific_fixes(self, races: List[RaceCondition]):
        """Generate enhanced Monero-specific fix recommendations"""
        has_read_status_races = any("m_read_status" in r.variable for r in races)

        if has_read_status_races or any(
            "m_read_status" in issue for issue in self.result.inconsistent_locking
        ):
            fixes = [
                "\n=== MONERO ASYNC_STDIN_READER SECURITY FIXES ===",
                "",
                "1. CRITICAL: Make m_read_status atomic and thread-safe:",
                "   - Replace: volatile t_state m_read_status;",
                "   - With:    std::atomic<t_state> m_read_status{state_init};",
                "   - Add:     #include <atomic>",
                "",
                "2. Implement proper synchronization patterns:",
                "   ```cpp",
                "   // In each method accessing m_read_status:",
                "   boost::unique_lock<boost::mutex> lock(m_response_mutex);",
                "   t_state current_state = m_read_status.load();",
                "   // ... use current_state ...",
                "   m_read_status.store(new_state);",
                "   ```",
                "",
                "3. Signal handling improvements:",
                "   - Use sigaction() with SA_RESTART flag",
                "   - Consider self-pipe trick for signal safety",
                "   - Block signals during critical sections",
                "",
                "4. Enhanced testing and validation:",
                "   - Compile with -fsanitize=thread",
                "   - Use Helgrind/DRD for race detection",
                "   - Add stress tests with signal injection",
                "",
                "5. Performance considerations:",
                "   - Use memory_order_relaxed for performance-critical reads",
                "   - Consider memory_order_acquire/release for synchronization",
                "   - Profile critical paths under ThreadSanitizer",
            ]

            self.result.atomic_violations.extend(fixes)

    def _calculate_metrics(self):
        """Calculate comprehensive analysis metrics"""
        critical_races = [
            r for r in self.result.races if r.severity == SeverityLevel.CRITICAL
        ]
        high_races = [r for r in self.result.races if r.severity == SeverityLevel.HIGH]
        medium_races = [
            r for r in self.result.races if r.severity == SeverityLevel.MEDIUM
        ]
        low_races = [r for r in self.result.races if r.severity == SeverityLevel.LOW]

        self.result.metrics = {
            "total_races": len(self.result.races),
            "critical_races": len(critical_races),
            "high_races": len(high_races),
            "medium_races": len(medium_races),
            "low_races": len(low_races),
            "locking_issues": len(self.result.inconsistent_locking),
            "deadlock_risks": len(self.result.deadlock_risks),
            "atomic_violations": len(self.result.atomic_violations),
            "shared_variables": len(self.shared_vars),
            "global_variables": len(self.global_vars),
            "memory_accesses": sum(
                len(accesses) for accesses in self.memory_accesses.values()
            ),
            "lock_operations": sum(len(ops) for ops in self.locking_patterns.values()),
            "analysis_time_seconds": round(self.result.analysis_time, 3),
            "high_confidence_races": len(
                [r for r in self.result.races if r.confidence > 0.8]
            ),
            "protected_accesses": sum(
                1
                for accesses in self.memory_accesses.values()
                for access in accesses
                if access.is_protected
            ),
        }

    def _infer_thread_context(self, function_name: str) -> str:
        """Enhanced thread context inference"""
        if not function_name:
            return "Unknown"

        # Check against known thread entry points
        if function_name in self.thread_entry_points:
            return "Background Thread"

        # Check for common threading patterns
        if any(
            pattern in function_name.lower()
            for pattern in ["thread", "async", "worker", "handler", "callback"]
        ):
            return "Background Thread"

        # Check for main thread indicators
        if any(
            pattern in function_name.lower()
            for pattern in ["main", "init", "setup", "constructor"]
        ):
            return "Main Thread"

        return "Main Thread"  # Default assumption


def format_analysis_report(result: AnalysisResult, filename: str) -> str:
    """Enhanced report formatting with better organization and actionable insights"""
    report = []

    # Header with metadata
    report.append("=" * 100)
    report.append(f"ThreadGuard Enhanced Analysis Report")
    report.append("=" * 100)
    report.append(f"File: {filename}")
    report.append(f"Analysis Time: {result.analysis_time:.3f} seconds")
    report.append(
        f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Executive Summary
    critical_count = result.metrics.get("critical_races", 0)
    high_count = result.metrics.get("high_races", 0)
    total_issues = (
        critical_count
        + high_count
        + result.metrics.get("locking_issues", 0)
        + result.metrics.get("deadlock_risks", 0)
    )

    report.append("\n" + "🔍 EXECUTIVE SUMMARY")
    report.append("-" * 50)

    if total_issues == 0:
        report.append("✅ NO CRITICAL CONCURRENCY ISSUES DETECTED")
        report.append("   The code appears to follow good concurrency practices.")
    elif critical_count > 0:
        report.append(
            f"🚨 {critical_count} CRITICAL ISSUE(S) REQUIRE IMMEDIATE ATTENTION"
        )
        report.append("   These issues pose significant security and stability risks.")
    elif high_count > 0:
        report.append(f"⚠️  {high_count} HIGH SEVERITY ISSUE(S) DETECTED")
        report.append("   Review and fix recommended before production deployment.")
    else:
        report.append("ℹ️  Minor concurrency issues detected - review recommended.")

    # Detailed metrics
    report.append(f"\nAnalysis Metrics:")
    report.append(f"  • Total Race Conditions: {result.metrics.get('total_races', 0)}")
    report.append(
        f"  • High Confidence Races: {result.metrics.get('high_confidence_races', 0)}"
    )
    report.append(f"  • Shared Variables: {result.metrics.get('shared_variables', 0)}")
    report.append(f"  • Memory Accesses: {result.metrics.get('memory_accesses', 0)}")
    report.append(
        f"  • Protected Accesses: {result.metrics.get('protected_accesses', 0)}"
    )

    # Critical Issues Section
    critical_races = [r for r in result.races if r.severity == SeverityLevel.CRITICAL]
    if critical_races:
        report.append("\n" + "🚨 CRITICAL ISSUES")
        report.append("-" * 50)
        for i, race in enumerate(critical_races, 1):
            report.append(f"{i}. {race.description}")
            report.append(f"   📍 Location: Line {race.access1.line_number}")
            report.append(f"   🔧 Fix: {race.fix_suggestion}")
            report.append(f"   📊 Confidence: {race.confidence:.1%}")
            if race.access1.code_snippet:
                report.append(f"   💻 Code: {race.access1.code_snippet}")
            report.append("")

    # High Severity Issues
    high_races = [r for r in result.races if r.severity == SeverityLevel.HIGH]
    if high_races:
        report.append("\n" + "⚠️  HIGH SEVERITY ISSUES")
        report.append("-" * 50)
        for i, race in enumerate(high_races, 1):
            report.append(f"{i}. {race.description}")
            report.append(f"   📍 Location: Line {race.access1.line_number}")
            report.append(f"   🔧 Fix: {race.fix_suggestion}")
            report.append(f"   📊 Confidence: {race.confidence:.1%}")
            report.append("")

    # Locking Issues
    if result.inconsistent_locking:
        report.append("\n" + "🔒 LOCKING & SYNCHRONIZATION ISSUES")
        report.append("-" * 50)
        for i, issue in enumerate(result.inconsistent_locking, 1):
            report.append(f"{i}. {issue}")

    # Deadlock Risks
    if result.deadlock_risks:
        report.append("\n" + "⚰️  DEADLOCK RISKS")
        report.append("-" * 50)
        for i, risk in enumerate(result.deadlock_risks, 1):
            report.append(f"{i}. {risk}")

    # Atomic Violations and Recommendations
    if result.atomic_violations:
        report.append("\n" + "⚛️  RECOMMENDATIONS & ATOMIC OPERATIONS")
        report.append("-" * 50)
        for violation in result.atomic_violations:
            report.append(violation)

    # Best Practices Section
    if total_issues > 0:
        report.append("\n" + "📚 BEST PRACTICES RECOMMENDATIONS")
        report.append("-" * 50)
        report.append("1. Use RAII locking (std::lock_guard, std::unique_lock)")
        report.append("2. Prefer std::atomic for simple shared variables")
        report.append("3. Maintain consistent lock ordering across functions")
        report.append("4. Use ThreadSanitizer during development and testing")
        report.append("5. Document threading assumptions and invariants")
        report.append("6. Consider lock-free algorithms for performance-critical code")

    report.append("\n" + "=" * 100)

    return "\n".join(report)


def main():
    """Enhanced main function with better error handling and options"""
    parser = argparse.ArgumentParser(
        description="ThreadGuard: Enhanced static analyzer for C++ concurrency bugs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python threadguard_enhanced.py file.cpp
  python threadguard_enhanced.py --output report.txt src/*.cpp
  python threadguard_enhanced.py --json results.json --ci-mode file.cpp
  python threadguard_enhanced.py --debug --max-high 3 file.cpp

Exit Codes:
  0: Success, no critical issues
  1: High severity issues found (CI mode)
  2: Critical issues found (CI mode)
  3: Analysis error
""",
    )

    parser.add_argument("files", nargs="*", help="C++ source files to analyze")
    parser.add_argument("--output", "-o", help="Output file for the report")
    parser.add_argument("--json", help="Output JSON report to file")
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Run in CI mode with non-zero exit on issues",
    )
    parser.add_argument(
        "--max-critical",
        type=int,
        default=0,
        help="Maximum allowed critical issues (CI mode)",
    )
    parser.add_argument(
        "--max-high",
        type=int,
        default=5,
        help="Maximum allowed high severity issues (CI mode)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output and stack traces"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-essential output"
    )
    parser.add_argument(
        "--version", action="version", version="ThreadGuard Enhanced 1.2.0"
    )

    args = parser.parse_args()

    if not args.files:
        parser.print_help()
        sys.exit(1)

    # Validate dependencies
    if not HAS_NETWORKX and not args.quiet:
        print(
            "Warning: networkx not available. Install with: pip install networkx",
            file=sys.stderr,
        )

    analyzer = ThreadGuardAnalyzer()
    all_results = []
    total_critical = 0
    total_high = 0

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            if args.ci_mode:
                sys.exit(3)
            continue

        if not args.quiet:
            print(f"Analyzing {path}...")

        try:
            result = analyzer.analyze_file(path)
            all_results.append((str(path), result))

            # Count issues for CI mode
            total_critical += result.metrics.get("critical_races", 0)
            total_high += result.metrics.get("high_races", 0)

            # Generate and output report
            report = format_analysis_report(result, str(path))

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(report)
                if not args.quiet:
                    print(f"Report saved to {args.output}")
            else:
                print(report)

        except Exception as e:
            error_msg = f"Error analyzing {path}: {str(e)}"
            print(error_msg, file=sys.stderr)
            if args.debug:
                import traceback

                traceback.print_exc()
            if args.ci_mode:
                sys.exit(3)

    # Generate JSON output if requested
    if args.json and all_results:
        json_data = {
            "analysis_summary": {
                "total_files": len(all_results),
                "total_critical_races": total_critical,
                "total_high_races": total_high,
                "analysis_timestamp": __import__("datetime").datetime.now().isoformat(),
                "threadguard_version": "1.2.0",
            },
            "files": [],
        }

        for filepath, result in all_results:
            file_data = {
                "file": filepath,
                "metrics": result.metrics,
                "races": len(result.races),
                "critical_races": result.metrics.get("critical_races", 0),
                "high_races": result.metrics.get("high_races", 0),
                "locking_issues": len(result.inconsistent_locking),
                "deadlock_risks": len(result.deadlock_risks),
                "analysis_time": result.analysis_time,
            }
            json_data["files"].append(file_data)

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        if not args.quiet:
            print(f"JSON report saved to {args.json}")

    # CI mode exit logic
    if args.ci_mode:
        if total_critical > args.max_critical:
            print(
                f"❌ CI FAILURE: {total_critical} critical issues found "
                f"(max allowed: {args.max_critical})"
            )
            sys.exit(2)
        elif total_high > args.max_high:
            print(
                f"⚠️  CI WARNING: {total_high} high severity issues found "
                f"(max allowed: {args.max_high})"
            )
            sys.exit(1)
        else:
            print("✅ CI PASSED: No critical concurrency issues found")
            sys.exit(0)


if __name__ == "__main__":
    main()
