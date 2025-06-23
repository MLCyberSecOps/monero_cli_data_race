#!/usr/bin/env python3
"""
ThreadGuard: Static Analysis Tool for Concurrency Bug Detection


Detects data races, inconsistent locking patterns, and thread safety violations
specifically targeting async I/O handlers like Monero's async_stdin_reader.

Authors: Pradeep Kumar
License: MIT
Version: 1.1.0
Date: 2025-06-22
"""

import re
import sys
import json
import argparse
import ast
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, NamedTuple, DefaultDict
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
import networkx as nx
from collections import defaultdict

class SeverityLevel(Enum):
    """Bug severity classification for mission-critical software"""
    CRITICAL = "CRITICAL"      # Mission failure risk
    HIGH = "HIGH"             # Data corruption risk  
    MEDIUM = "MEDIUM"         # Performance degradation
    LOW = "LOW"               # Code quality issue
    INFO = "INFO"             # Informational finding

class AccessType(Enum):
    """Memory access pattern classification"""
    READ = "READ"
    WRITE = "WRITE"
    RMW = "READ_MODIFY_WRITE"  # Atomic read-modify-write
    CALL = "CALL"              # Function call that may access

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

@dataclass
class LockingPattern:
    """Represents a locking/unlocking pattern"""
    mutex_name: str
    operation: str  # "lock", "unlock", "try_lock"
    line_number: int
    function: str
    success_guaranteed: bool = True

@dataclass
class RaceCondition:
    """Detected race condition between two memory accesses"""
    variable: str
    access1: MemoryAccess
    access2: MemoryAccess
    severity: SeverityLevel
    description: str
    fix_suggestion: str

@dataclass
class AnalysisResult:
    """Complete analysis results"""
    races: List[RaceCondition] = field(default_factory=list)
    inconsistent_locking: List[str] = field(default_factory=list)
    deadlock_risks: List[str] = field(default_factory=list)
    atomic_violations: List[str] = field(default_factory=list)
    metrics: Dict[str, int] = field(default_factory=dict)

class ThreadGuardAnalyzer:
    """
    Advanced static analyzer for detecting concurrency issues in C++ code,
    with special focus on Monero's async_stdin_reader patterns.
    """
    
    def __init__(self):
        # Shared state tracking
        self.shared_vars: Set[str] = set()
        self.global_vars: Set[str] = set()
        self.memory_accesses: Dict[str, List[MemoryAccess]] = defaultdict(list)
        self.locking_patterns: Dict[str, List[LockingPattern]] = defaultdict(list)
        self.call_graph = nx.DiGraph()
        
        # Monero-specific patterns
        self.monero_race_patterns = [
            # State transitions and checks
            r'm_read_status\s*(?:!|=|==|!=|>|<|>=|<=)\s*state_\w+',
            
            # Unsynchronized access patterns
            r'm_read_status(?:\s*[^=]|$)',  # Any access not followed by =
            r'!eos\s*\(\s*\)',
            
            # Signal handling patterns
            r'signal\s*\([^,]+,\s*SIG[A-Z]+\)',
            
            # Thread creation with potential race conditions
            r'boost\s*::\s*thread\s*\(\s*&\s*async_stdin_reader\s*::',
            
            # Condition variable patterns
            r'm_response_cv\.(wait|wait_for|wait_until|notify_(one|all))\s*\('
        ]
        
        # Thread entry points
        self.thread_entry_points = {
            'reader_thread_func',
            'run',
            'start'
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
        self.result = AnalysisResult()
        
        # Check if file exists first
        if not filepath.exists():
            error_msg = f"File does not exist: {filepath}"
            self.result.inconsistent_locking.append(error_msg)
            return self.result
            
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            
            # Run analysis phases
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
            self.result.inconsistent_locking.append(
                f"ERROR during analysis: {str(e)}"
            )
        
        return self.result
    
    def _analyze_monero_specific_patterns(self, content: str, filename: str):
        """Detect Monero-specific concurrency vulnerabilities"""
        lines = content.split('\n')
        in_async_stdin_reader = False
        current_method = ""
        
        for i, line in enumerate(lines, 1):
            # Track class and method context
            if 'class async_stdin_reader' in line:
                in_async_stdin_reader = True
                continue
                
            if in_async_stdin_reader:
                # Track method definitions
                method_match = re.search(r'(\w+)\s*::\s*(\w+)\s*\s*\w*\s*[({]', line)
                if method_match:
                    current_method = method_match.group(2)
                
                # Check for unsafe m_read_status access
                if 'm_read_status' in line:
                    # Check for missing synchronization
                    if not any(lock in line for lock in ['m_response_mutex', 'std::lock_guard', 'unique_lock', 'scoped_lock']):
                        self.result.inconsistent_locking.append(
                            f"UNSAFE: Unsynchronized access to m_read_status in {current_method}() at {filename}:{i}"
                        )
                        
                        # Check for common race patterns
                        if any(op in line for op in ['if (', 'while (', 'switch (', 'case ']):
                            self._add_race_condition(
                                variable="m_read_status",
                                line_num=i,
                                function=current_method,
                                filename=filename,
                                description=f"Race condition on m_read_status in {current_method}() - check without proper synchronization",
                                fix_suggestion=f"Protect access to m_read_status with a mutex lock in {current_method}()"
                            )
                            
    def _extract_shared_variables(self, content: str):
        """Extract shared variables from the code"""
        # Look for class member variables that might be shared
        class_pattern = r'class\s+\w+\s*{[^}]*}'
        var_pattern = r'm_\w+\s*;'
        
        # Extract member variables (typically prefixed with m_)
        for class_match in re.finditer(class_pattern, content, re.DOTALL):
            class_body = class_match.group(0)
            for var_match in re.finditer(var_pattern, class_body):
                var_name = var_match.group(0).split(';')[0].strip()
                if var_name.startswith('m_'):
                    self.shared_vars.add(var_name)

        # Extract top-level global variables (e.g., int counter = 0;)
        # Extract global variables (ignore lines with parentheses or brace-init)
        # Extract global variables of fundamental types before any function or class definitions
        header = content
        first_def = re.search(r'^\s*(?:class\s+\w+|[\w:<>]+\s+\w+\s*\()', content, re.MULTILINE)
        if first_def:
            header = content[: first_def.start()]

        global_pattern = (
            r'^\s*(?:static\s+)?'            # optional static
            r'(?:int|long|short|char|bool|float|double)\s+'  # fundamental type
            r'([A-Za-z_]\w*)'                  # variable name
            r'\s*(?:=[^;]*)?;'                 # optional initializer
        )
        for match in re.finditer(global_pattern, header, re.MULTILINE):
            var_name = match.group(1)
            if var_name not in self.shared_vars:
                self.global_vars.add(var_name)
                self.shared_vars.add(var_name)
    
    def _extract_memory_accesses(self, content: str, filename: str):
        """Extract memory accesses from the code"""
        lines = content.split('\n')
        current_function = ""
        
        for i, line in enumerate(lines, 1):
            # Track function definitions
            func_match = re.search(r'\w+\s+\w+\s*::\s*(\w+)\s*\s*\w*\s*[({]', line)
            if func_match:
                current_function = func_match.group(1)
            
            # Check for access to shared variables
            for var in self.shared_vars:
                if var in line:
                    access_type = AccessType.WRITE if '=' in line.split(var)[0] else AccessType.READ
                    self.memory_accesses[var].append(MemoryAccess(
                        variable=var,
                        access_type=access_type,
                        line_number=i,
                        function=current_function,
                        thread_context=self._infer_thread_context(current_function)
                    ))
    
    def _extract_locking_patterns(self, content: str, filename: str):
        """Extract locking patterns from the code"""
        lines = content.split('\n')
        current_function = ""
        
        # Track the current scope and locks held in each scope
        scope_stack = []  # Stack of (start_line, end_line, function_name, locks_held)
        raii_vars = {}    # Map of variable names to their RAII mutex names
        
        for i, line in enumerate(lines, 1):
            # Track function definitions - handle both member and free functions
            func_match = re.search(r'(?:\w+\s+)*(?:\w+\s*::\s*)?(\w+)\s*[^;{=]*[({]', line)
            if func_match and '{' in line and not any(kw in line for kw in ['if', 'while', 'for', 'switch']):
                current_function = func_match.group(1)
                scope_stack.append((i, None, current_function, set()))
            
            # Check for scope end
            if '}' in line and scope_stack:
                start_line, _, func_name, _ = scope_stack[-1]
                scope_stack[-1] = (start_line, i, func_name, scope_stack[-1][3])
            
            # Check for direct mutex operations (mutex.lock(), mutex.unlock())
            lock_match = re.search(r'(\w+)\.(lock|unlock|try_lock)\s*\(', line)
            if lock_match:
                mutex_name = lock_match.group(1)
                operation = lock_match.group(2)
                self.locking_patterns[mutex_name].append(LockingPattern(
                    mutex_name=mutex_name,
                    operation=operation,
                    line_number=i,
                    function=current_function if current_function else "<global>"
                ))
            
            # Check for std::lock_guard declaration (RAII-style locking)
            lock_guard_match = re.search(r'std\s*::\s*lock_guard\s*<[^>]+>\s+(\w+)\s*\(\s*([^)]+)\s*\)', line)
            if lock_guard_match:
                var_name = lock_guard_match.group(1)
                mutex_name = lock_guard_match.group(2).strip()
                raii_vars[var_name] = mutex_name
                
                # Add lock operation (lock_guard locks on construction)
                self.locking_patterns[mutex_name].append(LockingPattern(
                    mutex_name=mutex_name,
                    operation='lock',
                    line_number=i,
                    function=current_function if current_function else "<global>"
                ))
                
                # Track the lock in the current scope
                if scope_stack:
                    scope_stack[-1][3].add(mutex_name)
            
            # Check for RAII variable going out of scope
            for var_name, mutex_name in list(raii_vars.items()):
                if f'{var_name}.' in line or f'{var_name}->' in line:
                    # Variable is still in use
                    continue
                if var_name in line and '=' not in line.split('//')[0]:  # Not an assignment
                    # Variable might be going out of scope
                    raii_vars.pop(var_name, None)
    
    def _build_call_graph(self, content: str):
        """Build call graph from the code"""
        # Simple implementation - would need more sophisticated parsing
        # for a production tool
        pass
    
    def _analyze_thread_contexts(self, content: str):
        """Analyze thread contexts in the code"""
        # Simple implementation - would need more sophisticated analysis
        pass
    
    def _detect_data_races(self):
        """Detect data races in the code"""
        for var, accesses in self.memory_accesses.items():
            if len(accesses) > 1:
                for i in range(len(accesses)):
                    for j in range(i + 1, len(accesses)):
                        access1 = accesses[i]
                        access2 = accesses[j]

                        # Global variables are considered shared across threads
                        if var in self.global_vars:
                            self.result.races.append(RaceCondition(
                                variable=var,
                                access1=access1,
                                access2=access2,
                                severity=SeverityLevel.HIGH,
                                description=f"Potential race condition on {var} between {access1.function}() and {access2.function}()",
                                fix_suggestion=f"Protect access to {var} with a mutex in both functions"
                            ))
                        else:
                            # Check if accesses are in different threads and unprotected
                            if (access1.thread_context != access2.thread_context and 
                                not (access1.is_protected and access2.is_protected)):
                                self.result.races.append(RaceCondition(
                                    variable=var,
                                    access1=access1,
                                    access2=access2,
                                    severity=SeverityLevel.HIGH,
                                    description=f"Potential race condition on {var} between {access1.function}() and {access2.function}()",
                                    fix_suggestion=f"Protect access to {var} with a mutex in both functions"
                                ))
    
    def _analyze_locking_consistency(self):
        """Analyze locking patterns for consistency"""
        for mutex, operations in self.locking_patterns.items():
            lock_count = 0
            for op in operations:
                if op.operation == 'lock':
                    lock_count += 1
                elif op.operation == 'unlock':
                    lock_count -= 1
                    if lock_count < 0:
                        self.result.inconsistent_locking.append(
                            f"Unlock without matching lock for {mutex} in {op.function}() at line {op.line_number}"
                        )
                        lock_count = 0
            
            if lock_count > 0:
                self.result.inconsistent_locking.append(
                    f"Potential lock leak for {mutex} - {lock_count} lock(s) not released"
                )
    
    def _detect_deadlock_potential(self):
        """
        Detect potential deadlocks by analyzing lock acquisition patterns.
        
        This method identifies several types of deadlock risks:
        1. Lock ordering violations (different functions acquiring the same locks in different orders)
        2. Nested locks that could lead to deadlocks
        3. Missing lock releases
        4. Potential deadlock cycles in the lock acquisition graph
        """
        # Build a lock acquisition graph where nodes are functions and edges represent
        # lock acquisition order between functions
        lock_graph = nx.DiGraph()
        function_locks = defaultdict(list)  # function_name -> list of (lock_name, line_number, is_recursive, is_raii)
        function_calls = defaultdict(set)   # function_name -> set of called_functions
        
        # First pass: Track locks and function calls
        print("\n=== Locking Patterns ===")  # Debug
        for mutex, operations in self.locking_patterns.items():
            print(f"\nMutex: {mutex}")
            for op in operations:
                print(f"  - {op.function}(): {op.operation} at line {op.line_number}")
            # Group operations by function
            func_ops = defaultdict(list)
            for op in operations:
                func_ops[op.function].append(op)
                
            # Analyze lock order within each function
            for func, ops in func_ops.items():
                # Skip global scope for RAII detection
                if func == "<global>":
                    continue
                    
                # Sort operations by line number to determine acquisition order
                sorted_ops = sorted(ops, key=lambda x: x.line_number)
                acquired = {}
                
                for op in sorted_ops:
                    if op.operation == 'lock':
                        is_raii = 'lock_guard' in str(op)  # Check if this is an RAII-style lock
                        if mutex in acquired:
                            # Mark as recursive lock
                            function_locks[func].append((mutex, op.line_number, True, is_raii))
                            # Only report recursive locks for non-RAII locks
                            if not is_raii:
                                self.result.deadlock_risks.append(
                                    f"Recursive lock on {mutex} in {func}() at line {op.line_number}"
                                )
                        else:
                            function_locks[func].append((mutex, op.line_number, False, is_raii))
                            
                        # Track lock acquisition with RAII info
                        acquired[mutex] = {
                            'line': op.line_number,
                            'is_raii': is_raii
                        }
                        
                    elif op.operation == 'unlock':
                        if mutex in acquired:
                            del acquired[mutex]
                        else:
                            # Only report unmatched unlocks for non-RAII locks
                            if not any('lock_guard' in str(op2) for op2 in ops if op2.operation == 'lock' and op2.line_number < op.line_number):
                                self.result.deadlock_risks.append(
                                    f"Unlock without matching lock for {mutex} in {func}() at line {op.line_number}"
                                )
                    
                    # Skip call tracking for now as it's not used in current implementation
                
                # Check for potential lock leaks (only for non-RAII locks)
                for lock_name, lock_info in list(acquired.items()):
                    if not lock_info['is_raii']:
                        self.result.deadlock_risks.append(
                            f"Potential lock leak in {func}() - lock '{lock_name}' not released"
                        )
        
        # Second pass: Detect potential lock ordering issues
        # Build a map of lock sequences per function
        print("\n=== Function Lock Sequences ===")  # Debug
        function_lock_sequences = {}
        
        # Only consider functions that acquire at least two different locks
        for func, locks in function_locks.items():
            # Get unique non-recursive locks in acquisition order
            seen_locks = set()
            unique_locks = []
            for lock in locks:
                if not lock[2] and lock[0] not in seen_locks:  # Not recursive and not seen
                    seen_locks.add(lock[0])
                    unique_locks.append(lock[0])
            
            if len(unique_locks) >= 2:
                function_lock_sequences[func] = unique_locks
                print(f"\nFunction: {func}()")
                print(f"  Locks in order: {unique_locks}")
                
                # Add all locks to the lock graph
                for lock in unique_locks:
                    lock_graph.add_node(lock)
                
                # Add edges between locks acquired in sequence
                for i in range(len(unique_locks)):
                    for j in range(i + 1, len(unique_locks)):
                        lock1 = unique_locks[i]
                        lock2 = unique_locks[j]
                        if not lock_graph.has_edge(lock1, lock2):
                            lock_graph.add_edge(lock1, lock2, weight=1)
        
        # Third pass: Detect potential deadlocks by looking for cycles in the lock graph
        try:
            cycles = list(nx.simple_cycles(lock_graph))
            if cycles:
                for cycle in cycles:
                    if len(cycle) >= 2:  # We only care about cycles of length 2 or more
                        cycle_str = " -> ".join(cycle)
                        self.result.deadlock_risks.append(
                            f"Potential deadlock cycle detected: {cycle_str}"
                        )
        except nx.NetworkXNoCycle:
            pass  # No cycles found, which is good
            
        # Fourth pass: Check for inconsistent lock ordering across functions
        func_pairs = list(itertools.combinations(function_lock_sequences.items(), 2))
        for (func1, seq1), (func2, seq2) in func_pairs:
            # Find common locks between the two functions
            common_locks = set(seq1) & set(seq2)
            if len(common_locks) >= 2:
                # Check if the order of common locks is different
                order1 = [lock for lock in seq1 if lock in common_locks]
                order2 = [lock for lock in seq2 if lock in common_locks]
                
                # Check if the relative order of any two locks is different
                for i in range(len(order1)):
                    for j in range(i + 1, len(order1)):
                        lock1, lock2 = order1[i], order1[j]
                        try:
                            idx2_1 = order2.index(lock1)
                            idx2_2 = order2.index(lock2)
                            if idx2_1 > idx2_2:
                                # Different order detected
                                self.result.deadlock_risks.append(
                                    f"Inconsistent lock ordering between {func1}() and {func2}(): "
                                    f"{lock1} before {lock2} vs {lock2} before {lock1}"
                                )
                        except ValueError:
                            continue
        
        # Check for potential deadlocks between different functions
        functions = list(function_lock_sequences.keys())
        for i in range(len(functions)):
            for j in range(i + 1, len(functions)):
                func1 = functions[i]
                func2 = functions[j]
                seq1 = function_lock_sequences[func1]
                seq2 = function_lock_sequences[func2]
                
                # Find all pairs of locks that appear in both sequences in different orders
                for lock1 in seq1:
                    for lock2 in seq1:
                        if lock1 == lock2:
                            continue
                            
                        # Check if these locks appear in the opposite order in the other function
                        try:
                            idx1_1 = seq1.index(lock1)
                            idx1_2 = seq1.index(lock2)
                            
                            # Only proceed if lock1 is before lock2 in the first function
                            if idx1_1 >= idx1_2:
                                continue
                                
                            # Check if lock2 is before lock1 in the second function
                            if lock1 in seq2 and lock2 in seq2:
                                idx2_1 = seq2.index(lock1)
                                idx2_2 = seq2.index(lock2)
                                
                                if idx2_2 < idx2_1:  # Opposite order
                                    # Potential deadlock detected
                                    msg = (
                                        f"Potential deadlock between {func1}() and {func2}(): "
                                        f"{func1}() acquires {lock1} before {lock2}, "
                                        f"while {func2}() acquires them in the opposite order"
                                    )
                                    if msg not in self.result.deadlock_risks:
                                        self.result.deadlock_risks.append(msg)
                        except ValueError:
                            continue
        
        # Third pass: Check for lock hierarchy violations and recursive calls with locks
        for func, locks in function_locks.items():
            lock_stack = []  # Tracks currently held locks in order
            
            # Get all operations in line order (locks, unlocks, and calls)
            all_ops = sorted(locks + [(call, 0, False) for call in function_calls.get(func, [])], 
                           key=lambda x: x[1])
            
            for op in all_ops:
                lock_name, line, is_recursive = op
                
                # If this is a function call, check if it could lead to a deadlock
                if lock_name in function_calls:
                    called_func = lock_name
                    if called_func in function_locks:
                        # Get all non-recursive locks acquired by the called function
                        called_locks = [lock[0] for lock in function_locks[called_func] if not lock[2]]
                        
                        # Check if the called function tries to acquire any locks we're holding
                        for held_lock, _ in lock_stack:
                            if held_lock in called_locks:
                                self.result.deadlock_risks.append(
                                    f"Potential deadlock in {func}() at line {line}: "
                                    f"calls {called_func}() while holding {held_lock}, "
                                    f"and {called_func}() also acquires {held_lock}"
                                )
                # This is a lock operation
                elif is_recursive:
                    continue  # Already handled in first pass
                elif lock_name in [l[0] for l in lock_stack]:
                    # Lock already in stack - check if it's the top of stack
                    if lock_stack[-1][0] != lock_name:
                        self.result.deadlock_risks.append(
                            f"Lock hierarchy violation in {func}() at line {line}: "
                            f"Lock {lock_name} acquired while holding {lock_stack[-1][0]}"
                        )
                    lock_stack = [l for l in lock_stack if l[0] != lock_name]
                else:
                    lock_stack.append((lock_name, line))
        
        # Fourth pass: Check for potential deadlocks in the lock graph
        try:
            # Check for cycles in the lock graph (potential deadlocks)
            cycles = list(nx.simple_cycles(lock_graph))
            for cycle in cycles:
                if len(cycle) > 1:  # Only report cycles with 2+ nodes
                    cycle_str = " -> ".join(cycle)
                    self.result.deadlock_risks.append(
                        f"Potential deadlock cycle detected: {cycle_str}"
                    )
        except nx.NetworkXNoCycle:
            pass  # No cycles found
    
    def _validate_atomic_operations(self):
        """Validate atomic operations"""
        for var, accesses in self.memory_accesses.items():
            if any(not access.is_protected for access in accesses):
                self.result.atomic_violations.append(
                    f"Non-atomic access to {var} without proper synchronization"
                )
    
    def _calculate_metrics(self):
        """Calculate analysis metrics"""
        self.result.metrics = {
            'total_races': len(self.result.races),
            'critical_races': len([r for r in self.result.races if r.severity == SeverityLevel.CRITICAL]),
            'high_races': len([r for r in self.result.races if r.severity == SeverityLevel.HIGH]),
            'medium_races': len([r for r in self.result.races if r.severity == SeverityLevel.MEDIUM]),
            'low_races': len([r for r in self.result.races if r.severity == SeverityLevel.LOW]),
            'locking_issues': len(self.result.inconsistent_locking),
            'deadlock_risks': len(self.result.deadlock_risks),
            'atomic_violations': len(self.result.atomic_violations),
            'shared_variables': len(self.shared_vars),
            'memory_accesses': sum(len(accesses) for accesses in self.memory_accesses.values()),
            'lock_operations': sum(len(ops) for ops in self.locking_patterns.values())
        }
    
    def _add_race_condition(self, variable: str, line_num: int, function: str, 
                          filename: str, description: str, fix_suggestion: str):
        """Helper to add a race condition to results"""
        access = MemoryAccess(
            variable=variable,
            access_type=AccessType.READ if 'if' in function else AccessType.WRITE,
            line_number=line_num,
            function=function,
            thread_context=self._infer_thread_context(function)
        )
        
        # Create a synthetic second access to represent the race
        race_access = MemoryAccess(
            variable=variable,
            access_type=AccessType.WRITE,
            line_number=line_num,
            function="[Concurrent Thread]",
            thread_context="Concurrent Thread"
        )
        
        self.result.races.append(RaceCondition(
            variable=variable,
            access1=access,
            access2=race_access,
            severity=SeverityLevel.CRITICAL,
            description=description,
            fix_suggestion=fix_suggestion
        ))
    
    def _infer_thread_context(self, function_name: str) -> str:
        """Infer which thread context a function runs in"""
        if function_name in self.thread_entry_points:
            return "Background Thread"
        return "Main Thread"
    
    def _detect_async_io_races(self, content: str):
        """Detect async I/O race patterns specific to Monero"""
        # Check for reader thread function patterns
        if 'reader_thread_func' in content:
            # Look for the main read loop
            read_loop = re.search(r'while\s*\(\s*m_running|!m_stdin_has_eof', content)
            if read_loop:
                # Check if the loop properly handles synchronization
                if 'm_response_mutex.lock()' not in content[read_loop.start():read_loop.start()+500]:
                    self.result.inconsistent_locking.append(
                        "WARNING: reader_thread_func may lack proper synchronization in read loop"
                    )
    
    def _generate_monero_specific_fixes(self, races: List[RaceCondition]):
        """Generate Monero-specific fix recommendations"""
        if any('m_read_status' in r.variable for r in races):
            self.result.atomic_violations.extend([
                "\n=== MONERO ASYNC_STDIN_READER FIXES ===",
                "1. Make m_read_status atomic:\n"
                "   - Change: volatile t_state m_read_status;\n"
                "   - To:     std::atomic<t_state> m_read_status;\n"
                "   - Add:    #include <atomic>",
                "\n2. Add proper locking in all methods that access m_read_status:\n"
                "   - boost::unique_lock<boost::mutex> lock(m_response_mutex);\n"
                "   - // Access m_read_status here\n"
                "   - lock.unlock();  // Optional, destructor handles it",
                "\n3. For signal safety:\n"
                "   - Use sigaction() instead of signal()\n"
                "   - Set SA_RESTART flag for system calls\n"
                "   - Consider using self-pipe or signalfd for signal handling",
                "\n4. Testing recommendations:\n"
                "   - Build with -fsanitize=thread\n"
                "   - Run with ThreadSanitizer\n"
                "   - Add stress tests with concurrent signals and I/O"
            ])

# ... (other existing methods remain the same)

def main():
    """Main entry point for the ThreadGuard analyzer"""
    parser = argparse.ArgumentParser(
        description="ThreadGuard: Advanced static analyzer for concurrency bugs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python threadguard.py console_handler.h
  python threadguard.py --output report.txt src/*.cpp
  python threadguard.py --json results.json async_reader.cpp
"""
    )
    
    parser.add_argument('files', nargs='*', help='C++ source files to analyze')
    parser.add_argument('--output', '-o', help='Output file for the report')
    parser.add_argument('--json', help='Output JSON report to file')
    parser.add_argument('--ci-mode', action='store_true', 
                       help='Run in CI mode with non-zero exit on issues')
    parser.add_argument('--max-critical', type=int, default=0,
                       help='Maximum allowed critical issues (CI mode)')
    parser.add_argument('--max-high', type=int, default=5,
                       help='Maximum allowed high severity issues (CI mode)')
    
    args = parser.parse_args()
    
    if not args.files:
        parser.print_help()
        sys.exit(1)
    
    analyzer = ThreadGuardAnalyzer()
    
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            continue
            
        result = analyzer.analyze_file(path)
        report = format_analysis_report(result, str(path))
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
        else:
            print(report)
            
        if args.json:
            with open(args.json, 'w') as f:
                json.dump({
                    'races': len(result.races),
                    'critical_races': len([r for r in result.races if r.severity == SeverityLevel.CRITICAL]),
                    'high_races': len([r for r in result.races if r.severity == SeverityLevel.HIGH]),
                    'locking_issues': len(result.inconsistent_locking),
                    'deadlock_risks': len(result.deadlock_risks),
                    'atomic_violations': len(result.atomic_violations)
                }, f, indent=2)
    
    if args.ci_mode:
        critical_races = len([r for r in result.races if r.severity == SeverityLevel.CRITICAL])
        high_races = len([r for r in result.races if r.severity == SeverityLevel.HIGH])
        
        if critical_races > args.max_critical:
            print(f"❌ CI FAILURE: {critical_races} critical issues found (max allowed: {args.max_critical})")
            sys.exit(2)
        elif high_races > args.max_high:
            print(f"⚠️  CI WARNING: {high_races} high severity issues found (max allowed: {args.max_high})")
            sys.exit(1)
        else:
            print("✅ CI PASSED: No critical concurrency issues found")
            sys.exit(0)

def format_analysis_report(result: AnalysisResult, filename: str) -> str:
    """Format analysis results into a comprehensive report"""
    report = []
    report.append("=" * 80)
    report.append(f"ThreadGuard Analysis Report - {filename}")
    report.append("=" * 80)
    report.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}")
    
    # Summary
    critical_races = len([r for r in result.races if r.severity == SeverityLevel.CRITICAL])
    high_races = len([r for r in result.races if r.severity == SeverityLevel.HIGH])
    
    report.append("\n🔍 SUMMARY")
    report.append("-" * 40)
    report.append(f"Critical Races: {critical_races}")
    report.append(f"High Severity Races: {high_races}")
    report.append(f"Locking Issues: {len(result.inconsistent_locking)}")
    report.append(f"Deadlock Risks: {len(result.deadlock_risks)}")
    
    # Critical Issues
    if critical_races > 0:
        report.append("\n🚨 CRITICAL ISSUES")
        report.append("-" * 40)
        for race in [r for r in result.races if r.severity == SeverityLevel.CRITICAL]:
            report.append(f"• {race.description}")
            report.append(f"  File: {filename}:{race.access1.line_number}")
            report.append(f"  Fix: {race.fix_suggestion}")
    
    # High Severity Issues
    if high_races > 0:
        report.append("\n⚠️  HIGH SEVERITY ISSUES")
        report.append("-" * 40)
        for race in [r for r in result.races if r.severity == SeverityLevel.HIGH]:
            report.append(f"• {race.description}")
            report.append(f"  File: {filename}:{race.access1.line_number}")
    
    # Locking Issues
    if result.inconsistent_locking:
        report.append("\n🔒 LOCKING ISSUES")
        report.append("-" * 40)
        for issue in result.inconsistent_locking:
            report.append(f"• {issue}")
    
    # Atomic Violations
    if result.atomic_violations:
        report.append("\n⚛️  ATOMIC OPERATIONS")
        report.append("-" * 40)
        for violation in result.atomic_violations:
            report.append(violation)
    
    return "\n".join(report)

if __name__ == "__main__":
    main()
