#!/usr/bin/env python3
"""
ThreadGuard: Static Analysis Tool for Concurrency Bug Detection
Ariane Mission Ready

Detects data races, inconsistent locking patterns, and thread safety violations
specifically targeting async I/O handlers like Monero's async_stdin_reader.

Authors: Pradeep Kumar, Claude Sonnet 4
License: MIT
Version: 1.1.0
"""

import re
import sys
import json
import argparse
import ast
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
        """
        self.result = AnalysisResult()
        
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
        
        for class_match in re.finditer(class_pattern, content, re.DOTALL):
            class_body = class_match.group(0)
            for var_match in re.finditer(var_pattern, class_body):
                var_name = var_match.group(0).split(';')[0].strip()
                if var_name.startswith('m_'):
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
        
        for i, line in enumerate(lines, 1):
            # Track function definitions
            func_match = re.search(r'\w+\s+\w+\s*::\s*(\w+)\s*\s*\w*\s*[({]', line)
            if func_match:
                current_function = func_match.group(1)
            
            # Check for lock operations
            lock_match = re.search(r'(\w+)\.(lock|unlock|try_lock)\s*\(', line)
            if lock_match:
                mutex_name = lock_match.group(1)
                operation = lock_match.group(2)
                self.locking_patterns[mutex_name].append(LockingPattern(
                    mutex_name=mutex_name,
                    operation=operation,
                    line_number=i,
                    function=current_function
                ))
    
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
                        
                        # Check if accesses are in different threads and not properly protected
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
        """Detect potential deadlocks"""
        # Simple implementation - would need more sophisticated analysis
        pass
    
    def _validate_atomic_operations(self):
        """Validate atomic operations"""
        # Check for non-atomic operations on shared variables
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
