#!/usr/bin/env python3
"""
Simple Mutex Usage Checker

A straightforward tool that reads source code and determines:
1. Are mutexes declared?
2. Are they actually used for synchronization?
3. Which shared variables are protected vs unprotected?

This is much simpler than complex static analysis - just pattern matching!

Author: Pradeep Kumar
Version: 2.0.0 (Simplified)
Date: 2025-06-23
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
from collections import defaultdict

@dataclass
class MutexInfo:
    """Information about a mutex in the code"""
    name: str
    declared_line: int
    lock_usages: List[int] = None
    unlock_usages: List[int] = None
    raii_usages: List[int] = None  # lock_guard, unique_lock, etc.
    
    def __post_init__(self):
        if self.lock_usages is None:
            self.lock_usages = []
        if self.unlock_usages is None:
            self.unlock_usages = []
        if self.raii_usages is None:
            self.raii_usages = []
    
    @property
    def is_used(self) -> bool:
        """Check if mutex is actually used (not just declared)"""
        return len(self.lock_usages) > 0 or len(self.raii_usages) > 0
    
    @property
    def usage_count(self) -> int:
        """Total number of times mutex is used"""
        return len(self.lock_usages) + len(self.raii_usages)

@dataclass
class SharedVariable:
    """Information about a shared variable"""
    name: str
    declared_line: int
    access_lines: List[int] = None
    protected_access_lines: List[int] = None
    
    def __post_init__(self):
        if self.access_lines is None:
            self.access_lines = []
        if self.protected_access_lines is None:
            self.protected_access_lines = []
    
    @property
    def has_unprotected_access(self) -> bool:
        """Check if variable has unprotected access"""
        unprotected = set(self.access_lines) - set(self.protected_access_lines)
        return len(unprotected) > 0
    
    @property
    def protection_ratio(self) -> float:
        """Ratio of protected vs total accesses"""
        if not self.access_lines:
            return 1.0
        return len(self.protected_access_lines) / len(self.access_lines)

class SimpleMutexChecker:
    """
    Simple, direct approach to checking mutex usage in C++ code
    """
    
    def __init__(self):
        # Simple regex patterns for common mutex patterns
        self.patterns = {
            # Mutex declarations
            'mutex_decl': re.compile(r'(std::\s*)?mutex\s+(\w+)', re.IGNORECASE),
            'boost_mutex_decl': re.compile(r'boost::\s*mutex\s+(\w+)', re.IGNORECASE),
            'recursive_mutex_decl': re.compile(r'(std::\s*)?recursive_mutex\s+(\w+)', re.IGNORECASE),
            
            # Member variable mutexes
            'member_mutex': re.compile(r'\b(m_\w*mutex\w*|m_\w*lock\w*)\b', re.IGNORECASE),
            
            # Mutex usage
            'explicit_lock': re.compile(r'(\w+)\.lock\s*\(\s*\)'),
            'explicit_unlock': re.compile(r'(\w+)\.unlock\s*\(\s*\)'),
            'try_lock': re.compile(r'(\w+)\.try_lock\s*\(\s*\)'),
            
            # RAII locking
            'lock_guard': re.compile(r'lock_guard\s*<[^>]*>\s*\w*\s*\(\s*(\w+)\s*\)'),
            'unique_lock': re.compile(r'unique_lock\s*<[^>]*>\s*\w*\s*\(\s*(\w+)\s*\)'),
            'scoped_lock': re.compile(r'scoped_lock\s*<[^>]*>\s*\w*\s*\(\s*(\w+)\s*\)'),
            
            # Shared variables (common patterns)
            'global_var': re.compile(r'^\s*(?:extern\s+)?(?:static\s+)?(?:int|long|bool|char|float|double|size_t)\s+(\w+)', re.MULTILINE),
            'member_var': re.compile(r'\b(m_\w+)\b'),
            'static_member': re.compile(r'static\s+\w+\s+(\w+)'),
            
            # Thread creation (indicates potential shared access)
            'thread_creation': re.compile(r'(std::\s*)?thread\s+\w+|pthread_create|boost::\s*thread'),
            
            # Atomic variables
            'atomic_var': re.compile(r'(std::\s*)?atomic\s*<[^>]*>\s*(\w+)'),
        }
    
    def check_file(self, filepath: Path) -> Dict:
        """
        Simple check: read file and determine mutex usage
        """
        if not filepath.exists():
            return {"error": f"File not found: {filepath}"}
        
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            # Find all mutexes
            mutexes = self._find_mutexes(lines)
            
            # Find shared variables
            shared_vars = self._find_shared_variables(lines)
            
            # Check if threading is used
            has_threading = self._check_threading_usage(content)
            
            # Analyze protection
            protection_analysis = self._analyze_protection(lines, mutexes, shared_vars)
            
            return {
                "file": str(filepath),
                "has_threading": has_threading,
                "mutexes": mutexes,
                "shared_variables": shared_vars,
                "protection_analysis": protection_analysis,
                "summary": self._generate_summary(mutexes, shared_vars, has_threading)
            }
            
        except Exception as e:
            return {"error": f"Error analyzing {filepath}: {str(e)}"}
    
    def _find_mutexes(self, lines: List[str]) -> Dict[str, MutexInfo]:
        """Find all mutex declarations and their usage"""
        mutexes = {}
        
        for i, line in enumerate(lines, 1):
            # Check for mutex declarations
            for pattern_name, pattern in [
                ('mutex_decl', self.patterns['mutex_decl']),
                ('boost_mutex_decl', self.patterns['boost_mutex_decl']),
                ('recursive_mutex_decl', self.patterns['recursive_mutex_decl']),
                ('member_mutex', self.patterns['member_mutex'])
            ]:
                matches = pattern.findall(line)
                for match in matches:
                    if isinstance(match, tuple):
                        mutex_name = match[-1]  # Get the last group (variable name)
                    else:
                        mutex_name = match
                    
                    if mutex_name and mutex_name not in mutexes:
                        mutexes[mutex_name] = MutexInfo(
                            name=mutex_name,
                            declared_line=i
                        )
        
        # Find mutex usage
        for i, line in enumerate(lines, 1):
            # Explicit lock/unlock
            lock_match = self.patterns['explicit_lock'].search(line)
            if lock_match:
                mutex_name = lock_match.group(1)
                if mutex_name in mutexes:
                    mutexes[mutex_name].lock_usages.append(i)
            
            unlock_match = self.patterns['explicit_unlock'].search(line)
            if unlock_match:
                mutex_name = unlock_match.group(1)
                if mutex_name in mutexes:
                    mutexes[mutex_name].unlock_usages.append(i)
            
            # RAII usage
            for pattern_name in ['lock_guard', 'unique_lock', 'scoped_lock']:
                raii_match = self.patterns[pattern_name].search(line)
                if raii_match:
                    mutex_name = raii_match.group(1)
                    if mutex_name in mutexes:
                        mutexes[mutex_name].raii_usages.append(i)
        
        return mutexes
    
    def _find_shared_variables(self, lines: List[str]) -> Dict[str, SharedVariable]:
        """Find variables that might be shared between threads"""
        shared_vars = {}
        
        for i, line in enumerate(lines, 1):
            # Global variables
            global_matches = self.patterns['global_var'].findall(line)
            for var_name in global_matches:
                if var_name not in shared_vars:
                    shared_vars[var_name] = SharedVariable(
                        name=var_name,
                        declared_line=i
                    )
            
            # Member variables (m_ prefix typically indicates shared state)
            member_matches = self.patterns['member_var'].findall(line)
            for var_name in member_matches:
                if var_name not in shared_vars:
                    shared_vars[var_name] = SharedVariable(
                        name=var_name,
                        declared_line=i
                    )
            
            # Static members
            static_matches = self.patterns['static_member'].findall(line)
            for var_name in static_matches:
                if var_name not in shared_vars:
                    shared_vars[var_name] = SharedVariable(
                        name=var_name,
                        declared_line=i
                    )
        
        # Find variable accesses
        for var_name, var_info in shared_vars.items():
            for i, line in enumerate(lines, 1):
                if var_name in line:
                    # Simple check: if variable appears in line
                    var_info.access_lines.append(i)
                    
                    # Check if this access is protected (mutex usage nearby)
                    if self._is_access_protected(lines, i, var_name):
                        var_info.protected_access_lines.append(i)
        
        return shared_vars
    
    def _check_threading_usage(self, content: str) -> bool:
        """Check if the code uses threading"""
        return bool(self.patterns['thread_creation'].search(content))
    
    def _is_access_protected(self, lines: List[str], line_num: int, var_name: str) -> bool:
        """
        Simple check: is variable access protected by a mutex?
        Look at surrounding lines for lock patterns
        """
        # Check 5 lines before and 2 lines after for mutex usage
        start = max(0, line_num - 6)
        end = min(len(lines), line_num + 3)
        
        context = '\n'.join(lines[start:end])
        
        # Look for any mutex-related patterns in the context
        protection_patterns = [
            r'\.lock\s*\(',
            r'lock_guard',
            r'unique_lock',
            r'scoped_lock',
            r'std::lock',
            r'boost::lock_guard'
        ]
        
        return any(re.search(pattern, context, re.IGNORECASE) for pattern in protection_patterns)
    
    def _analyze_protection(self, lines: List[str], mutexes: Dict[str, MutexInfo], 
                          shared_vars: Dict[str, SharedVariable]) -> Dict:
        """Analyze the overall protection level"""
        total_mutexes = len(mutexes)
        used_mutexes = sum(1 for m in mutexes.values() if m.is_used)
        
        total_shared_vars = len(shared_vars)
        protected_vars = sum(1 for v in shared_vars.values() if v.protection_ratio > 0.5)
        
        unprotected_accesses = []
        for var_name, var_info in shared_vars.items():
            if var_info.has_unprotected_access:
                unprotected_count = len(var_info.access_lines) - len(var_info.protected_access_lines)
                unprotected_accesses.append({
                    "variable": var_name,
                    "total_accesses": len(var_info.access_lines),
                    "unprotected_accesses": unprotected_count,
                    "protection_ratio": var_info.protection_ratio
                })
        
        return {
            "total_mutexes": total_mutexes,
            "used_mutexes": used_mutexes,
            "unused_mutexes": total_mutexes - used_mutexes,
            "total_shared_variables": total_shared_vars,
            "protected_variables": protected_vars,
            "unprotected_variables": total_shared_vars - protected_vars,
            "unprotected_accesses": unprotected_accesses
        }
    
    def _generate_summary(self, mutexes: Dict, shared_vars: Dict, has_threading: bool) -> Dict:
        """Generate a simple summary"""
        if not has_threading:
            return {
                "status": "OK",
                "message": "No threading detected - synchronization not required",
                "risk_level": "NONE"
            }
        
        used_mutexes = sum(1 for m in mutexes.values() if m.is_used)
        total_shared_vars = len(shared_vars)
        unprotected_vars = sum(1 for v in shared_vars.values() if v.has_unprotected_access)
        
        if total_shared_vars > 0 and used_mutexes == 0:
            return {
                "status": "HIGH RISK",
                "message": f"Threading detected with {total_shared_vars} shared variables but no mutex usage",
                "risk_level": "CRITICAL"
            }
        
        if unprotected_vars > total_shared_vars * 0.5:
            return {
                "status": "MEDIUM RISK", 
                "message": f"{unprotected_vars}/{total_shared_vars} shared variables appear unprotected",
                "risk_level": "HIGH"
            }
        
        if used_mutexes > 0 and unprotected_vars == 0:
            return {
                "status": "GOOD",
                "message": f"Mutexes in use ({used_mutexes}) and variables appear protected",
                "risk_level": "LOW"
            }
        
        return {
            "status": "REVIEW NEEDED",
            "message": f"Mixed protection: {used_mutexes} mutexes, {unprotected_vars} unprotected variables",
            "risk_level": "MEDIUM"
        }

def format_simple_report(result: Dict) -> str:
    """Format results in a simple, readable format"""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    report = []
    report.append("=" * 80)
    report.append(f"Simple Mutex Usage Report: {result['file']}")
    report.append("=" * 80)
    
    # Summary
    summary = result['summary']
    status_emoji = {
        "OK": "✅",
        "GOOD": "✅", 
        "REVIEW NEEDED": "⚠️",
        "MEDIUM RISK": "⚠️",
        "HIGH RISK": "🚨"
    }
    
    emoji = status_emoji.get(summary['status'], "❓")
    report.append(f"\n{emoji} STATUS: {summary['status']}")
    report.append(f"   {summary['message']}")
    report.append(f"   Risk Level: {summary['risk_level']}")
    
    # Threading detection
    threading_status = "✅ Detected" if result['has_threading'] else "❌ Not detected"
    report.append(f"\n🧵 Threading: {threading_status}")
    
    # Mutex analysis
    protection = result['protection_analysis']
    report.append(f"\n🔒 Mutex Analysis:")
    report.append(f"   • Total mutexes declared: {protection['total_mutexes']}")
    report.append(f"   • Mutexes actually used: {protection['used_mutexes']}")
    if protection['unused_mutexes'] > 0:
        report.append(f"   • ⚠️ Unused mutexes: {protection['unused_mutexes']}")
    
    # Variable analysis
    report.append(f"\n📊 Shared Variables:")
    report.append(f"   • Total shared variables: {protection['total_shared_variables']}")
    report.append(f"   • Protected variables: {protection['protected_variables']}")
    if protection['unprotected_variables'] > 0:
        report.append(f"   • ⚠️ Unprotected variables: {protection['unprotected_variables']}")
    
    # Detailed mutex info
    if result['mutexes']:
        report.append(f"\n🔐 Mutex Details:")
        for mutex_name, mutex_info in result['mutexes'].items():
            status = "✅ USED" if mutex_info.is_used else "❌ DECLARED BUT UNUSED"
            report.append(f"   • {mutex_name} (line {mutex_info.declared_line}): {status}")
            if mutex_info.is_used:
                report.append(f"     - Lock calls: {len(mutex_info.lock_usages)}")
                report.append(f"     - RAII usage: {len(mutex_info.raii_usages)}")
    
    # Unprotected accesses
    if protection['unprotected_accesses']:
        report.append(f"\n⚠️ Unprotected Variable Accesses:")
        for access in protection['unprotected_accesses']:
            protection_pct = access['protection_ratio'] * 100
            report.append(f"   • {access['variable']}: {access['unprotected_accesses']}/{access['total_accesses']} unprotected ({protection_pct:.0f}% protected)")
    
    return '\n'.join(report)

def main():
    """Main function for simple mutex checker"""
    if len(sys.argv) < 2:
        print("Usage: python3 simple_mutex_checker.py <file.cpp> [file2.cpp ...]")
        print("\nExample: python3 simple_mutex_checker.py src/async_reader.cpp")
        sys.exit(1)
    
    checker = SimpleMutexChecker()
    
    for filepath in sys.argv[1:]:
        path = Path(filepath)
        result = checker.check_file(path)
        report = format_simple_report(result)
        print(report)
        print()

if __name__ == "__main__":
    main()


