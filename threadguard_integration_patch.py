#!/usr/bin/env python3
"""
ThreadGuard Integration Patch

This patch integrates the enhanced fixes into the main threadguard_enhanced.py file.
Apply this by merging the methods into your ThreadGuardAnalyzer class.

Key Fixes:
1. Missing unlock detection with early returns
2. Complex deadlock detection in nested method calls  
3. STL container race detection

Usage:
1. Add the new methods to your ThreadGuardAnalyzer class
2. Add the new method calls to the analyze_file method
3. Add the new data structures to __init__
"""

# ADD THESE IMPORTS TO THE TOP OF threadguard_enhanced.py
"""
from collections import deque
from typing import Set
"""

# ADD THESE DATACLASSES AFTER THE EXISTING ONES
"""
@dataclass
class ControlFlowPath:
    function_name: str
    entry_line: int
    exit_line: int
    path_type: str  # "normal", "early_return", "exception", "break", "continue"
    locks_acquired: List[str]
    locks_released: List[str]
    has_early_exit: bool = False

@dataclass
class STLContainerAccess:
    container_name: str
    container_type: str  # "vector", "map", "set", etc.
    access_method: str   # "push_back", "insert", "erase", "[]", etc.
    is_thread_safe: bool
    line_number: int
    function: str
"""

# ADD THESE TO ThreadGuardAnalyzer.__init__ method
def add_to_init():
    """
    Add these lines to the ThreadGuardAnalyzer.__init__ method
    """
    return """
    # Enhanced tracking for test failure fixes
    self.control_flow_paths = defaultdict(list)
    self.stl_container_accesses = defaultdict(list)
    self.method_call_graph = defaultdict(set)
    self.recursive_locks = defaultdict(set)
    
    # STL containers that are NOT thread-safe
    self.non_thread_safe_containers = {
        'std::vector', 'std::deque', 'std::list', 'std::forward_list',
        'std::array', 'std::map', 'std::multimap', 'std::set', 'std::multiset',
        'std::unordered_map', 'std::unordered_multimap', 
        'std::unordered_set', 'std::unordered_multiset',
        'std::string', 'std::wstring'
    }
    
    # Thread-safe containers (C++11+)
    self.thread_safe_containers = {
        'std::atomic', 'std::shared_ptr'  # Limited thread safety
    }
    """

# ADD THESE PATTERNS TO _compile_patterns method
def add_to_compile_patterns():
    """
    Add these patterns to the existing _compile_patterns method
    """
    return """
    # Add these to self.patterns dictionary:

