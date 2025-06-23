#!/usr/bin/env python3
"""
ThreadGuard Runtime Monitor (TGRM) - Dynamic Mutex Analysis

Monitors mutex usage at runtime to detect:
- Actual mutex utilization patterns
- Lock contention hotspots
- Unused mutexes
- Lock hierarchy violations
- Potential deadlock scenarios

Authors: Pradeep Kumar
License: MIT
Version: 1.0.0
Date: 2025-06-24
"""

import inspect
import json
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class LockType(Enum):
    MUTEX = "mutex"
    RWLOCK = "rwlock"
    SPINLOCK = "spinlock"
    UNKNOWN = "unknown"

class LockOperation(Enum):
    LOCK = "acquire"
    UNLOCK = "release"
    TRY_LOCK = "try_acquire"
    READ_LOCK = "read_acquire"
    WRITE_LOCK = "write_acquire"

@dataclass
class MutexProfile:
    name: str
    lock_type: LockType
    created_at: float
    created_by: str
    lock_count: int = 0
    wait_time: float = 0.0
    hold_time: float = 0.0
    contention_count: int = 0
    last_used: float = 0.0
    protection_scope: Set[str] = field(default_factory=set)

@dataclass
class LockEvent:
    thread_id: int
    thread_name: str
    operation: LockOperation
    timestamp: float
    duration: float = 0.0
    call_stack: List[str] = field(default_factory=list)
    protected_resource: Optional[str] = None

@dataclass
class DeadlockRisk:
    chain: List[Tuple[str, int]]
    resources: Set[str]
    severity: str
    detection_time: float

@dataclass
class RuntimeReport:
    active_mutexes: Dict[str, MutexProfile]
    lock_events: List[LockEvent]
    deadlock_risks: List[DeadlockRisk]
    unused_mutexes: Set[str]
    high_contention: List[str]
    metrics: Dict[str, Any]

class ThreadGuardRuntime:
    """Dynamic runtime monitor for mutex usage analysis"""

    def __init__(self, track_unused=True, deadlock_detection=True):
        self.track_unused = track_unused
        self.deadlock_detection = deadlock_detection
        self.mutex_registry = {}
        self.lock_events = []
        self.active_locks = defaultdict(deque)
        self.lock_graph = defaultdict(set)
        self.resource_map = defaultdict(set)
        self.thread_state = defaultdict(dict)
        self.start_time = time.time()
        self.unused_mutexes = set()

        # Hook synchronization primitives
        self._install_hooks()

    def _install_hooks(self):
        """Monkey-patch threading primitives for instrumentation"""
        self.original_lock = threading.Lock
        self.original_rlock = threading.RLock
        self.original_condition = threading.Condition

        # Lock instrumentation
        def instrumented_lock(*args, **kwargs):
            lock = self.original_lock(*args, **kwargs)
            return self._wrap_lock(lock, LockType.MUTEX)

        # RLock instrumentation
        def instrumented_rlock(*args, **kwargs):
            rlock = self.original_rlock(*args, **kwargs)
            return self._wrap_lock(rlock, LockType.MUTEX)

        # Condition variable instrumentation
        def instrumented_condition(*args, **kwargs):
            cond = self.original_condition(*args, **kwargs)
            cond._is_condition = True
            return self._wrap_lock(cond, LockType.MUTEX)

        threading.Lock = instrumented_lock
        threading.RLock = instrumented_rlock
        threading.Condition = instrumented_condition

    def _wrap_lock(self, lock, lock_type):
        """Wrap lock object with instrumentation"""
        lock_id = id(lock)
        creation_time = time.time()
        creator = self._get_caller_context(2)

        # Create profile
        profile = MutexProfile(
            name=f"mutex_{lock_id}",
            lock_type=lock_type,
            created_at=creation_time,
            created_by=creator
        )

        self.mutex_registry[lock_id] = profile
        if self.track_unused:
            self.unused_mutexes.add(lock_id)

        # Instrument acquire/release
        original_acquire = lock.acquire
        original_release = lock.release

        def instrumented_acquire(blocking=True, timeout=-1):
            start_time = time.time()
            thread_id = threading.get_ident()
            thread_name = threading.current_thread().name

            # Record lock attempt
            pre_event = LockEvent(
                thread_id=thread_id,
                thread_name=thread_name,
                operation=LockOperation.LOCK,
                timestamp=start_time,
                call_stack=self._capture_stack(),
                protected_resource=self._detect_protected_resource()
            )

            # Actual acquire
            result = original_acquire(blocking, timeout)
            end_time = time.time()

            # Record acquisition
            duration = end_time - start_time
            post_event = LockEvent(
                thread_id=thread_id,
                thread_name=thread_name,
                operation=LockOperation.LOCK,
                timestamp=end_time,
                duration=duration,
                call_stack=pre_event.call_stack,
                protected_resource=pre_event.protected_resource
            )

            # Update profile
            profile = self.mutex_registry[lock_id]
            profile.lock_count += 1
            profile.wait_time += duration
            profile.last_used = end_time
            profile.contention_count += (1 if duration > 0.001 else 0)

            if self.track_unused and lock_id in self.unused_mutexes:
                self.unused_mutexes.remove(lock_id)

            # Update thread state
            self.active_locks[thread_id].append(lock_id)
            self.thread_state[thread_id][lock_id] = end_time

            # Build lock graph for deadlock detection
            if self.active_locks[thread_id]:
                previous_lock = self.active_locks[thread_id][-2] if len(self.active_locks[thread_id]) > 1 else None
                if previous_lock:
                    self.lock_graph[previous_lock].add(lock_id)

            # Map resource to lock
            if post_event.protected_resource:
                profile.protection_scope.add(post_event.protected_resource)
                self.resource_map[post_event.protected_resource].add(lock_id)

            self.lock_events.extend([pre_event, post_event])
            return result

        def instrumented_release():
            release_time = time.time()
            thread_id = threading.get_ident()
            thread_name = threading.current_thread().name

            if self.active_locks[thread_id]:
                lock_id = self.active_locks[thread_id].pop()
                acquire_time = self.thread_state[thread_id].pop(lock_id, release_time)
                hold_duration = release_time - acquire_time

                # Update profile
                profile = self.mutex_registry.get(lock_id)
                if profile:
                    profile.hold_time += hold_duration

                # Record release
                event = LockEvent(
                    thread_id=thread_id,
                    thread_name=thread_name,
                    operation=LockOperation.UNLOCK,
                    timestamp=release_time,
                    duration=hold_duration,
                    call_stack=self._capture_stack(),
                    protected_resource=self._detect_protected_resource()
                )
                self.lock_events.append(event)

            return original_release()

        lock.acquire = instrumented_acquire
        lock.release = instrumented_release
        return lock

    def _detect_protected_resource(self) -> Optional[str]:
        """Heuristically identify protected resources"""
        try:
            stack = inspect.stack(0)
            for frame in stack[3:6]:  # Look 3-5 frames deep
                code = frame.frame.f_code
                # Identify variable access patterns
                if 'self.' in code.co_names:
                    return f"{code.co_name}::self"
                for var in code.co_varnames:
                    if var.startswith('m_') or var.endswith('_'):
                        return f"{code.co_name}::{var}"
            return None
        except:
            return None

    def _capture_stack(self, depth=5) -> List[str]:
        """Capture simplified call stack"""
        stack = []
        try:
            frames = inspect.stack()[1:depth+1]
            for frame in reversed(frames):
                frame_info = inspect.getframeinfo(frame.frame)
                stack.append(f"{frame_info.function}@{frame_info.filename}:{frame_info.lineno}")
        except:
            pass
        return stack

    def _get_caller_context(self, depth=1) -> str:
        """Get context of lock creation"""
        try:
            frame = inspect.stack()[depth]
            return f"{frame.filename}:{frame.lineno}"
        except:
            return "unknown"

    def detect_deadlocks(self):
        """Detect potential deadlocks using lock graph"""
        if not self.deadlock_detection:
            return []

        deadlock_risks = []
        # Convert graph to nodes and edges
        nodes = set(self.lock_graph.keys())
        for children in self.lock_graph.values():
            nodes.update(children)

        # Look for cycles
        for start in nodes:
            visited = set()
            stack = [(start, [start])]

            while stack:
                node, path = stack.pop()
                visited.add(node)

                for neighbor in self.lock_graph.get(node, []):
                    if neighbor in path:
                        # Cycle detected
                        cycle = path[path.index(neighbor):] + [neighbor]
                        if len(cycle) > 2:  # Meaningful cycle
                            resources = set()
                            for lock_id in cycle:
                                if profile := self.mutex_registry.get(lock_id):
                                    resources.update(profile.protection_scope)

                            deadlock_risks.append(DeadlockRisk(
                                chain=[(self.mutex_registry.get(lock_id, "unknown").name for lock_id in cycle],
                                resources=resources,
                                severity="HIGH" if len(resources) > 0 else "MEDIUM",
                                detection_time=time.time()
                            ))
                    elif neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))

        return deadlock_risks

    def generate_report(self) -> RuntimeReport:
        """Generate runtime analysis report"""
        # Identify high-contention mutexes
        high_contention = []
        for profile in self.mutex_registry.values():
            if profile.contention_count > 10 and profile.wait_time > 0.01 * profile.lock_count:
                high_contention.append(profile.name)

        # Detect deadlock risks
        deadlock_risks = self.detect_deadlocks()

        # Prepare metrics
        metrics = {
            "total_mutexes": len(self.mutex_registry),
            "active_mutexes": sum(1 for p in self.mutex_registry.values() if p.lock_count > 0),
            "total_lock_operations": len(self.lock_events),
            "total_wait_time": sum(p.wait_time for p in self.mutex_registry.values()),
            "total_hold_time": sum(p.hold_time for p in self.mutex_registry.values()),
            "deadlock_risks": len(deadlock_risks),
            "high_contention": len(high_contention),
            "unused_mutexes": len(self.unused_mutexes),
            "monitoring_duration": time.time() - self.start_time
        }

        return RuntimeReport(
            active_mutexes={k: v for k, v in self.mutex_registry.items() if v.lock_count > 0},
            lock_events=self.lock_events,
            deadlock_risks=deadlock_risks,
            unused_mutexes={self.mutex_registry[lock_id].name for lock_id in self.unused_mutexes},
            high_contention=high_contention,
            metrics=metrics
        )

    def save_report(self, filename: str):
        """Save report to JSON file"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

def monitor_demo():
    """Demo of runtime monitoring capabilities"""
    print("Starting ThreadGuard Runtime Monitor demo...")
    monitor = ThreadGuardRuntime()

    # Create shared resources
    shared_counter = 0
    shared_list = []

    # Create mutexes
    counter_lock = threading.Lock()
    list_lock = threading.RLock()
    unused_lock = threading.Lock()  # Will remain unused

    def counter_task():
        nonlocal shared_counter
        for _ in range(1000):
            with counter_lock:
                shared_counter += 1

    def list_task():
        nonlocal shared_list
        for i in range(1000):
            with list_lock:
                shared_list.append(i)
                # Simulate nested locking
                if i % 100 == 0:
                    with counter_lock:
                        shared_list.remove(i)

    # Start threads
    threads = [
        threading.Thread(target=counter_task),
        threading.Thread(target=list_task),
        threading.Thread(target=counter_task)
    ]

    for t in threads:
        t.start()

    # Let threads run for a bit
    time.sleep(0.5)

    # Generate report
    report = monitor.generate_report()
    print("\nRuntime Monitoring Results:")
    print(f"Active mutexes: {len(report.active_mutexes)}")
    print(f"Unused mutexes: {len(report.unused_mutexes)}")
    print(f"Deadlock risks detected: {len(report.deadlock_risks)}")
    print(f"High contention mutexes: {report.high_contention}")

    # Save full report
    monitor.save_report("runtime_monitor_report.json")
    print("Full report saved to runtime_monitor_report.json")

if __name__ == "__main__":
    # For demo purposes - in real usage, import and create monitor
    # before starting application threads
    monitor_demo()
