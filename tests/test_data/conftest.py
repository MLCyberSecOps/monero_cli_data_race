"""Pytest fixtures for runtime monitoring tests"""

# Define fixtures here as needed, e.g.:

import pytest


@pytest.fixture
def example_report():
    """Provide a basic empty report structure"""
    return {
        "active_mutexes": {},
        "lock_events": [],
        "deadlock_risks": [],
        "unused_mutexes": [],
        "high_contention": [],
        "metrics": {},
    }
