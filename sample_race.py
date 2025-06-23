"""Python implementation of race condition for testing"""
import threading


class UnsafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()  # Will be instrumented

    def increment(self):
        # Simulate unsafe access pattern
        self.value += 1


class SafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1


def run_counter_test(counter_class, threads=5, iterations=10000):
    counter = counter_class()
    threads = []

    def worker():
        for _ in range(iterations):
            counter.increment()

    for _ in range(threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return counter.value
