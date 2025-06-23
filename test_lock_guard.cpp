#include <mutex>
#include <vector>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <numeric>
#include <algorithm>

/**
 * @brief A thread-safe wrapper around std::vector with basic operations
 *
 * This class provides thread-safe operations on a vector using std::mutex
 * and RAII lock guards for exception safety.
 */
class ThreadSafeVector {
    std::vector<int> data;
    mutable std::mutex mtx;
    std::atomic<bool> throw_on_push{false};

public:
    /**
     * @brief Add an element to the end of the vector
     * @param value The value to add
     * @throws std::runtime_error if throw_on_push is set to true (for testing)
     */
    void push_back(int value) {
        std::lock_guard<std::mutex> lock(mtx);
        if (throw_on_push) {
            throw std::runtime_error("Test exception on push");
        }
        data.push_back(value);
    }

    /**
     * @brief Get the current size of the vector
     * @return The number of elements in the vector
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return data.size();
    }

    /**
     * @brief Access element at specified index with bounds checking
     * @param index The index of the element to access
     * @return Reference to the requested element
     * @throws std::out_of_range if index is out of bounds
     */
    int at(size_t index) const {
        std::lock_guard<std::mutex> lock(mtx);
        return data.at(index);
    }

    /**
     * @brief Remove all elements from the vector
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        data.clear();
    }

    /**
     * @brief Check if the vector is empty
     * @return true if the vector is empty, false otherwise
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return data.empty();
    }

    /**
     * @brief Calculate the sum of all elements in the vector
     * @return The sum of all elements
     */
    int sum() const {
        std::lock_guard<std::mutex> lock(mtx);
        return std::accumulate(data.begin(), data.end(), 0);
    }

    /**
     * @brief Set whether to throw an exception on next push (for testing)
     * @param should_throw Whether to throw on next push
     */
    void set_throw_on_push(bool should_throw) {
        throw_on_push = should_throw;
    }

    /**
     * @brief Sort the elements in the vector
     */
    void sort() {
        std::lock_guard<std::mutex> lock(mtx);
        std::sort(data.begin(), data.end());
    }
};

// ==================== TESTS ====================

#include <gtest/gtest.h>

class ThreadSafeVectorTest : public ::testing::Test {
protected:
    ThreadSafeVector vec;

    void SetUp() override {
        vec.clear();
    }
};

// Basic functionality tests
TEST_F(ThreadSafeVectorTest, BasicOperations) {
    EXPECT_TRUE(vec.empty());

    vec.push_back(42);
    EXPECT_EQ(1, vec.size());
    EXPECT_EQ(42, vec.at(0));

    vec.push_back(100);
    EXPECT_EQ(2, vec.size());
    EXPECT_EQ(100, vec.at(1));

    vec.clear();
    EXPECT_TRUE(vec.empty());
}

// Test exception safety
TEST_F(ThreadSafeVectorTest, ExceptionSafety) {
    vec.set_throw_on_push(true);
    EXPECT_THROW(vec.push_back(1), std::runtime_error);

    // Should still be in valid state after exception
    EXPECT_NO_THROW(vec.size());
    EXPECT_TRUE(vec.empty());

    // Should work normally after clearing the error state
    vec.set_throw_on_push(false);
    EXPECT_NO_THROW(vec.push_back(42));
    EXPECT_EQ(1, vec.size());
}

// Test concurrent access
TEST_F(ThreadSafeVectorTest, ConcurrentAccess) {
    const int num_threads = 10;
    const int num_inserts = 1000;

    auto worker = [&](int start) {
        for (int i = 0; i < num_inserts; ++i) {
            vec.push_back(start + i);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i * num_inserts);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(num_threads * num_inserts, vec.size());

    // Verify all elements were inserted correctly
    vec.sort();
    for (int i = 0; i < num_threads * num_inserts; ++i) {
        EXPECT_EQ(i, vec.at(i));
    }
}

// Performance test
TEST(ThreadSafeVectorPerformance, ConcurrentPerformance) {
    ThreadSafeVector vec;
    const int num_threads = std::thread::hardware_concurrency();
    const int num_inserts = 100000;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < num_inserts; ++j) {
                vec.push_back(j);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\nPerformance Test Results:";
    std::cout << "\nThreads: " << num_threads;
    std::cout << "\nTotal inserts: " << (num_threads * num_inserts);
    std::cout << "\nTime taken: " << duration.count() << "ms";
    std::cout << "\nThroughput: " << (num_threads * num_inserts * 1000.0 / duration.count()) << " ops/s\n";

    EXPECT_EQ(num_threads * num_inserts, vec.size());
}
};
