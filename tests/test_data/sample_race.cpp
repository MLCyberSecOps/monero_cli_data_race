/*
 * Sample test file with a data race for testing ThreadGuard.
 * This file contains a simple data race between two threads.
 */

#include <thread>
#include <iostream>

int counter = 0;

void increment() {
    for (int i = 0; i < 1000; ++i) {
        counter++;  // Data race here
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Final counter value: " << counter << std::endl;
    return 0;
}
