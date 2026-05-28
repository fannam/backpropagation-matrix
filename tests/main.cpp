#include "test_helpers.hpp"

#include <exception>
#include <iostream>
#include <vector>

namespace {

void append_tests(std::vector<TestCase>& out, const std::vector<TestCase>& tests) {
    out.insert(out.end(), tests.begin(), tests.end());
}

} // namespace

int main() {
    std::vector<TestCase> tests;
    append_tests(tests, tensor_tests());
    append_tests(tests, operator_tests());
    append_tests(tests, activation_tests());
    append_tests(tests, init_tests());
    append_tests(tests, avg_pool_tests());

    int passed = 0;
    int total = static_cast<int>(tests.size());

    for (const auto& t : tests) {
        bool ok = false;
        try {
            ok = t.fn();
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] " << t.name << ": exception: " << e.what() << "\n";
            continue;
        }

        if (ok) {
            std::cout << "[PASS] " << t.name << "\n";
            passed++;
        } else {
            std::cout << "[FAIL] " << t.name << "\n";
        }
    }

    std::cout << "\n" << passed << "/" << total << " tests passed\n";
    return (passed == total) ? 0 : 1;
}
