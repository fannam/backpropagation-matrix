#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/Tensor.hpp"

struct TestCase {
    const char* name;
    bool (*fn)();
};

inline std::shared_ptr<Tensor> make_tensor(int rows, int cols, const std::vector<double>& values) {
    return Tensor::create(rows, cols, values);
}

inline bool close_enough(double a, double b, double eps = 1e-9) {
    double diff = std::fabs(a - b);
    double scale = 1.0 + std::max(std::fabs(a), std::fabs(b));
    return diff <= eps * scale;
}

inline bool expect_vector_close(const std::vector<double>& actual,
                                const std::vector<double>& expected,
                                const std::string& label,
                                std::string& err,
                                double eps = 1e-9) {
    if (actual.size() != expected.size()) {
        err = label + ": size mismatch (" + std::to_string(actual.size()) + " vs " +
              std::to_string(expected.size()) + ")";
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual[i], expected[i], eps)) {
            err = label + ": index " + std::to_string(i) + " got " + std::to_string(actual[i]) +
                  " expected " + std::to_string(expected[i]);
            return false;
        }
    }
    return true;
}

inline bool expect_scalar_close(double actual,
                                double expected,
                                const std::string& label,
                                std::string& err,
                                double eps = 1e-9) {
    if (!close_enough(actual, expected, eps)) {
        err = label + ": got " + std::to_string(actual) + " expected " + std::to_string(expected);
        return false;
    }
    return true;
}

inline bool expect_throws_runtime(const std::function<void()>& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }

    std::cerr << label << ": expected std::runtime_error\n";
    return false;
}

inline double finite_difference(const std::function<double()>& fn, double& x, double eps = 1e-6) {
    double original = x;
    x = original + eps;
    double plus = fn();
    x = original - eps;
    double minus = fn();
    x = original;
    return (plus - minus) / (2.0 * eps);
}

inline void print_vector(const std::vector<double>& v,
                         const std::string& label,
                         int rows = 1,
                         int cols = -1) {
    std::cout << label << ":\n";
    if (cols <= 0) {
        for (size_t i = 0; i < v.size(); ++i) {
            std::cout << v[i] << (i + 1 == v.size() ? "\n" : " ");
        }
        return;
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            size_t idx = static_cast<size_t>(r * cols + c);
            std::cout << v[idx] << (c + 1 == cols ? "\n" : " ");
        }
    }
}

std::vector<TestCase> operator_tests();
std::vector<TestCase> activation_tests();
std::vector<TestCase> tensor_tests();
std::vector<TestCase> init_tests();
std::vector<TestCase> avg_pool_tests();

#endif
