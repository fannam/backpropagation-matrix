#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "nn/Init.hpp"

namespace {

bool test_uniform_seed_is_deterministic() {
    auto a = Tensor::create(2, 3);
    auto b = Tensor::create(2, 3);

    std::cout << "\n[uniform_seed_is_deterministic]\n";

    uniform_(a, -0.5, 0.5, 123);
    uniform_(b, -0.5, 0.5, 123);

    print_vector(a->data, "a", 2, 3);
    print_vector(b->data, "b", 2, 3);

    std::string err;
    if (!expect_vector_close(a->data, b->data, "same seed uniform", err, 0.0)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_uniform_values_in_range() {
    auto t = Tensor::create(4, 4);
    double low = -2.0;
    double high = 3.0;

    std::cout << "\n[uniform_values_in_range]\n";

    uniform_(t, low, high, 7);
    print_vector(t->data, "data", 4, 4);

    for (double value : t->data) {
        if (value < low || value > high) {
            std::cerr << "value out of range: " << value << "\n";
            return false;
        }
    }

    return true;
}

bool test_uniform_keeps_grad_unchanged() {
    auto t = Tensor::create(1, 3);
    t->grad = {1.0, 2.0, 3.0};

    std::cout << "\n[uniform_keeps_grad_unchanged]\n";

    uniform_(t, 0.0, 1.0, 99);

    std::string err;
    if (!expect_vector_close(t->grad, {1.0, 2.0, 3.0}, "grad unchanged", err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_uniform_invalid_inputs() {
    auto t = Tensor::create(1, 1);

    std::cout << "\n[uniform_invalid_inputs]\n";

    bool ok = true;
    ok = expect_throws_runtime(
             [&]() {
                 uniform_(nullptr, 0.0, 1.0, 1);
             },
             "uniform null tensor") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 uniform_(t, 2.0, 1.0, 1);
             },
             "uniform invalid range") &&
         ok;

    return ok;
}

bool test_zeros_and_ones() {
    auto t = Tensor::create(2, 3, {9.0, 8.0, 7.0, 6.0, 5.0, 4.0});

    std::cout << "\n[zeros_and_ones]\n";

    zeros_(t);
    print_vector(t->data, "zeros", 2, 3);

    std::string err;
    if (!expect_vector_close(t->data, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, "zeros", err)) {
        std::cerr << err << "\n";
        return false;
    }

    ones_(t);
    print_vector(t->data, "ones", 2, 3);

    if (!expect_vector_close(t->data, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, "ones", err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_normal_seed_is_deterministic() {
    auto a = Tensor::create(2, 3);
    auto b = Tensor::create(2, 3);

    std::cout << "\n[normal_seed_is_deterministic]\n";

    normal_(a, 0.0, 1.0, 321);
    normal_(b, 0.0, 1.0, 321);

    print_vector(a->data, "a", 2, 3);
    print_vector(b->data, "b", 2, 3);

    std::string err;
    if (!expect_vector_close(a->data, b->data, "same seed normal", err, 0.0)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_normal_zero_stddev() {
    auto t = Tensor::create(1, 4);

    std::cout << "\n[normal_zero_stddev]\n";

    normal_(t, 2.5, 0.0, 1);

    std::string err;
    if (!expect_vector_close(t->data, {2.5, 2.5, 2.5, 2.5}, "zero stddev normal", err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_xavier_uniform_range() {
    auto t = Tensor::create(3, 5);
    double bound = std::sqrt(6.0 / (3.0 + 5.0));

    std::cout << "\n[xavier_uniform_range]\n";

    xavier_uniform_(t, 11);
    print_vector(t->data, "data", 3, 5);

    for (double value : t->data) {
        if (value < -bound || value > bound) {
            std::cerr << "xavier value out of range: " << value << "\n";
            return false;
        }
    }

    return true;
}

bool test_kaiming_uniform_range() {
    auto t = Tensor::create(3, 5);
    double bound = std::sqrt(6.0 / 3.0);

    std::cout << "\n[kaiming_uniform_range]\n";

    kaiming_uniform_(t, 13);
    print_vector(t->data, "data", 3, 5);

    for (double value : t->data) {
        if (value < -bound || value > bound) {
            std::cerr << "kaiming value out of range: " << value << "\n";
            return false;
        }
    }

    return true;
}

bool test_init_invalid_inputs() {
    auto t = Tensor::create(1, 1);

    std::cout << "\n[init_invalid_inputs]\n";

    bool ok = true;
    ok = expect_throws_runtime(
             [&]() {
                 zeros_(nullptr);
             },
             "zeros null tensor") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 ones_(nullptr);
             },
             "ones null tensor") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 normal_(nullptr, 0.0, 1.0, 1);
             },
             "normal null tensor") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 normal_(t, 0.0, -1.0, 1);
             },
             "normal negative stddev") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 xavier_uniform_(nullptr, 1);
             },
             "xavier null tensor") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 kaiming_uniform_(nullptr, 1);
             },
             "kaiming null tensor") &&
         ok;

    return ok;
}

} // namespace

std::vector<TestCase> init_tests() {
    return {
        {"uniform_seed_is_deterministic", test_uniform_seed_is_deterministic},
        {"uniform_values_in_range", test_uniform_values_in_range},
        {"uniform_keeps_grad_unchanged", test_uniform_keeps_grad_unchanged},
        {"uniform_invalid_inputs", test_uniform_invalid_inputs},
        {"zeros_and_ones", test_zeros_and_ones},
        {"normal_seed_is_deterministic", test_normal_seed_is_deterministic},
        {"normal_zero_stddev", test_normal_zero_stddev},
        {"xavier_uniform_range", test_xavier_uniform_range},
        {"kaiming_uniform_range", test_kaiming_uniform_range},
        {"init_invalid_inputs", test_init_invalid_inputs},
    };
}
