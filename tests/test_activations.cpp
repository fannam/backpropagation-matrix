#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "activations/Activations.hpp"

namespace {

std::vector<double> expected_log_softmax_vector(const std::vector<double>& values) {
    double max_val = *std::max_element(values.begin(), values.end());
    double sum_exp = 0.0;
    for (double x : values) {
        sum_exp += std::exp(x - max_val);
    }

    double log_sum_exp = std::log(sum_exp);
    std::vector<double> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected[i] = values[i] - max_val - log_sum_exp;
    }
    return expected;
}

std::vector<double> expected_log_softmax_rows(const std::vector<double>& values, int rows, int cols) {
    std::vector<double> expected(values.size());
    for (int r = 0; r < rows; ++r) {
        size_t row_start = static_cast<size_t>(r * cols);
        double max_val = values[row_start];
        for (int c = 1; c < cols; ++c) {
            max_val = std::max(max_val, values[row_start + static_cast<size_t>(c)]);
        }

        double sum_exp = 0.0;
        for (int c = 0; c < cols; ++c) {
            sum_exp += std::exp(values[row_start + static_cast<size_t>(c)] - max_val);
        }

        double log_sum_exp = std::log(sum_exp);
        for (int c = 0; c < cols; ++c) {
            size_t idx = row_start + static_cast<size_t>(c);
            expected[idx] = values[idx] - max_val - log_sum_exp;
        }
    }
    return expected;
}

std::vector<double> expected_log_softmax_grad_vector(const std::vector<double>& log_probs,
                                                     const std::vector<double>& seed) {
    double sum_seed = 0.0;
    for (double g : seed) {
        sum_seed += g;
    }

    std::vector<double> expected(seed.size());
    for (size_t i = 0; i < seed.size(); ++i) {
        expected[i] = seed[i] - std::exp(log_probs[i]) * sum_seed;
    }
    return expected;
}

std::vector<double> expected_log_softmax_grad_rows(const std::vector<double>& log_probs,
                                                   const std::vector<double>& seed,
                                                   int rows,
                                                   int cols) {
    std::vector<double> expected(seed.size());
    for (int r = 0; r < rows; ++r) {
        size_t row_start = static_cast<size_t>(r * cols);
        double sum_seed = 0.0;
        for (int c = 0; c < cols; ++c) {
            sum_seed += seed[row_start + static_cast<size_t>(c)];
        }
        for (int c = 0; c < cols; ++c) {
            size_t idx = row_start + static_cast<size_t>(c);
            expected[idx] = seed[idx] - std::exp(log_probs[idx]) * sum_seed;
        }
    }
    return expected;
}

double weighted_sum(const std::vector<double>& values, const std::vector<double>& weights) {
    double total = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        total += values[i] * weights[i];
    }
    return total;
}

bool test_relu_activation() {
    auto a = make_tensor(1, 3, {-1.0, 0.0, 2.0});
    auto c = relu(a);

    std::cout << "\n[relu]\n";
    print_vector(a->data, "a", 1, 3);
    print_vector(c->data, "relu(a)", 1, 3);

    std::string err;
    if (!expect_vector_close(c->data, {0.0, 0.0, 2.0}, "relu data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 3);
    if (!expect_vector_close(a->grad, {0.0, 0.0, 1.0}, "relu grad", err)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_leaky_relu_activation() {
    auto a = make_tensor(1, 3, {-2.0, 0.0, 3.0});
    auto c = leaky_relu(a, 0.1f);

    std::cout << "\n[leaky_relu]\n";
    print_vector(a->data, "a", 1, 3);
    print_vector(c->data, "leaky_relu(a)", 1, 3);

    std::string err;
    if (!expect_vector_close(c->data, {-0.2, 0.0, 3.0}, "leaky_relu data", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 3);
    if (!expect_vector_close(a->grad, {0.1, 0.1, 1.0}, "leaky_relu grad", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_silu_activation() {
    auto a = make_tensor(1, 2, {0.0, 1.0});
    auto c = silu(a);

    std::cout << "\n[silu]\n";
    print_vector(a->data, "a", 1, 2);
    print_vector(c->data, "silu(a)", 1, 2);

    std::string err;
    if (!expect_vector_close(c->data, {0.0, 0.7310585786300049}, "silu data", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 2);
    if (!expect_vector_close(a->grad, {0.5, 0.9276705118714867}, "silu grad", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_gelu_exact_activation() {
    auto a = make_tensor(1, 2, {0.0, 1.0});
    auto c = gelu(a, "none");

    std::cout << "\n[gelu_exact]\n";
    print_vector(a->data, "a", 1, 2);
    print_vector(c->data, "gelu(a, none)", 1, 2);

    std::string err;
    if (!expect_vector_close(c->data, {0.0, 0.8413447460685429}, "gelu exact data", err, 1e-5)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 2);
    if (!expect_vector_close(a->grad, {0.5, 1.0833154705876864}, "gelu exact grad", err, 1e-5)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_gelu_tanh_activation() {
    auto a = make_tensor(1, 2, {0.0, 1.0});
    auto c = gelu(a, "tanh");

    std::cout << "\n[gelu_tanh]\n";
    print_vector(a->data, "a", 1, 2);
    print_vector(c->data, "gelu(a, tanh)", 1, 2);

    std::string err;
    if (!expect_vector_close(c->data, {0.0, 0.8411919906082768}, "gelu tanh data", err, 1e-5)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 2);
    if (!expect_vector_close(a->grad, {0.5, 1.0829640838457826}, "gelu tanh grad", err, 1e-5)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_sigmoid_activation() {
    auto a = make_tensor(1, 2, {0.0, 1.0});
    auto c = sigmoid(a);

    std::cout << "\n[sigmoid]\n";
    print_vector(a->data, "a", 1, 2);
    print_vector(c->data, "sigmoid(a)", 1, 2);

    std::string err;
    if (!expect_vector_close(c->data, {0.5, 0.7310585786300049}, "sigmoid data", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 2);
    if (!expect_vector_close(a->grad, {0.25, 0.19661193324148185}, "sigmoid grad", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_tanh_activation() {
    auto a = make_tensor(1, 2, {0.0, 0.5});
    auto c = tanh(a);

    std::cout << "\n[tanh]\n";
    print_vector(a->data, "a", 1, 2);
    print_vector(c->data, "tanh(a)", 1, 2);

    std::string err;
    if (!expect_vector_close(c->data, {0.0, 0.46211715726000974}, "tanh data", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 2);
    if (!expect_vector_close(a->grad, {1.0, 0.7864477329659274}, "tanh grad", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_log_softmax_row_vector() {
    std::vector<double> values = {1.0, 2.0, 3.0};
    auto a = make_tensor(1, 3, values);
    auto c = log_softmax(a);

    std::cout << "\n[log_softmax_row_vector]\n";
    print_vector(a->data, "a", 1, 3);
    print_vector(c->data, "log_softmax(a)", 1, 3);

    std::vector<double> expected_data = expected_log_softmax_vector(values);

    std::string err;
    if (!expect_vector_close(c->data, expected_data, "log_softmax data", err, 1e-9)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 3);

    std::vector<double> expected_grad = expected_log_softmax_grad_vector(expected_data, {1.0, 1.0, 1.0});
    if (!expect_vector_close(a->grad, expected_grad, "log_softmax grad", err, 1e-9)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_log_softmax_column_vector() {
    std::vector<double> values = {1.0, 2.0, 3.0};
    auto a = make_tensor(3, 1, values);
    auto c = log_softmax(a);

    std::cout << "\n[log_softmax_column_vector]\n";
    print_vector(a->data, "a", 3, 1);
    print_vector(c->data, "log_softmax(a)", 3, 1);

    std::string err;
    if (!expect_vector_close(c->data, expected_log_softmax_vector(values), "log_softmax column data", err, 1e-9)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_log_softmax_matrix_rows() {
    std::vector<double> values = {1.0, 2.0, 3.0, 1.0, 3.0, 5.0};
    auto a = make_tensor(2, 3, values);
    auto c = log_softmax(a);

    std::cout << "\n[log_softmax_matrix_rows]\n";
    print_vector(a->data, "a", 2, 3);
    print_vector(c->data, "log_softmax(a)", 2, 3);

    std::string err;
    if (!expect_vector_close(c->data, expected_log_softmax_rows(values, 2, 3), "log_softmax matrix data", err, 1e-9)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_log_softmax_seed_backward() {
    std::vector<double> values = {1.0, 2.0, 3.0, 1.0, 3.0, 5.0};
    std::vector<double> seed = {1.0, 0.0, -2.0, 0.5, 1.5, -1.0};
    auto a = make_tensor(2, 3, values);
    auto c = log_softmax(a);

    std::cout << "\n[log_softmax_seed_backward]\n";
    c->backward(seed);
    print_vector(a->grad, "grad a", 2, 3);

    std::vector<double> expected_data = expected_log_softmax_rows(values, 2, 3);
    std::vector<double> expected_grad = expected_log_softmax_grad_rows(expected_data, seed, 2, 3);

    std::string err;
    if (!expect_vector_close(a->grad, expected_grad, "log_softmax seeded grad", err, 1e-9)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_gelu_invalid_approximation() {
    auto a = make_tensor(1, 1, {1.0});

    std::cout << "\n[gelu_invalid_approximation]\n";
    return expect_throws_runtime(
        [&]() {
            gelu(a, "fast");
        },
        "gelu invalid approximation");
}

bool test_sigmoid_numerical_gradient() {
    double x = 0.7;
    auto a = make_tensor(1, 1, {x});
    auto c = sigmoid(a);
    c->backward();

    double numerical = finite_difference(
        [&]() {
            return sigmoid(make_tensor(1, 1, {x}))->data[0];
        },
        x);

    std::cout << "\n[sigmoid_numerical_gradient]\n";
    std::string err;
    if (!expect_scalar_close(a->grad[0], numerical, "sigmoid numerical grad", err, 1e-5)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_tanh_numerical_gradient() {
    double x = 0.7;
    auto a = make_tensor(1, 1, {x});
    auto c = tanh(a);
    c->backward();

    double numerical = finite_difference(
        [&]() {
            return tanh(make_tensor(1, 1, {x}))->data[0];
        },
        x);

    std::cout << "\n[tanh_numerical_gradient]\n";
    std::string err;
    if (!expect_scalar_close(a->grad[0], numerical, "tanh numerical grad", err, 1e-5)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_gelu_numerical_gradient() {
    double x = 0.7;
    auto a = make_tensor(1, 1, {x});
    auto c = gelu(a, "none");
    c->backward();

    double numerical = finite_difference(
        [&]() {
            return gelu(make_tensor(1, 1, {x}), "none")->data[0];
        },
        x,
        1e-5);

    std::cout << "\n[gelu_numerical_gradient]\n";
    std::string err;
    if (!expect_scalar_close(a->grad[0], numerical, "gelu numerical grad", err, 1e-4)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_log_softmax_numerical_gradient() {
    double x = 1.0;
    std::vector<double> seed = {0.5, -1.25, 2.0};
    auto a = make_tensor(1, 3, {x, 2.0, -0.5});
    auto c = log_softmax(a);
    c->backward(seed);

    double numerical = finite_difference(
        [&]() {
            auto out = log_softmax(make_tensor(1, 3, {x, 2.0, -0.5}));
            return weighted_sum(out->data, seed);
        },
        x);

    std::cout << "\n[log_softmax_numerical_gradient]\n";
    std::string err;
    if (!expect_scalar_close(a->grad[0], numerical, "log_softmax numerical grad", err, 1e-5)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

} // namespace

std::vector<TestCase> activation_tests() {
    return {
        {"relu", test_relu_activation},
        {"leaky_relu", test_leaky_relu_activation},
        {"silu", test_silu_activation},
        {"gelu_exact", test_gelu_exact_activation},
        {"gelu_tanh", test_gelu_tanh_activation},
        {"sigmoid", test_sigmoid_activation},
        {"tanh", test_tanh_activation},
        {"log_softmax_row_vector", test_log_softmax_row_vector},
        {"log_softmax_column_vector", test_log_softmax_column_vector},
        {"log_softmax_matrix_rows", test_log_softmax_matrix_rows},
        {"log_softmax_seed_backward", test_log_softmax_seed_backward},
        {"gelu_invalid_approximation", test_gelu_invalid_approximation},
        {"sigmoid_numerical_gradient", test_sigmoid_numerical_gradient},
        {"tanh_numerical_gradient", test_tanh_numerical_gradient},
        {"gelu_numerical_gradient", test_gelu_numerical_gradient},
        {"log_softmax_numerical_gradient", test_log_softmax_numerical_gradient},
    };
}
