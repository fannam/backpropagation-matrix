#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/Tensor.hpp"
#include "operators/BinaryOps.hpp"
#include "operators/MatrixOps.hpp"
#include "operators/UnaryOps.hpp"

namespace {

std::shared_ptr<Tensor> make_tensor(int rows, int cols, const std::vector<double>& values) {
    auto t = Tensor::create(rows, cols);
    if (static_cast<int>(values.size()) != rows * cols) {
        throw std::runtime_error("make_tensor: size mismatch");
    }
    for (int i = 0; i < rows * cols; ++i) {
        t->data[i] = values[i];
    }
    return t;
}

bool close_enough(double a, double b, double eps = 1e-9) {
    double diff = std::fabs(a - b);
    double scale = 1.0 + std::max(std::fabs(a), std::fabs(b));
    return diff <= eps * scale;
}

bool expect_vector_close(const std::vector<double>& actual,
                         const std::vector<double>& expected,
                         const std::string& label,
                         std::string& err) {
    if (actual.size() != expected.size()) {
        err = label + ": size mismatch (" + std::to_string(actual.size()) + " vs " +
              std::to_string(expected.size()) + ")";
        return false;
    }
    for (long unsigned int i = 0; i < actual.size(); ++i) {
        if (!close_enough(actual[i], expected[i])) {
            err = label + ": index " + std::to_string(i) + " got " + std::to_string(actual[i]) +
                  " expected " + std::to_string(expected[i]);
            return false;
        }
    }
    return true;
}

void print_vector(const std::vector<double>& v, const std::string& label, int rows = 1, int cols = -1) {
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

bool test_add() {
    auto a = make_tensor(2, 2, {1, 2, 3, 4});
    auto b = make_tensor(2, 2, {5, 6, 7, 8});
    auto c = add(a, b);

    std::cout << "\n[add]\n";
    print_vector(a->data, "a", 2, 2);
    print_vector(b->data, "b", 2, 2);
    print_vector(c->data, "a + b", 2, 2);

    std::string err;
    if (!expect_vector_close(c->data, {6, 8, 10, 12}, "add data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 2, 2);
    print_vector(b->grad, "grad b", 2, 2);
    if (!expect_vector_close(a->grad, {1, 1, 1, 1}, "add grad a", err)) {
        std::cerr << err << "\n";
        return false;
    }
    if (!expect_vector_close(b->grad, {1, 1, 1, 1}, "add grad b", err)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_hadamard_mul() {
    auto a = make_tensor(2, 2, {1, 2, 3, 4});
    auto b = make_tensor(2, 2, {5, 6, 7, 8});
    auto c = hadamard_mul(a, b);

    std::cout << "\n[hadamard_mul]\n";
    print_vector(a->data, "a", 2, 2);
    print_vector(b->data, "b", 2, 2);
    print_vector(c->data, "a * b", 2, 2);

    std::string err;
    if (!expect_vector_close(c->data, {5, 12, 21, 32}, "h_mul data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 2, 2);
    print_vector(b->grad, "grad b", 2, 2);
    if (!expect_vector_close(a->grad, b->data, "h_mul grad a", err)) {
        std::cerr << err << "\n";
        return false;
    }
    if (!expect_vector_close(b->grad, a->data, "h_mul grad b", err)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_hadamard_div() {
    auto a = make_tensor(2, 2, {2, 4, 6, 8});
    auto b = make_tensor(2, 2, {1, 2, 3, 4});
    auto c = hadamard_div(a, b);

    std::cout << "\n[hadamard_div]\n";
    print_vector(a->data, "a", 2, 2);
    print_vector(b->data, "b", 2, 2);
    print_vector(c->data, "a / b", 2, 2);

    std::string err;
    if (!expect_vector_close(c->data, {2, 2, 2, 2}, "h_div data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 2, 2);
    print_vector(b->grad, "grad b", 2, 2);
    if (!expect_vector_close(a->grad, {1, 0.5, 1.0 / 3.0, 0.25}, "h_div grad a", err)) {
        std::cerr << err << "\n";
        return false;
    }
    if (!expect_vector_close(b->grad, {-2, -1, -2.0 / 3.0, -0.5}, "h_div grad b", err)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_negate() {
    auto a = make_tensor(1, 2, {3, -4});
    auto c = negate(a);

    std::cout << "\n[negate]\n";
    print_vector(a->data, "a", 1, 2);
    print_vector(c->data, "-a", 1, 2);

    std::string err;
    if (!expect_vector_close(c->data, {-3, 4}, "neg data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 2);
    if (!expect_vector_close(a->grad, {-1, -1}, "neg grad", err)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_exp_log_chain() {
    auto a = make_tensor(1, 2, {-1, 0.5});
    auto b = exp_op(a);
    auto c = log_op(b);

    std::cout << "\n[exp_log_chain]\n";
    print_vector(a->data, "a", 1, 2);
    print_vector(b->data, "exp(a)", 1, 2);
    print_vector(c->data, "log(exp(a))", 1, 2);

    std::string err;
    if (!expect_vector_close(c->data, a->data, "log(exp(a)) data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 1, 2);
    if (!expect_vector_close(a->grad, {1, 1}, "log(exp(a)) grad", err)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_scalar_ops() {
    auto a = make_tensor(1, 2, {2, 4});
    auto c = tensor_mul_scalar(a, 3.0);

    std::cout << "\n[scalar_ops]\n";
    print_vector(a->data, "a", 1, 2);
    print_vector(c->data, "a * 3", 1, 2);

    std::string err;
    if (!expect_vector_close(c->data, {6, 12}, "mul scalar data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a (from a*3)", 1, 2);
    if (!expect_vector_close(a->grad, {3, 3}, "mul scalar grad", err)) {
        std::cerr << err << "\n";
        return false;
    }

    auto a2 = make_tensor(1, 2, {2, 4});
    auto d = scalar_div_tensor(8.0, a2);
    d->backward();
    print_vector(a2->data, "a2", 1, 2);
    print_vector(d->data, "8 / a2", 1, 2);
    print_vector(a2->grad, "grad a2 (from 8/a2)", 1, 2);
    if (!expect_vector_close(a2->grad, {-2, -0.5}, "scalar_div grad", err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_transpose() {
    auto a = make_tensor(2, 3, {1, 2, 3, 4, 5, 6});
    auto b = transpose(a);

    std::cout << "\n[transpose]\n";
    print_vector(a->data, "a", 2, 3);
    print_vector(b->data, "a^T", 3, 2);

    std::string err;
    if (!expect_vector_close(b->data, {1, 4, 2, 5, 3, 6}, "transpose data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    b->backward();
    print_vector(a->grad, "grad a", 2, 3);
    if (!expect_vector_close(a->grad, {1, 1, 1, 1, 1, 1}, "transpose grad", err)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_matmul() {
    auto a = make_tensor(2, 3, {1, 2, 3, 4, 5, 6});
    auto b = make_tensor(3, 2, {7, 8, 9, 10, 11, 12});
    auto c = matmul(a, b);

    std::cout << "\n[matmul]\n";
    print_vector(a->data, "a", 2, 3);
    print_vector(b->data, "b", 3, 2);
    print_vector(c->data, "a @ b", 2, 2);

    std::string err;
    if (!expect_vector_close(c->data, {58, 64, 139, 154}, "matmul data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    c->backward();
    print_vector(a->grad, "grad a", 2, 3);
    print_vector(b->grad, "grad b", 3, 2);
    if (!expect_vector_close(a->grad, {15, 19, 23, 15, 19, 23}, "matmul grad a", err)) {
        std::cerr << err << "\n";
        return false;
    }
    if (!expect_vector_close(b->grad, {5, 5, 7, 7, 9, 9}, "matmul grad b", err)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    struct TestCase {
        const char* name;
        bool (*fn)();
    } tests[] = {
        {"add", test_add},
        {"hadamard_mul", test_hadamard_mul},
        {"hadamard_div", test_hadamard_div},
        {"negate", test_negate},
        {"exp_log_chain", test_exp_log_chain},
        {"scalar_ops", test_scalar_ops},
        {"transpose", test_transpose},
        {"matmul", test_matmul},
    };

    int passed = 0;
    int total = static_cast<int>(sizeof(tests) / sizeof(tests[0]));

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
