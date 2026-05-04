#include "test_helpers.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "operators/BinaryOps.hpp"
#include "operators/MatrixOps.hpp"
#include "operators/UnaryOps.hpp"

namespace {

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

bool test_operator_shape_errors() {
    auto a = make_tensor(1, 2, {1.0, 2.0});
    auto b = make_tensor(2, 1, {3.0, 4.0});
    auto m1 = make_tensor(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto m2 = make_tensor(2, 2, {1.0, 2.0, 3.0, 4.0});

    std::cout << "\n[operator_shape_errors]\n";

    bool ok = true;
    ok = expect_throws_runtime(
             [&]() {
                 add(a, b);
             },
             "add shape mismatch") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 hadamard_mul(a, b);
             },
             "hadamard_mul shape mismatch") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 hadamard_div(a, b);
             },
             "hadamard_div shape mismatch") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 matmul(m1, m2);
             },
             "matmul shape mismatch") &&
         ok;
    return ok;
}

bool test_divide_by_zero_errors() {
    auto a = make_tensor(1, 2, {2.0, 4.0});
    auto b = make_tensor(1, 2, {1.0, 0.0});
    auto zero = make_tensor(1, 1, {0.0});

    std::cout << "\n[divide_by_zero_errors]\n";

    bool ok = true;
    ok = expect_throws_runtime(
             [&]() {
                 hadamard_div(a, b);
             },
             "hadamard_div divide by zero") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 scalar_div_tensor(1.0, zero);
             },
             "scalar_div_tensor divide by zero") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 auto result = a / 0.0;
                 (void)result;
             },
             "tensor divide by scalar zero") &&
         ok;
    return ok;
}

bool test_log_op_domain_error() {
    auto a = make_tensor(1, 3, {1.0, 0.0, -1.0});

    std::cout << "\n[log_op_domain_error]\n";
    return expect_throws_runtime(
        [&]() {
            log_op(a);
        },
        "log_op non-positive input");
}

bool test_hadamard_mul_numerical_gradient() {
    double x = 2.0;
    auto a = make_tensor(1, 1, {x});
    auto b = make_tensor(1, 1, {3.0});
    auto c = hadamard_mul(a, b);
    c->backward();

    double numerical = finite_difference(
        [&]() {
            auto out = hadamard_mul(make_tensor(1, 1, {x}), make_tensor(1, 1, {3.0}));
            return out->data[0];
        },
        x);

    std::cout << "\n[hadamard_mul_numerical_gradient]\n";
    std::string err;
    if (!expect_scalar_close(a->grad[0], numerical, "hadamard_mul numerical grad", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_hadamard_div_numerical_gradient() {
    double numerator = 6.0;
    double denominator = 3.0;
    auto a = make_tensor(1, 1, {numerator});
    auto b = make_tensor(1, 1, {denominator});
    auto c = hadamard_div(a, b);
    c->backward();

    double numerical_a = finite_difference(
        [&]() {
            auto out = hadamard_div(make_tensor(1, 1, {numerator}), make_tensor(1, 1, {denominator}));
            return out->data[0];
        },
        numerator);
    double numerical_b = finite_difference(
        [&]() {
            auto out = hadamard_div(make_tensor(1, 1, {numerator}), make_tensor(1, 1, {denominator}));
            return out->data[0];
        },
        denominator);

    std::cout << "\n[hadamard_div_numerical_gradient]\n";
    std::string err;
    if (!expect_scalar_close(a->grad[0], numerical_a, "hadamard_div numerator grad", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }
    if (!expect_scalar_close(b->grad[0], numerical_b, "hadamard_div denominator grad", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

bool test_matmul_numerical_gradient() {
    double x = 2.0;
    auto a = make_tensor(1, 2, {x, 4.0});
    auto b = make_tensor(2, 1, {3.0, 5.0});
    auto c = matmul(a, b);
    c->backward();

    double numerical = finite_difference(
        [&]() {
            auto out = matmul(make_tensor(1, 2, {x, 4.0}), make_tensor(2, 1, {3.0, 5.0}));
            return out->data[0];
        },
        x);

    std::cout << "\n[matmul_numerical_gradient]\n";
    std::string err;
    if (!expect_scalar_close(a->grad[0], numerical, "matmul numerical grad", err, 1e-6)) {
        std::cerr << err << "\n";
        return false;
    }
    return true;
}

} // namespace

std::vector<TestCase> operator_tests() {
    return {
        {"add", test_add},
        {"hadamard_mul", test_hadamard_mul},
        {"hadamard_div", test_hadamard_div},
        {"negate", test_negate},
        {"exp_log_chain", test_exp_log_chain},
        {"scalar_ops", test_scalar_ops},
        {"transpose", test_transpose},
        {"matmul", test_matmul},
        {"operator_shape_errors", test_operator_shape_errors},
        {"divide_by_zero_errors", test_divide_by_zero_errors},
        {"log_op_domain_error", test_log_op_domain_error},
        {"hadamard_mul_numerical_gradient", test_hadamard_mul_numerical_gradient},
        {"hadamard_div_numerical_gradient", test_hadamard_div_numerical_gradient},
        {"matmul_numerical_gradient", test_matmul_numerical_gradient},
    };
}
