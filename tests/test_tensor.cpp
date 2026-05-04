#include "test_helpers.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "operators/BinaryOps.hpp"

namespace {

bool expect_runtime_error(const std::function<void()>& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }

    std::cerr << label << ": expected std::runtime_error\n";
    return false;
}

bool test_create_from_values() {
    auto t = Tensor::create(2, 2, {1.0, 2.0, 3.0, 4.0}, "values");

    std::cout << "\n[tensor_create_from_values]\n";
    print_vector(t->data, "data", 2, 2);

    std::string err;
    if (t->rows != 2 || t->cols != 2) {
        std::cerr << "shape mismatch\n";
        return false;
    }
    if (t->label != "values") {
        std::cerr << "label mismatch\n";
        return false;
    }
    if (!expect_vector_close(t->data, {1.0, 2.0, 3.0, 4.0}, "create data", err)) {
        std::cerr << err << "\n";
        return false;
    }
    if (!expect_vector_close(t->grad, {0.0, 0.0, 0.0, 0.0}, "create grad", err)) {
        std::cerr << err << "\n";
        return false;
    }

    auto scalar = Tensor::create(1, 1, {0.1}, "scalar");
    if (!close_enough(scalar->at(0, 0), 0.1) || scalar->label != "scalar") {
        std::cerr << "single-value initializer create failed\n";
        return false;
    }

    return true;
}

bool test_create_from_values_size_mismatch() {
    std::cout << "\n[tensor_create_from_values_size_mismatch]\n";
    return expect_runtime_error(
        []() {
            Tensor::create(2, 2, {1.0, 2.0, 3.0});
        },
        "create values size mismatch");
}

bool test_invalid_shape() {
    std::cout << "\n[tensor_invalid_shape]\n";
    bool ok = true;
    ok = expect_runtime_error(
             []() {
                 Tensor::create(0, 2);
             },
             "zero rows") &&
         ok;
    ok = expect_runtime_error(
             []() {
                 Tensor::create(2, 0);
             },
             "zero cols") &&
         ok;
    ok = expect_runtime_error(
             []() {
                 Tensor::create(-1, 2);
             },
             "negative rows") &&
         ok;
    return ok;
}

bool test_const_accessors() {
    auto t = Tensor::create(1, 2, {3.0, 4.0});
    t->grad_at(0, 1) = 7.0;

    const Tensor& ref = *t;

    std::cout << "\n[tensor_const_accessors]\n";
    std::cout << "at(0, 1): " << ref.at(0, 1) << "\n";
    std::cout << "grad_at(0, 1): " << ref.grad_at(0, 1) << "\n";

    if (!close_enough(ref.at(0, 1), 4.0)) {
        std::cerr << "const at returned wrong value\n";
        return false;
    }
    if (!close_enough(ref.grad_at(0, 1), 7.0)) {
        std::cerr << "const grad_at returned wrong value\n";
        return false;
    }

    return true;
}

bool test_bounds_checking() {
    auto t = Tensor::create(2, 2, {1.0, 2.0, 3.0, 4.0});
    const Tensor& ref = *t;

    std::cout << "\n[tensor_bounds_checking]\n";

    t->at(1, 1) = 5.0;
    t->grad_at(1, 1) = 6.0;
    if (!close_enough(ref.at(1, 1), 5.0) || !close_enough(ref.grad_at(1, 1), 6.0)) {
        std::cerr << "valid bounds access failed\n";
        return false;
    }

    bool ok = true;
    ok = expect_runtime_error(
             [&]() {
                 t->at(-1, 0);
             },
             "negative row") &&
         ok;
    ok = expect_runtime_error(
             [&]() {
                 t->at(0, 2);
             },
             "col out of bounds") &&
         ok;
    ok = expect_runtime_error(
             [&]() {
                 t->grad_at(2, 0);
             },
             "grad row out of bounds") &&
         ok;
    ok = expect_runtime_error(
             [&]() {
                 ref.at(2, 0);
             },
             "const row out of bounds") &&
         ok;

    return ok;
}

bool test_zero_grad() {
    auto t = Tensor::create(2, 2, {1.0, 2.0, 3.0, 4.0});
    t->grad = {0.5, -1.0, 2.0, 3.5};
    t->grad_at(1, 1) = -4.0;

    std::cout << "\n[tensor_zero_grad]\n";
    print_vector(t->grad, "grad before", 2, 2);

    t->zero_grad();
    print_vector(t->grad, "grad after", 2, 2);

    std::string err;
    if (!expect_vector_close(t->grad, {0.0, 0.0, 0.0, 0.0}, "zero_grad", err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_backward_seed_size_mismatch() {
    auto t = Tensor::create(1, 2, {1.0, 2.0});

    std::cout << "\n[tensor_backward_seed_size_mismatch]\n";
    return expect_runtime_error(
        [&]() {
            t->backward({1.0});
        },
        "backward seed size mismatch");
}

bool test_backward_seed_grad_add() {
    auto a = Tensor::create(1, 2, {2.0, 4.0});
    auto b = Tensor::create(1, 2, {10.0, 20.0});
    auto c = add(a, b);

    std::cout << "\n[tensor_backward_seed_grad_add]\n";

    c->backward({3.0, -2.0});

    std::string err;
    if (!expect_vector_close(c->grad, {3.0, -2.0}, "seed output grad", err)) {
        std::cerr << err << "\n";
        return false;
    }
    if (!expect_vector_close(a->grad, {3.0, -2.0}, "seed grad a", err)) {
        std::cerr << err << "\n";
        return false;
    }
    if (!expect_vector_close(b->grad, {3.0, -2.0}, "seed grad b", err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

} // namespace

std::vector<TestCase> tensor_tests() {
    return {
        {"tensor_create_from_values", test_create_from_values},
        {"tensor_create_from_values_size_mismatch", test_create_from_values_size_mismatch},
        {"tensor_invalid_shape", test_invalid_shape},
        {"tensor_const_accessors", test_const_accessors},
        {"tensor_bounds_checking", test_bounds_checking},
        {"tensor_zero_grad", test_zero_grad},
        {"tensor_backward_seed_size_mismatch", test_backward_seed_size_mismatch},
        {"tensor_backward_seed_grad_add", test_backward_seed_grad_add},
    };
}
