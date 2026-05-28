#include "test_helpers.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "nn/AvgPool2d.hpp"

namespace {

bool test_avgpool_scalar_ctor_sets_fields() {
    std::cout << "\n[avgpool_scalar_ctor_sets_fields]\n";

    AvgPool2d pool(3, 2, 1, true);

    if (pool.kernel_rows != 3 || pool.kernel_cols != 3 || pool.stride_rows != 2 ||
        pool.stride_cols != 2 || pool.padding_rows != 1 || pool.padding_cols != 1 ||
        !pool.count_include_pad) {
        std::cerr << "scalar ctor did not set expected fields\n";
        return false;
    }

    return true;
}

bool test_avgpool_pair_ctor_sets_fields() {
    std::cout << "\n[avgpool_pair_ctor_sets_fields]\n";

    AvgPool2d pool({2, 3}, {1, 2}, {0, 1}, false);

    if (pool.kernel_rows != 2 || pool.kernel_cols != 3 || pool.stride_rows != 1 ||
        pool.stride_cols != 2 || pool.padding_rows != 0 || pool.padding_cols != 1 ||
        pool.count_include_pad) {
        std::cerr << "pair ctor did not set expected fields\n";
        return false;
    }

    return true;
}

bool test_avgpool_forward_without_padding() {
    auto x = make_tensor(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    AvgPool2d pool(2, 1, 0, false);
    auto out = pool.forward(x);

    std::cout << "\n[avgpool_forward_without_padding]\n";
    print_vector(x->data, "x", 3, 3);
    print_vector(out->data, "avg_pool2d(x)", out->rows, out->cols);

    std::string err;
    if (!expect_vector_close(out->data, {3, 4, 6, 7}, "avg pool no pad data", err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_avgpool_forward_count_include_pad() {
    auto x = make_tensor(2, 2, {1, 2, 3, 4});
    AvgPool2d pool(2, 1, 1, true);
    auto out = pool.forward(x);

    std::cout << "\n[avgpool_forward_count_include_pad]\n";
    print_vector(x->data, "x", 2, 2);
    print_vector(out->data, "avg_pool2d(x)", out->rows, out->cols);

    std::string err;
    if (!expect_vector_close(out->data,
                             {0.25, 0.75, 0.5, 1.0, 2.5, 1.5, 0.75, 1.75, 1.0},
                             "avg pool include pad data",
                             err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_avgpool_backward_without_padding() {
    auto x = make_tensor(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    AvgPool2d pool(2, 1, 0, false);
    auto out = pool.forward(x);

    std::cout << "\n[avgpool_backward_without_padding]\n";

    out->backward();
    print_vector(x->grad, "grad x", 3, 3);

    std::string err;
    if (!expect_vector_close(x->grad,
                             {0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25},
                             "avg pool no pad grad",
                             err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_avgpool_backward_count_include_pad() {
    auto x = make_tensor(2, 2, {1, 2, 3, 4});
    AvgPool2d pool(2, 1, 1, true);
    auto out = pool.forward(x);

    std::cout << "\n[avgpool_backward_count_include_pad]\n";

    out->backward();
    print_vector(x->grad, "grad x", 2, 2);

    std::string err;
    if (!expect_vector_close(x->grad, {1, 1, 1, 1}, "avg pool include pad grad", err)) {
        std::cerr << err << "\n";
        return false;
    }

    return true;
}

bool test_avgpool_invalid_inputs() {
    auto x = make_tensor(2, 2, {1, 2, 3, 4});

    std::cout << "\n[avgpool_invalid_inputs]\n";

    bool ok = true;
    ok = expect_throws_runtime(
             [&]() {
                 AvgPool2d pool(0, 1, 0);
             },
             "avg pool invalid scalar kernel") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 AvgPool2d pool({2, 0}, {1, 1}, {0, 0});
             },
             "avg pool invalid pair kernel") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 AvgPool2d pool(2, 1, 0);
                 pool.forward(nullptr);
             },
             "avg pool null input") &&
         ok;
    ok = expect_throws_runtime(
             [&]() {
                 AvgPool2d pool(3, 1, 0);
                 pool.forward(x);
             },
             "avg pool kernel larger than input") &&
         ok;

    return ok;
}

} // namespace

std::vector<TestCase> avg_pool_tests() {
    return {
        {"avgpool_scalar_ctor_sets_fields", test_avgpool_scalar_ctor_sets_fields},
        {"avgpool_pair_ctor_sets_fields", test_avgpool_pair_ctor_sets_fields},
        {"avgpool_forward_without_padding", test_avgpool_forward_without_padding},
        {"avgpool_forward_count_include_pad", test_avgpool_forward_count_include_pad},
        {"avgpool_backward_without_padding", test_avgpool_backward_without_padding},
        {"avgpool_backward_count_include_pad", test_avgpool_backward_count_include_pad},
        {"avgpool_invalid_inputs", test_avgpool_invalid_inputs},
    };
}
