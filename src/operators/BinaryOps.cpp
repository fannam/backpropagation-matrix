#include "core/Tensor.hpp"
#include "operators/BinaryOps.hpp"
#include<cmath>
#include<stdexcept>

std::shared_ptr<Tensor> add (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if(a->rows != b->rows || a->cols != b->cols) {
        throw(std::runtime_error("Kích thước hai ma trận không khớp"));
    }
    auto out = Tensor::create(a->rows, a->cols, {a, b}, "+");

    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }

    out->_backward = [a, b, out]() {
        for(size_t i = 0; i < a->data.size(); ++i) {
            a->grad[i] += out->grad[i];
            b->grad[i] += out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> hadamard_mul (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if(a->rows != b->rows || a->cols != b->cols) {
        throw(std::runtime_error("Kích thước hai ma trận không khớp"));
    }
    auto out = Tensor::create(a->rows, a->cols, {a, b}, "h_mul");

    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = a->data[i] * b->data[i];
    }

    out->_backward = [a, b, out]() {
        for(size_t i = 0; i < a->data.size(); ++i) {
            a->grad[i] += b->data[i] * out->grad[i];
            b->grad[i] += a->data[i] * out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> hadamard_div (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if(a->rows != b->rows || a->cols != b->cols) {
        throw(std::runtime_error("Kích thước hai ma trận không khớp"));
    }
    auto out = Tensor::create(a->rows, a->cols, {a, b}, "h_div");

    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = a->data[i] / b->data[i];
    }

    out->_backward = [a, b, out]() {
        for(size_t i = 0; i < a->data.size(); ++i) {
            double b_val = b->data[i];
            a->grad[i] += (1.0/b_val) * out->grad[i];
            b->grad[i] -= (a->data[i] / (b_val * b_val)) * out->grad[i]; 
        }
    };

    return out;
}
