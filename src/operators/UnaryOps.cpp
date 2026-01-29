#include<cmath>
#include "core/Tensor.hpp"
#include "operators/UnaryOps.hpp"

std::shared_ptr<Tensor> negate(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->rows, a->cols, {a}, "neg");

    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = -a->data[i];
    }

    out->_backward = [a, out]() {
        for(size_t i = 0; i < a->grad.size(); ++i) {
            a->grad[i] -= out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> exp_op(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->rows, a->cols, {a}, "exp");

    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = std::exp(a->data[i]);
    }

    out->_backward = [a, out]() {
        for(size_t i = 0; i < a->grad.size(); ++i) {
            a->grad[i] += out->data[i] * out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> log_op(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->rows, a->cols, {a}, "ln");

    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = std::log(a->data[i]);
    }

    out->_backward = [a, out]() {
        for(size_t i = 0; i < a->data.size(); ++i) {
            a->grad[i] += (1.0/a->data[i]) * out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> scalar_div_tensor(double scalar, std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->rows, a->cols, {a}, "scl_div_tensor");

    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = scalar/a->data[i];
    }

    out->_backward = [a, scalar, out]() {
        for(size_t i = 0; i < a->data.size(); i++) {
            double a_val = a->data[i];
            a->grad[i] -= (scalar / (a_val * a_val)) * out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> reciprocal(std::shared_ptr<Tensor> a) {
    return scalar_div_tensor(1.0, a);
}

std::shared_ptr<Tensor> tensor_add_scalar(std::shared_ptr<Tensor> a, double scalar) {
    auto out = Tensor::create(a->rows, a->cols, {a}, "tensor_add_scl");

    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = a->data[i] + scalar;
    }

    out->_backward = [a, out]() {
        for(size_t i = 0; i < a->data.size(); ++i) {
            a->grad[i] += out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> tensor_mul_scalar(std::shared_ptr<Tensor> a, double scalar) {
    auto out = Tensor::create(a->rows, a->cols, {a}, "tensor_mul_scl");
    for(size_t i = 0; i < a->data.size(); ++i) {
        out->data[i] = a->data[i] * scalar;
    }

    out->_backward = [a, scalar, out] {
        for(size_t i = 0; i < a->data.size(); ++i) {
            a->grad[i] += scalar * out->grad[i];
        }
    };

    return out;
}
