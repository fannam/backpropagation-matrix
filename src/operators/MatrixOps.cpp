#include "core/Tensor.hpp"
#include "operators/MatrixOps.hpp"
#include "operators/UnaryOps.hpp"
#include<cmath>
#include<stdexcept>

std::shared_ptr<Tensor> Tensor::T() {
    return transpose(shared_from_this());
}

std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->cols, a->rows, {a}, "T");

    for(size_t i = 0; i < static_cast<size_t>(a->rows); ++i){
        for(size_t j = 0; j < static_cast<size_t>(a->cols); ++j){
            out->at(j, i) = a->at(i, j);
        }
    }

    out->_backward = [a, out]() {
        for(size_t i = 0; i < static_cast<size_t>(a->rows); ++i){
            for(size_t j = 0; j < static_cast<size_t>(a->cols); ++j){
                a->grad_at(i, j) += out->grad_at(j, i);
            }
        }
    };

    return out;
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if(a->cols != b->rows){
        throw(std::runtime_error("Kích thước ma trận không thể nhân với nhau!"));
    }

    int m = a->rows;
    int n = a->cols; 
    int p = b->cols;
    size_t ms = static_cast<size_t>(m);
    size_t ns = static_cast<size_t>(n);
    size_t ps = static_cast<size_t>(p);
    auto out = Tensor::create(m, p, {a, b}, "mat_mul");

    // C = A @ B
    // c_{i,j} = sum_{k=1}^{n} a_{i,k}.b_{k, j}, i=1...m, j=1...p
    for(size_t i = 0; i < ms; ++i){
        for(size_t j = 0; j < ps; ++j){
            for(size_t k = 0; k < ns; ++k){
                out->at(i, j) += a->at(i, k) * b->at(k, j);
            }
        }
    }

    // a_{i,k} ảnh hưởng đến c_{i,1}, c_{i,2},..., c_{i,p} --> gradient sẽ sum 1->p (i=1...m)
    // ∂L/∂a_{i,k} = sum_{j=1}^{p} ∂L/∂c_{i,j} · ∂c_{i,j}/∂a_{i,k}
    // ∂L/∂a_{i,k} = sum_{j=1}^{p} ∂L/∂c_{i,j} . b_{k,j}

    // b_{k,j} ảnh hưởng đến c_{1,j}, c_{2,j},..., c_{m, j} --> gradient sẽ sum 1->m (j=1...p)
    // ∂L/∂b_{k,j} = sum_{i=1}^{m} ∂L/∂c_{i,j} . ∂c_{i,j}/∂b_{k,j}
    // ∂L/∂b_{k,j} = sum_{i=1}^{m} ∂L/∂c_{i,j} . a_{i,k}

    out->_backward = [a, b, out, ms, ns, ps]() {
        for(size_t i = 0; i < ms; ++i){
            for(size_t j = 0; j < ps; ++j){
                double grad_val = out->grad_at(i, j);
                for(size_t k = 0; k < ns; ++k){
                    a->grad_at(i, k) += grad_val * b->at(k, j);
                }
            }
        }

        for(size_t i = 0; i < ms; ++i){
            for(size_t j = 0; j < ps; ++j){
                double grad_val = out->grad_at(i, j);
                for(size_t k = 0; k < ns; ++k){
                    b->grad_at(k, j) += grad_val * a->at(i, k);
                }
            }
        }
    };

    return out;
}
