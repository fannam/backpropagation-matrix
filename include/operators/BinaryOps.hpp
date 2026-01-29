//BinaryOps.hpp

#ifndef BINARY_OPS_HPP
#define BINARY_OPS_HPP

#include "core/Tensor.hpp"
#include "operators/UnaryOps.hpp"

std::shared_ptr<Tensor> add (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> hadamard_mul (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> hadamard_div (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

inline std::shared_ptr<Tensor> operator + (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
    return add(a, b);
}

inline std::shared_ptr<Tensor> operator - (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
    return add(a, negate(b));
}

inline std::shared_ptr<Tensor> operator * (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
    return hadamard_mul(a, b);
}

inline std::shared_ptr<Tensor> operator / (std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
    return hadamard_div(a, b);
}


#endif