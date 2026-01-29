//UnaryOps.hpp

#ifndef UNARY_OPS_HPP
#define UNARY_OPS_HPP

#include "core/Tensor.hpp"

std::shared_ptr<Tensor> negate(std::shared_ptr<Tensor> a); // phép lấy số đối

std::shared_ptr<Tensor> exp_op(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> ln(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> scalar_div_tensor(double scalar, std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> tensor_div_scalar(std::shared_ptr<Tensor> a, double scalar);

std::shared_ptr<Tensor> reciprocal(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> tensor_add_scalar(std::shared_ptr<Tensor> a, double scalar);

std::shared_ptr<Tensor> tensor_sub_scalar(std::shared_ptr<Tensor> a, double scalar);

inline std::shared_ptr<Tensor> operator - (std::shared_ptr<Tensor> a) {
    return negate(a);
}

inline std::shared_ptr<Tensor> operator / (double scalar, std::shared_ptr<Tensor> a) {
    return scalar_div_tensor(scalar, a);
}

inline std::shared_ptr<Tensor> operator / (std::shared_ptr<Tensor> a, double scalar) {
    return tensor_div_scalar(a, scalar);
}

inline std::shared_ptr<Tensor> operator + (double scalar, std::shared_ptr<Tensor> a) {
    return tensor_add_scalar(a, scalar);
}

inline std::shared_ptr<Tensor> operator + (std::shared_ptr<Tensor> a, double scalar) {
    return tensor_add_scalar(a, scalar);
}

inline std::shared_ptr<Tensor> operator - (double scalar, std::shared_ptr<Tensor> a) {
    return tensor_add_scalar(negate(a), scalar);
}

inline std::shared_ptr<Tensor> operator - (std::shared_ptr<Tensor> a, double scalar) {
    return tensor_add_scalar(a, -scalar);
}



#endif
