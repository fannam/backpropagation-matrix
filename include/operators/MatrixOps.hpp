//MatrixOps.hpp

#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include "core/Tensor.hpp"

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> transpose(std::shared_ptr<Tensor> a);

#endif