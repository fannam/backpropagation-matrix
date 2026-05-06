#ifndef REDUCTION_OPS_HPP
#define REDUCTION_OPS_HPP

#include "core/Tensor.hpp"

struct Index2D {
    int row;
    int col;
};

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> min(std::shared_ptr<Tensor> a);

Index2D argmax(std::shared_ptr<Tensor> a);

Index2D argmin(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> variance(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> std_op(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> sum_rows(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> mean_rows(std::shared_ptr<Tensor> a);

#endif