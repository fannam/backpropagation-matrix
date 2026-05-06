#ifndef AVG_POOL2D_HPP
#define AVG_POOL2D_HPP

#include <memory>
#include "core/Tensor.hpp"

class AvgPool2d {
public:
    int kernel_size;
    int stride;
    int padding;

    AvgPool2d(int kernel_size, int stride = 1, int padding = 0);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

#endif
