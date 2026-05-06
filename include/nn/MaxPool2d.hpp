#ifndef MAX_POOL2D_HPP
#define MAX_POOL2D_HPP

#include <memory>
#include "core/Tensor.hpp"

class MaxPool2d {
public:
    int kernel_size;
    int stride;
    int padding;

    MaxPool2d(int kernel_size, int stride = 1, int padding = 0);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

#endif
