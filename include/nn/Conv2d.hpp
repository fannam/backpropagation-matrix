#ifndef CONV2D_HPP
#define CONV2D_HPP

#include <memory>
#include <vector>
#include "core/Tensor.hpp"

class Conv2d {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);

    std::vector<std::shared_ptr<Tensor>> parameters();
};

#endif
