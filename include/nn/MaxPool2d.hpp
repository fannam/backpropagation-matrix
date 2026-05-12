#ifndef MAX_POOL2D_HPP
#define MAX_POOL2D_HPP

#include <memory>
#include <utility>
#include "core/Tensor.hpp"

class MaxPool2d {
public:
    int kernel_rows;
    int kernel_cols;
    int stride_rows;
    int stride_cols;
    int padding_rows;
    int padding_cols;

    MaxPool2d(int kernel_size, int stride = 1, int padding = 0);
    MaxPool2d(
        std::pair<int, int> kernel_size,
        std::pair<int, int> stride = {1, 1},
        std::pair<int, int> padding = {0, 0}
    );

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

#endif
