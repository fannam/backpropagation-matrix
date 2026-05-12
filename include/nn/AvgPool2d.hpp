#ifndef AVG_POOL2D_HPP
#define AVG_POOL2D_HPP

#include <memory>
#include <utility>
#include "core/Tensor.hpp"

class AvgPool2d {
public:
    int kernel_rows;
    int kernel_cols;
    int stride_rows;
    int stride_cols;
    int padding_rows;
    int padding_cols;
    bool count_include_pad;

    AvgPool2d(int kernel_size, int stride = 1, int padding = 0, bool count_include_pad = false);
    AvgPool2d(
        std::pair<int, int> kernel_size,
        std::pair<int, int> stride = {1, 1},
        std::pair<int, int> padding = {0, 0},
        bool count_include_pad = false
    );

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

#endif
