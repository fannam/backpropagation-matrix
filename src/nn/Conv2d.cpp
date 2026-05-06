#include "nn/Conv2d.hpp"
#include "core/Tensor.hpp"
#include <memory>
#include <vector>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
}

std::shared_ptr<Tensor> Conv2d::forward(std::shared_ptr<Tensor> x) {
}

std::vector<std::shared_ptr<Tensor>> Conv2d::parameters() {
}
