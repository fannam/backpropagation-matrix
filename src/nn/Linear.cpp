#include "nn/Linear.hpp"
#include "core/Tensor.hpp"
#include <memory>
#include <vector>

Linear::Linear(int in_features, int out_features) {
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x) {
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
}
