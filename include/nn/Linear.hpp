#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <memory>
#include <vector>
#include "core/Tensor.hpp"

class Linear {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    Linear(int in_features, int out_features);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);

    std::vector<std::shared_ptr<Tensor>> parameters();
};

#endif
