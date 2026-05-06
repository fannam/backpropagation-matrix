#ifndef LOSS_HPP
#define LOSS_HPP

#include <vector>
#include "core/Tensor.hpp"

std::shared_ptr<Tensor> mse_loss(
    std::shared_ptr<Tensor> prediction,
    std::shared_ptr<Tensor> target
);

std::shared_ptr<Tensor> nll_loss(
    std::shared_ptr<Tensor> log_probs,
    const std::vector<int>& targets
);

std::shared_ptr<Tensor> cross_entropy(
    std::shared_ptr<Tensor> logits,
    const std::vector<int>& targets
);

#endif
