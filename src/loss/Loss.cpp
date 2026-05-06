#include "core/Tensor.hpp"
#include "loss/Loss.hpp"
#include "operators/ReductionOps.hpp"
#include "operators/BinaryOps.hpp"
#include "operators/UnaryOps.hpp"
#include "activations/Activations.hpp"
#include <cmath>
#include <stdexcept>

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) {
}

std::shared_ptr<Tensor> nll_loss(std::shared_ptr<Tensor> log_probs, const std::vector<int>& targets) {
}

std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> logits, const std::vector<int>& targets) {
}
