#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "core/Tensor.hpp"
#include<memory>
#include<cmath>
#include<string>

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> leaky_relu(std::shared_ptr<Tensor> a, float alpha);

std::shared_ptr<Tensor> silu(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> a, std::string approximate);

std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> a);

#endif 