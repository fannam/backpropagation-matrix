#include <iostream>

#include "activations/Activations.hpp"
#include "core/Tensor.hpp"
#include "operators/BinaryOps.hpp"
#include "operators/MatrixOps.hpp"

int main() {
    auto x = Tensor::create(1, 2, {1.0, 2.0}, "x");
    auto w = Tensor::create(2, 1, {0.5, -1.0}, "w");
    auto b = Tensor::create(1, 1, {0.1}, "b");

    auto logits = matmul(x, w) + b;
    auto prediction = sigmoid(logits);

    prediction->backward();

    std::cout << "prediction: " << prediction->at(0, 0) << "\n";
    std::cout << "grad x: " << x->grad_at(0, 0) << " " << x->grad_at(0, 1) << "\n";
    std::cout << "grad w: " << w->grad_at(0, 0) << " " << w->grad_at(1, 0) << "\n";
    std::cout << "grad b: " << b->grad_at(0, 0) << "\n";

    return 0;
}
