#ifndef INIT_HPP
#define INIT_HPP

#include <memory>
#include "core/Tensor.hpp"

void uniform_(std::shared_ptr<Tensor> t, double low, double high, unsigned seed);

#endif
