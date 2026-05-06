#ifndef SGD_HPP
#define SGD_HPP

#include <memory>
#include <vector>
#include "core/Tensor.hpp"

class SGD {
public:
    std::vector<std::shared_ptr<Tensor>> params;
    double lr;

    SGD(std::vector<std::shared_ptr<Tensor>> params, double lr);

    void step();

    void zero_grad();
};

#endif
