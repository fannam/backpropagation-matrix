#ifndef TENSOR_HPP
#define TENSOR_HPP

#include<vector>
#include<memory>
#include<functional>
#include<string>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<double>data;
    std::vector<double>grad;
    int rows, cols;
    std::string label;

    std::vector<std::shared_ptr<Tensor>>prev;
    std::function<void()> _backward;

    //Constructor
    Tensor(int r, int c, std::vector<std::shared_ptr<Tensor>> parents = {}, std::string lbl="");

    static std::shared_ptr<Tensor> create(int r, int c, std::vector<std::shared_ptr<Tensor>>parents = {}, std::string lbl = ""); 
    double &at(int i, int j);
    double &grad_at(int i, int j);

    void backward();
};

#endif
