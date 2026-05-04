#ifndef TENSOR_HPP
#define TENSOR_HPP

#include<vector>
#include<memory>
#include<functional>
#include<initializer_list>
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
    static std::shared_ptr<Tensor> create(int r, int c, const std::vector<double>& values, std::string lbl = "");
    static std::shared_ptr<Tensor> create(int r, int c, std::initializer_list<double> values, std::string lbl = "");
    std::shared_ptr<Tensor> T();
    double &at(int i, int j);
    const double &at(int i, int j) const;
    double &grad_at(int i, int j);
    const double &grad_at(int i, int j) const;

    void backward();
    void backward(const std::vector<double>& seed_grad);
    void zero_grad();
    
};

#endif
