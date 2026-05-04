#include "core/Tensor.hpp"
#include<algorithm>
#include<initializer_list>
#include<map>
#include<stdexcept>
#include<string>

namespace {

size_t tensor_size(int rows, int cols) {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols);
}

size_t checked_index(const Tensor& tensor, int i, int j, const char* accessor) {
    if(i < 0 || i >= tensor.rows || j < 0 || j >= tensor.cols) {
        throw std::runtime_error(std::string(accessor) + ": index out of bounds");
    }
    return static_cast<size_t>(i * tensor.cols + j);
}

} // namespace

// Constructor
Tensor::Tensor(int r, int c, std::vector<std::shared_ptr<Tensor>> parents, std::string lbl)
    : rows(r), cols(c), label(lbl), prev(parents) {
        if(rows <= 0 || cols <= 0) {
            throw std::runtime_error("Tensor shape must have positive rows and cols");
        }
        data.resize(tensor_size(rows, cols), 0.0);
        grad.resize(tensor_size(rows, cols), 0.0);
        _backward = nullptr;
    }

std::shared_ptr<Tensor> Tensor::create(int r, int c, std::vector<std::shared_ptr<Tensor>>parents, std::string lbl) {
    return std::make_shared<Tensor>(r, c, parents, lbl);
}

std::shared_ptr<Tensor> Tensor::create(int r, int c, const std::vector<double>& values, std::string lbl) {
    auto tensor = std::make_shared<Tensor>(r, c, std::vector<std::shared_ptr<Tensor>>{}, lbl);
    if(values.size() != tensor->data.size()) {
        throw std::runtime_error("Tensor::create values size mismatch");
    }
    tensor->data = values;
    return tensor;
}

std::shared_ptr<Tensor> Tensor::create(int r, int c, std::initializer_list<double> values, std::string lbl) {
    return create(r, c, std::vector<double>(values), lbl);
}

double &Tensor::at(int i, int j) {
    return data[checked_index(*this, i, j, "Tensor::at")];
}

const double &Tensor::at(int i, int j) const {
    return data[checked_index(*this, i, j, "Tensor::at")];
}

double &Tensor::grad_at(int i, int j) {
    return grad[checked_index(*this, i, j, "Tensor::grad_at")];
}

const double &Tensor::grad_at(int i, int j) const {
    return grad[checked_index(*this, i, j, "Tensor::grad_at")];
}

void Tensor::backward() {
    std::vector<std::shared_ptr<Tensor>> topo;
    std::map<std::shared_ptr<Tensor>, int> visited;

    std::function<void(std::shared_ptr<Tensor>)> build_topo = [&](std::shared_ptr<Tensor> v){
        if(visited[v]==1){
            throw(std::runtime_error("Phát hiện chu trình trên đồ thị tính toán"));
        }
        if(visited[v]==2) return;

        visited[v]=1;
        for(auto &parent : v->prev) {
            build_topo(parent);
        }

        visited[v]=2;
        topo.push_back(v);
    };

    build_topo(shared_from_this());
    std::fill(this->grad.begin(), this->grad.end(), 1.0);

    for(auto itr = topo.rbegin(); itr != topo.rend(); ++itr) {
        if((*itr)->_backward){
            (*itr)->_backward();
        }
    }
}

void Tensor::backward(const std::vector<double>& seed_grad){
    if(seed_grad.size() != this->grad.size()) {
        throw std::runtime_error("Tensor::backward seed_grad size mismatch");
    }

    std::vector<std::shared_ptr<Tensor>> topo;
    std::map<std::shared_ptr<Tensor>, int> visited;

    std::function<void(std::shared_ptr<Tensor>)> build_topo = [&](std::shared_ptr<Tensor> v){
        if(visited[v]==1){
            throw(std::runtime_error("Phát hiện chu trình trên đồ thị tính toán"));
        }
        if(visited[v]==2) return;

        visited[v]=1;
        for(auto &parent : v->prev) {
            build_topo(parent);
        }

        visited[v]=2;
        topo.push_back(v);
    };

    build_topo(shared_from_this());

    this->grad = seed_grad;

    for(auto itr = topo.rbegin(); itr != topo.rend(); ++itr) {
        if((*itr)->_backward){
            (*itr)->_backward();
        }
    }
}

void Tensor::zero_grad() {
    std::fill(this->grad.begin(), this->grad.end(), 0.0);
}
