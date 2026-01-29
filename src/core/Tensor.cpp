#include "core/Tensor.hpp"
#include<iostream>
#include<stdexcept>
#include<map>


// Constructor
Tensor::Tensor(int r, int c, std::vector<std::shared_ptr<Tensor>> parents, std::string lbl)
    : rows(r), cols(c), label(lbl), prev(parents) {
        data.resize(rows * cols, 0.0);
        grad.resize(rows * cols, 0.0);
        _backward = nullptr;
    }

std::shared_ptr<Tensor> Tensor::create(int r, int c, std::vector<std::shared_ptr<Tensor>>parents, std::string lbl) {
    return std::make_shared<Tensor>(r, c, parents, lbl);
}

double &Tensor::at(int i, int j) {
    return data[i * cols + j];
}

double &Tensor::grad_at(int i, int j) {
    return grad[i * cols + j];
}

void Tensor::backward() {
    std::vector<std::shared_ptr<Tensor>> topo;
    std::map<std::shared_ptr<Tensor>, int> visited;

    std::function<void(std::shared_ptr<Tensor>)> build_topo = [&](std::shared_ptr<Tensor> v){
        if(visited[v]==1){
            throw("Phát hiện chu trình trên đồ thị tính toán");
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

    for(auto itr = topo.rbegin(); itr != topo.rend(); itr++) {
        if((*itr)->_backward){
            (*itr)->_backward();
        }
    }
}