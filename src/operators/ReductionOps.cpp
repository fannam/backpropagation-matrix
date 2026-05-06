#include "core/Tensor.hpp"
#include "operators/ReductionOps.hpp"
#include<cmath>
#include<stdexcept>
#include<algorithm>
#include<numeric>

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(1, 1, {a}, "sum");
    out->data[0] = std::accumulate(a->data.begin(), a->data.end(), 0.0);

    //f(x) = x1+x2+...+x_n
    //∂L/∂x_i = ∂L/∂f.∂f/∂x_i = ∂L/∂x_i 
    out->_backward = [a, out](){
        double imcoming_grad = out->grad[0];
        for(size_t i=0; i < a->grad.size(); ++i){
            a->grad[i] += imcoming_grad;
        }
    };
    return out;
}

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(1, 1, {a}, "mean");
    size_t n = a->data.size();
    out->data[0] = std::accumulate(a->data.begin(), a->data.end(), 0.0);
    out->data[0] = out->data[0]/n;

    //f(x) = (x1+x2+...+x_n)/n
    //∂L/∂x_i = ∂L/∂f.∂f/∂x_i = ∂L/∂x_i * 1/n
    out->_backward = [a, out](){
        double grad_per_elem = out->grad[0] / a->grad.size();
        for(size_t i = 0; i < a->grad.size(); ++i){
            a->grad[i] += grad_per_elem;
        }
    };
    return out;
}

std::shared_ptr<Tensor> max(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(1, 1, {a}, "max");
    double max_val = a->data[0];
    std::vector<size_t> max_indices = {0};
    for(size_t i = 1; i < a->data.size(); ++i){
        if(a->data[i] > max_val){
            max_val = a->data[i];
            max_indices.clear();
            max_indices.push_back(i);
        } 
        else if(a->data[i] == max_val){
            max_indices.push_back(i);
        }
    }
    out->data[0] = max_val;

    //f(x) = max(x1,...,x_n)
    //∂L/∂x_i = ∂L/∂f / count nếu x_i = max, else 0
    out->_backward = [a, out, max_indices](){
        double share = out->grad[0] / max_indices.size();
        for(size_t idx : max_indices){
            a->grad[idx] += share;
        }
    };
    return out;
}

std::shared_ptr<Tensor> min(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(1, 1, {a}, "min");
    double min_val = a->data[0];
    std::vector<size_t> min_indices;
    for(size_t i = 1; i < a->data.size(); ++i){
        if(a->data[i] < min_val){
            min_val = a->data[i];
            min_indices.clear();
            min_indices.push_back(i);
        }
        else if(a->data[i] == min_val){
            min_indices.push_back(i);
        }
    }
    out->data[0] = min_val;

    //f(x) = max(x1,...,x_n)
    //∂L/∂x_i = ∂L/∂f / count nếu x_i = min, else 0
    out->_backward = [a, out, min_indices](){
        double share = out->grad[0] / min_indices.size();
        for(size_t idx : min_indices){
            a->grad[idx] += share;
        }
    };
    return out;
}

Index2D argmax(std::shared_ptr<Tensor> a) {
    size_t best = 0;
    double max_val = a->data[0];
    for(size_t i = 1; i < a->data.size(); ++i){
        if(a->data[i] > max_val){
            max_val = a->data[i];
            best = i;
        }
    }
    return Index2D{static_cast<int>(best / a->cols), static_cast<int>(best % a->cols)};
}

Index2D argmin(std::shared_ptr<Tensor> a) {
    size_t best = 0;
    double min_val = a->data[0];
    for(size_t i = 1; i < a->data.size(); ++i){
        if(a->data[i] < min_val){
            min_val = a->data[i];
            best = i;
        }
    }
    return Index2D{static_cast<int>(best / a->cols), static_cast<int>(best % a->cols)};
}

std::shared_ptr<Tensor> variance(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(1, 1, {a}, "var");
    size_t n = a->data.size();
    double mean = std::accumulate(a->data.begin(), a->data.end(), 0.0) / n;

    double var = 0;
    for(double x : a->data){
        double d = x - mean;
        var += d * d;
    }
    out->data[0] = var / n;

    //f(x) = 1/n * sum_{i=1}^{n} (x_i - mean)^2
    //∂L/∂x_i = ∂L/∂f . 2/n . (x_i - mean)
    out->_backward = [a, out, mean](){
        size_t n = a->data.size();
        double scale = (2.0 / n) * out->grad[0];
        for(size_t i = 0; i < a->grad.size(); ++i){
            a->grad[i] += scale * (a->data[i] - mean);
        }
    };
    return out;
}

std::shared_ptr<Tensor> std_op(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(1, 1, {a}, "std");
    size_t n = a->data.size();
    double mean = std::accumulate(a->data.begin(), a->data.end(), 0.0) / n;

    double var = 0;
    for(double x : a->data){
        double d = x - mean;
        var += d * d;
    }
    var /= n;
    double sigma = std::sqrt(var);
    out->data[0] = sigma;

    //σ = sqrt(variance)
    //∂σ/∂x_i = ∂L/∂σ . (x_i - mean) / (n * σ)
    out->_backward = [a, out, mean, sigma](){
        if(sigma == 0.0){
            throw std::runtime_error("std_op backward: sigma = 0, gradient undefined");
        }
        size_t n = a->data.size();
        double scale = out->grad[0] / (n * sigma);
        for(size_t i = 0; i < a->grad.size(); ++i){
            a->grad[i] += scale * (a->data[i] - mean);
        }
    };
    return out;
}

std::shared_ptr<Tensor> sum_rows(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->rows, 1, {a}, "sum_rows");
    for(size_t i = 0; i < a->rows; ++i){
        double row_sum = 0;
        for(size_t j = 0; j < a->cols; ++j){
            row_sum += a->at(i, j);
        }
        out->data[i] = row_sum;
    }

    //y_i = x_i1 + x_i2 + ... + x_in
    //∂L/∂x_ij = ∂L/∂y_i.∂y_i/∂x_ij = ∂L/∂y_i 
    out->_backward = [a, out](){
        for(size_t i = 0; i < a->rows; ++i){
            double g = out->grad[i];
            for(size_t j = 0; j < a->cols; ++j){
                a->grad_at(i, j) += g;
            }
        }
    };
    return out;
}

std::shared_ptr<Tensor> mean_rows(std::shared_ptr<Tensor> a) {
    auto out = Tensor::create(a->rows, 1, {a}, "mean_rows");
    int cols = a->cols;
    for(size_t i = 0; i < a->rows; ++i){
        double row_sum = 0;
        for(size_t j = 0; j < a->cols; ++j){
            row_sum += a->at(i, j);
        }
        out->data[i] = row_sum / cols;
    }

    //y_i = (x_i1 + x_i2 + ... + x_in) / cols
    //∂L/∂x_ij = ∂L/∂y_i / cols
    out->_backward = [a, out](){
        int cols = a->cols;
        for(size_t i = 0; i < a->rows; ++i){
            double g_per_elem = out->grad[i] / cols;
            for(size_t j = 0; j < a->cols; ++j){
                a->grad_at(i, j) += g_per_elem;
            }
        }
    };
    return out;
}