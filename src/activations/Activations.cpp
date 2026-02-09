#include "core/Tensor.hpp"
#include "activations/Activations.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <cassert>
#include <stdexcept>
#include <vector>

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a){
    //ReLU: f(x) = x if x > 0 otherwise 0
    //derivative: f'(x) = 1 if x > 0 otherwise 0

    auto out = Tensor::create(a->rows, a->cols, {a}, "relu");

    for(size_t i = 0; i < a->data.size(); ++i){
        float a_val = a->data[i];
        out->data[i] = (a_val > 0) ? a_val : 0;
    }

    out->_backward = [a, out](){
        for(size_t i = 0; i < a->grad.size(); ++i){
            float a_val = a->data[i];
            a->grad[i] += (a_val > 0) ? out->grad[i] : 0;
        }
    };

    return out;
}

std::shared_ptr<Tensor> leaky_relu(std::shared_ptr<Tensor> a, float alpha){
    //leaky_relu: f(x) = x if x > 0, alpha*x otherwise
    //derivative: f'(x) = 1 if x > 0, alpha otherwwise

    auto out = Tensor::create(a->rows, a->cols, {a}, "leaky_relu");

    for(size_t i = 0; i < a->data.size(); ++i){
        float a_val = a->data[i];
        out->data[i] = (a_val > 0) ? a_val : alpha * a_val;
    }

    out->_backward = [a, alpha, out](){
        for(size_t i = 0; i < a->grad.size(); ++i){
            float out_grad = out->grad[i];
            a->grad[i] += (a->data[i] > 0) ? out_grad : out_grad * alpha;
        }
    };

    return out;
}

std::shared_ptr<Tensor> silu(std::shared_ptr<Tensor> a){
    //swish aka silu: f(x) = x*sigmoid(x)
    //derivative: f'(x) = sigmoid(x) + x*f(x)*(1-f(x))

    auto out = Tensor::create(a->rows, a->cols, {a}, "silu");

    for(size_t i = 0; i < a->data.size(); ++i){
        float a_val = a->data[i];
        out->data[i] = a_val * 1.0 / (1 + exp(-a_val));
    }

    out->_backward = [a, out](){
        for(size_t i = 0; i < a->grad.size(); ++i){
            float a_val = a->data[i];
            float sig_val = 1.0 / (1 + exp(-a_val));
            float derive = sig_val + a_val * sig_val * (1 - sig_val);
            a->grad[i] += derive * out->grad[i];
        }
    };

    return out;
}

std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a){
    //tanh: f(x) = (e^x - e^{-x})/(e^x + e^{-x})
    //derivative: f'(x) = 1 - f(x)*f(x)

    auto out = Tensor::create(a->rows, a->cols, {a}, "tanh");

    for(size_t i = 0; i < a->data.size(); ++i){
        out->data[i] = std::tanh(a->data[i]);
    }

    out->_backward = [a, out](){
        for(size_t i = 0; i < a->grad.size(); ++i){
            float tanh_val = std::tanh(a->data[i]);
            a->grad[i] += out->grad[i] * (1 - tanh_val * tanh_val);
        }
    };

    return out;
}

std::shared_ptr<Tensor> gelu_exact(std::shared_ptr<Tensor> a){
    // f(x) = 0.5*x*(1+erf(x/sqrt(2)))
    // d/dx erf(x) = 2/sqrt(pi) * e^{-x^2}
    // derivative: f'(x) = 0.5*(1+erf(x/sqrt(2))) + x/sqrt(2*pi) * exp(-x^2/2)

    auto out = Tensor::create(a->rows, a->cols, {a}, "gelu_exact");
    const float sqrt2pi = std::sqrt(2.0*M_PI);

    for(size_t i = 0; i < a->data.size(); ++i){
        float a_val = a->data[i];
        out->data[i] = 0.5 * a_val * (1 + std::erf(a_val / std::sqrt(2)));
    }

    out->_backward = [a, out, sqrt2pi](){
        for(size_t i = 0; i < a->grad.size(); ++i){
            float a_val = a->data[i];
            float derive = 0.5 * (1 + std::erf(a_val/std::sqrt(2))) + a_val/std::sqrt(2*M_PI) * std::exp(-a_val*a_val/2);
            a->grad[i] += out->grad[i] * derive; 
        }
    };

    return out;
}

std::shared_ptr<Tensor> gelu_tanh(std::shared_ptr<Tensor> a){
    // f(x) = 0.5*x*(1+tanh(sqrt(2/pi) * (x+0.044715*x^3)))
    // f'(x) = 0.5*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) 
    //      + 0.5*x*(sqrt(2/pi)*(1+0.134145*x^2))*(1-tanh^2((sqrt(2/pi)*(x+0.044715*x^3))))

    auto out = Tensor::create(a->rows, a->cols, {a}, "gelu_tanh");
    const float sqrt2topi = std::sqrt(2.0/M_PI);
    for(size_t i = 0; i < a->data.size(); ++i){
        float a_val = a->data[i];
        float u = sqrt2topi * (a_val + 0.044715f*a_val*a_val*a_val);
        out->data[i] = 0.5f*a_val*(1+std::tanh(u));
    }
    out->_backward = [a, out, sqrt2topi](){
        
        for(size_t i = 0; i < a->grad.size(); ++i){
            float a_val = a->data[i];
            float u = std::tanh(sqrt2topi * (a_val + 0.044715f*a_val*a_val*a_val));
            float derive = 0.5f*(1 + u) + 0.5f*a_val*(sqrt2topi*(1.0f+0.134145f*a_val*a_val))*(1-u*u);
            a->grad[i] += out->grad[i] * derive;
        }
    };

    return out;
}

std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> a, std::string approximate="none"){
    //GeLU: 
    // + gelu exact: f(x) = 0.5*x*(1+erf(x/sqrt(2)))
    // + gelu tanh: f(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))

    assert(approximate == "none" || approximate == "tanh");
    if(approximate == "none") return gelu_exact(a);
    else if(approximate == "tanh") return gelu_tanh(a);
    else{
        throw(std::runtime_error("Tham số approximation phải là 'none' hoặc 'tanh'!"));
    }
}

std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a){
    //sigmoid: f(x) = 1/(1+e^{-x})
    //derivative: f'(x) = -e^{-x}/(1+e^{-x})^2 = f(x)*(1-f(x))

    auto out = Tensor::create(a->rows, a->cols, {a}, "sigmoid");

    for(size_t i = 0; i < a->data.size(); ++i){
        out->data[i] = 1.0 / (1 + exp(-a->data[i]));
    }

    out->_backward = [a, out](){
        for(size_t i = 0; i < a->data.size(); ++i){
            float sig_val = 1.0 / (1 + exp(-a->data[i]));
            a->grad[i] += out->grad[i] * sig_val * (1 - sig_val);
        }
    };

    return out;
}

