#include "core/Tensor.hpp"
#include "activations/Activations.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a){
    //ReLU: f(x) = x if x > 0 otherwise 0
    //derivative: f'(x) = 1 if x > 0 otherwise 0

    auto out = Tensor::create(a->rows, a->cols, {a}, "relu");

    for(size_t i = 0; i < a->data.size(); ++i){
        double a_val = a->data[i];
        out->data[i] = (a_val > 0) ? a_val : 0;
    }

    out->_backward = [a](Tensor* out){
        for(size_t i = 0; i < a->grad.size(); ++i){
            double a_val = a->data[i];
            a->grad[i] += (a_val > 0) ? out->grad[i] : 0.0;
        }
    };

    return out;
}

std::shared_ptr<Tensor> leaky_relu(std::shared_ptr<Tensor> a, double alpha){
    //leaky_relu: f(x) = x if x > 0, alpha*x otherwise
    //derivative: f'(x) = 1 if x > 0, alpha otherwise

    auto out = Tensor::create(a->rows, a->cols, {a}, "leaky_relu");

    for(size_t i = 0; i < a->data.size(); ++i){
        double a_val = a->data[i];
        out->data[i] = (a_val > 0) ? a_val : alpha * a_val;
    }

    out->_backward = [a, alpha](Tensor* out){
        for(size_t i = 0; i < a->grad.size(); ++i){
            double out_grad = out->grad[i];
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
        double a_val = a->data[i];
        out->data[i] = a_val / (1.0 + std::exp(-a_val));
    }

    out->_backward = [a](Tensor* out){
        for(size_t i = 0; i < a->grad.size(); ++i){
            double a_val = a->data[i];
            double sig_val = 1.0 / (1.0 + std::exp(-a_val));
            double derive = sig_val + a_val * sig_val * (1.0 - sig_val);
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

    out->_backward = [a](Tensor* out){
        for(size_t i = 0; i < a->grad.size(); ++i){
            double tanh_val = std::tanh(a->data[i]);
            a->grad[i] += out->grad[i] * (1.0 - tanh_val * tanh_val);
        }
    };

    return out;
}

std::shared_ptr<Tensor> gelu_exact(std::shared_ptr<Tensor> a){
    // f(x) = 0.5*x*(1+erf(x/sqrt(2)))
    // d/dx erf(x) = 2/sqrt(pi) * e^{-x^2}
    // derivative: f'(x) = 0.5*(1+erf(x/sqrt(2))) + x/sqrt(2*pi) * exp(-x^2/2)

    auto out = Tensor::create(a->rows, a->cols, {a}, "gelu_exact");
    const double sqrt2 = std::sqrt(2.0);
    const double sqrt2pi = std::sqrt(2.0 * M_PI);

    for(size_t i = 0; i < a->data.size(); ++i){
        double a_val = a->data[i];
        out->data[i] = 0.5 * a_val * (1 + std::erf(a_val / sqrt2));
    }

    out->_backward = [a, sqrt2, sqrt2pi](Tensor* out){
        for(size_t i = 0; i < a->grad.size(); ++i){
            double a_val = a->data[i];
            double derive = 0.5 * (1 + std::erf(a_val / sqrt2)) +
                            a_val / sqrt2pi * std::exp(-a_val * a_val / 2);
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
    const double sqrt2topi = std::sqrt(2.0 / M_PI);
    for(size_t i = 0; i < a->data.size(); ++i){
        double a_val = a->data[i];
        double u = sqrt2topi * (a_val + 0.044715 * a_val * a_val * a_val);
        out->data[i] = 0.5 * a_val * (1 + std::tanh(u));
    }
    out->_backward = [a, sqrt2topi](Tensor* out){
        for(size_t i = 0; i < a->grad.size(); ++i){
            double a_val = a->data[i];
            double u = std::tanh(sqrt2topi * (a_val + 0.044715 * a_val * a_val * a_val));
            double derive = 0.5 * (1 + u) +
                            0.5 * a_val * (sqrt2topi * (1.0 + 0.134145 * a_val * a_val)) *
                            (1 - u * u);
            a->grad[i] += out->grad[i] * derive;
        }
    };

    return out;
}

std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> a, std::string approximate="none"){
    //GeLU: 
    // + gelu exact: f(x) = 0.5*x*(1+erf(x/sqrt(2)))
    // + gelu tanh: f(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))

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
        out->data[i] = 1.0 / (1.0 + std::exp(-a->data[i]));
    }

    out->_backward = [a](Tensor* out){
        for(size_t i = 0; i < a->data.size(); ++i){
            double sig_val = 1.0 / (1.0 + std::exp(-a->data[i]));
            a->grad[i] += out->grad[i] * sig_val * (1.0 - sig_val);
        }
    };

    return out;
}

std::shared_ptr<Tensor> log_softmax(std::shared_ptr<Tensor> a){
    //log_softmax(x_i) = log(e^{x_i}/sum_{j} e^{x_j}) = x_i - log(sum_{j} e^{x_j})
    //log_softmax(x_i) = log(e^{x_i-max_x}/sum_{j} e^{x_j-max_x}) = x_i - max_x - log(sum_{j} e^{x_j-max_x})

    //derivative (vector): dL/dx_i = dL/dy_i - softmax_i * sum_j dL/dy_j
    auto out = Tensor::create(a->rows, a->cols, {a}, "log_softmax");

    const int rows = a->rows;
    const int cols = a->cols;

    if(a->rows == 1 || a->cols == 1){
        // treat as a single vector
        double max_val = a->data[0];
        for(double x : a->data){
            if(x > max_val) max_val = x;
        }
        double sum_exp = 0.0;
        for(double x : a->data){
            sum_exp += std::exp(x - max_val);
        }
        double log_sum_exp = std::log(sum_exp);
        for(size_t i = 0; i < out->data.size(); ++i){
            out->data[i] = a->data[i] - max_val - log_sum_exp;
        }

        out->_backward = [a](Tensor* out){
            double sum_grad = 0.0;
            for(double g : out->grad){
                sum_grad += g;
            }
            for(size_t i = 0; i < out->grad.size(); ++i){
                double softmax_i = std::exp(out->data[i]);
                a->grad[i] += out->grad[i] - softmax_i * sum_grad;
            }
        };
    } 
    else {
        // apply log_softmax along each row
        for(int r = 0; r < rows; ++r){
            double max_val = a->at(r, 0);
            for(int c = 1; c < cols; ++c){
                double v = a->at(r, c);
                if(v > max_val) max_val = v;
            }
            double sum_exp = 0.0;
            for(int c = 0; c < cols; ++c){
                sum_exp += std::exp(a->at(r, c) - max_val);
            }
            double log_sum_exp = std::log(sum_exp);
            for(int c = 0; c < cols; ++c){
                out->at(r, c) = a->at(r, c) - max_val - log_sum_exp;
            }
        }

        out->_backward = [a, rows, cols](Tensor* out){
            for(int r = 0; r < rows; ++r){
                double sum_grad = 0.0;
                for(int c = 0; c < cols; ++c){
                    sum_grad += out->grad_at(r, c);
                }
                for(int c = 0; c < cols; ++c){
                    double softmax_i = std::exp(out->at(r, c));
                    a->grad_at(r, c) += out->grad_at(r, c) - softmax_i * sum_grad;
                }
            }
        };
    }

    return out;
}
