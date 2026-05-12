#include "nn/Linear.hpp"
#include "core/Tensor.hpp"
#include "operators/MatrixOps.hpp"
#include "nn/Init.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

// Khởi tạo layer tuyến tính với weight [in_features, out_features] và bias [1, out_features].
Linear::Linear(int in_features, int out_features) {
    if (in_features <= 0) {
        throw std::runtime_error("Linear: in_features must be > 0");
    }

    if (out_features <= 0) {
        throw std::runtime_error("Linear: out_features must be > 0");
    }

    weight = Tensor::create(in_features, out_features, std::vector<std::shared_ptr<Tensor>>{}, "linear_weight");
    bias = Tensor::create(1, out_features, std::vector<std::shared_ptr<Tensor>>{}, "linear_bias");

    xavier_uniform_(weight, 42);
    zeros_(bias);
}

// Tính y = x @ weight + bias, trong đó bias được broadcast theo từng dòng của batch.
std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x) {
    if (!x) {
        throw std::runtime_error("Linear::forward: input is null");
    }

    if (!weight) {
        throw std::runtime_error("Linear::forward: weight is null");
    }

    if (!bias) {
        throw std::runtime_error("Linear::forward: bias is null");
    }

    if (x->cols != weight->rows) {
        throw std::runtime_error("Linear::forward: input features dimension mismatch");
    }

    if (bias->rows != 1 || bias->cols != weight->cols) {
        throw std::runtime_error("Linear::forward: bias shape must be [1, out_features]");
    }

    auto y = matmul(x, weight);
    auto bias_param = bias;
    auto out = Tensor::create(y->rows, y->cols, {y, bias_param}, "linear");

    for (int i = 0; i < y->rows; ++i) {
        for (int j = 0; j < y->cols; ++j) {
            out->at(i, j) = y->at(i, j) + bias_param->at(0, j);
        }
    }

    // Backward:
    // out = y + bias, nên dL/dy[i,j] += dL/dout[i,j].
    // bias[0,j] dùng chung cho mọi dòng batch, nên dL/dbias[0,j] += sum_i dL/dout[i,j].
    // Gradient cho x và weight đi tiếp qua node y = matmul(x, weight).
    out->_backward = [y, bias_param](Tensor* out) {
        for (int i = 0; i < out->rows; ++i) {
            for (int j = 0; j < out->cols; ++j) {
                double grad_val = out->grad_at(i, j);

                y->grad_at(i, j) += grad_val;
                // Cộng dồn theo batch vì cùng một bias[0,j] ảnh hưởng tới mọi out[i,j].
                bias_param->grad_at(0, j) += grad_val;
            }
        }
    };

    return out;
}

// Trả về danh sách tham số trainable để optimizer cập nhật.
std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    return {weight, bias};
}
