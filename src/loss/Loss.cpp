#include "core/Tensor.hpp"
#include "loss/Loss.hpp"
#include <cmath>
#include <stdexcept>

// ============================================================================
// MSE Loss (Mean Squared Error - Sai số bình phương trung bình)
// ============================================================================
// Công thức:
//   L = (1/n) * sum_{i=1}^{n} (p_i - t_i)^2
//   trong đó p là prediction (dự đoán), t là target (nhãn thực tế),
//   n là tổng số phần tử.
//
// Đạo hàm:
//   ∂L/∂p_i = (2/n) * (p_i - t_i)
//   ∂L/∂t_i = -(2/n) * (p_i - t_i)
// ============================================================================
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) {
    // Kiểm tra prediction và target phải có cùng shape
    if(prediction->rows != target->rows || prediction->cols != target->cols){
        throw std::runtime_error("mse_loss: prediction và target không cùng shape");
    }

    // Output là scalar (1x1), có 2 parents là prediction và target
    auto out = Tensor::create(1, 1, {prediction, target}, "mse_loss");

    // Forward: tính tổng bình phương sai số
    size_t n = prediction->data.size();
    double sum_sq = 0.0;
    for(size_t i = 0; i < n; ++i){
        double d = prediction->data[i] - target->data[i];
        sum_sq += d * d;
    }
    // Lấy trung bình để có giá trị loss cuối cùng
    out->data[0] = sum_sq / static_cast<double>(n);

    // Backward: lan truyền gradient từ out về prediction và target
    // ∂L/∂p_i = ∂L/∂out * (2/n) * (p_i - t_i)
    // ∂L/∂t_i = -∂L/∂out * (2/n) * (p_i - t_i)
    out->_backward = [prediction, target](Tensor* out){
        size_t n = prediction->data.size();
        // scale = ∂L/∂out * (2/n) - tính một lần để tái sử dụng trong vòng lặp
        double scale = (2.0 / static_cast<double>(n)) * out->grad[0];
        for(size_t i = 0; i < n; ++i){
            double d = prediction->data[i] - target->data[i];
            // Cộng dồn gradient (vì có thể có nhiều nhánh chảy về cùng tensor)
            prediction->grad[i] += scale * d;
            target->grad[i] -= scale * d;
        }
    };
    return out;
}

// ============================================================================
// NLL Loss (Negative Log Likelihood - Hàm mất mát log âm)
// ============================================================================
// Đầu vào:
//   - log_probs: tensor shape [N, C] đã qua log_softmax
//                (N = batch size, C = số class)
//   - targets:   vector các nhãn nguyên, mỗi giá trị thuộc [0, C)
//
// Công thức:
//   L = -(1/N) * sum_{i=1}^{N} log_probs[i, targets[i]]
//
// Đạo hàm (gradient sparse - chỉ khác 0 ở vị trí target):
//   ∂L/∂log_probs[i, j] = -1/N nếu j == targets[i], ngược lại = 0
//
// Trường hợp đặc biệt: nếu log_probs là vector 1D (rows==1 hoặc cols==1)
// thì coi như N=1 sample với C class.
// ============================================================================
std::shared_ptr<Tensor> nll_loss(std::shared_ptr<Tensor> log_probs, const std::vector<int>& targets) {
    // Phát hiện trường hợp 1D (vector) hay 2D (batch matrix)
    bool is_1d = (log_probs->rows == 1 || log_probs->cols == 1);
    int N = is_1d ? 1 : log_probs->rows;                                // số sample
    int C = is_1d ? static_cast<int>(log_probs->data.size())            // số class
                  : log_probs->cols;

    // Số target phải khớp với số sample
    if(static_cast<int>(targets.size()) != N){
        throw std::runtime_error("nll_loss: kích thước targets phải bằng batch size N");
    }
    // Mỗi target phải nằm trong [0, C)
    for(int t : targets){
        if(t < 0 || t >= C){
            throw std::runtime_error("nll_loss: giá trị target nằm ngoài khoảng [0, C)");
        }
    }

    // Output là scalar, parent duy nhất là log_probs
    auto out = Tensor::create(1, 1, {log_probs}, "nll_loss");

    // Forward: lấy giá trị log_probs tại vị trí target của từng sample, đảo dấu, cộng dồn
    double sum_loss = 0.0;
    if(is_1d){
        // Trường hợp 1D: chỉ có 1 sample
        sum_loss = -log_probs->data[targets[0]];
    } else {
        // Trường hợp 2D: lặp qua từng row trong batch
        for(int i = 0; i < N; ++i){
            sum_loss -= log_probs->at(i, targets[i]);
        }
    }
    // Lấy trung bình theo batch
    out->data[0] = sum_loss / static_cast<double>(N);

    // Backward: gradient sparse, chỉ đặt -1/N tại đúng vị trí targets[i]
    // ∂L/∂log_probs[i, j] = -∂L/∂out / N nếu j == targets[i], ngược lại = 0
    out->_backward = [log_probs, targets, N, is_1d](Tensor* out){
        double scale = -out->grad[0] / static_cast<double>(N);
        if(is_1d){
            log_probs->grad[targets[0]] += scale;
        } else {
            for(int i = 0; i < N; ++i){
                log_probs->grad_at(i, targets[i]) += scale;
            }
        }
    };
    return out;
}

// ============================================================================
// Cross Entropy Loss (Hàm mất mát chéo cho phân loại nhiều lớp)
// ============================================================================
// Phiên bản FUSED: gộp log_softmax + NLL trong cùng một op.
// Lợi ích:
//   1) Ổn định số học: trừ max trước khi exp tránh overflow
//   2) Forward 1 lần duyệt thay vì 2 (log_softmax + nll_loss)
//   3) Gradient được rút gọn:
//        ∂L/∂logits[i, j] = (1/N) * (softmax[i, j] - 1_{j == targets[i]})
//      Không cần lan truyền qua chuỗi log_softmax → nll_loss
//
// Công thức forward (sử dụng log-sum-exp trick):
//   m_i           = max_j logits[i, j]
//   log_sum_exp_i = log(sum_j exp(logits[i, j] - m_i))
//   L             = -(1/N) * sum_i (logits[i, t_i] - m_i - log_sum_exp_i)
//
// Đạo hàm:
//   softmax[i, j]      = exp(logits[i, j] - m_i) / sum_k exp(logits[i, k] - m_i)
//   ∂L/∂logits[i, j]   = (1/N) * (softmax[i, j] - 1_{j == targets[i]})
// ============================================================================
std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> logits, const std::vector<int>& targets) {
    // Phát hiện trường hợp 1D / 2D giống nll_loss
    bool is_1d = (logits->rows == 1 || logits->cols == 1);
    int N = is_1d ? 1 : logits->rows;                                   // số sample
    int C = is_1d ? static_cast<int>(logits->data.size())               // số class
                  : logits->cols;

    // Kiểm tra hợp lệ
    if(static_cast<int>(targets.size()) != N){
        throw std::runtime_error("cross_entropy: kích thước targets phải bằng batch size N");
    }
    for(int t : targets){
        if(t < 0 || t >= C){
            throw std::runtime_error("cross_entropy: giá trị target nằm ngoài khoảng [0, C)");
        }
    }

    // Output là scalar, parent là logits
    auto out = Tensor::create(1, 1, {logits}, "cross_entropy");

    // Lưu lại softmax để dùng trong backward (tiết kiệm việc tính lại)
    std::vector<double> softmax(static_cast<size_t>(N) * static_cast<size_t>(C));
    double total_loss = 0.0;

    // Forward: duyệt từng sample (mỗi row của batch)
    for(int i = 0; i < N; ++i){
        // Bước 1: tìm max trong row để chuẩn bị log-sum-exp trick
        double max_val = is_1d ? logits->data[0] : logits->at(i, 0);
        for(int j = 1; j < C; ++j){
            double v = is_1d ? logits->data[j] : logits->at(i, j);
            if(v > max_val) max_val = v;
        }

        // Bước 2: tính exp(logits - max) và tổng để chuẩn hoá softmax
        double sum_exp = 0.0;
        for(int j = 0; j < C; ++j){
            double v = is_1d ? logits->data[j] : logits->at(i, j);
            double e = std::exp(v - max_val);
            softmax[static_cast<size_t>(i) * C + j] = e;
            sum_exp += e;
        }

        // Bước 3: chuẩn hoá softmax = exp(...) / sum_exp
        // Dùng inv_sum_exp để thay phép chia bằng phép nhân (nhanh hơn)
        double log_sum_exp = std::log(sum_exp);
        double inv_sum_exp = 1.0 / sum_exp;
        for(int j = 0; j < C; ++j){
            softmax[static_cast<size_t>(i) * C + j] *= inv_sum_exp;
        }

        // Bước 4: cộng dồn loss của sample i: -(logits[i, t_i] - max - log_sum_exp)
        double v_t = is_1d ? logits->data[targets[i]] : logits->at(i, targets[i]);
        total_loss -= (v_t - max_val - log_sum_exp);
    }
    // Lấy trung bình theo batch
    out->data[0] = total_loss / static_cast<double>(N);

    // Backward: dùng softmax đã cache, gradient = (softmax - one_hot) / N
    // Lambda capture softmax theo giá trị (copy) để giữ buffer sống đến khi backward chạy
    out->_backward = [logits, targets, N, C, is_1d, softmax](Tensor* out){
        double scale = out->grad[0] / static_cast<double>(N);
        for(int i = 0; i < N; ++i){
            for(int j = 0; j < C; ++j){
                double s = softmax[static_cast<size_t>(i) * C + j];
                // indicator = 1 nếu j là class đúng của sample i, ngược lại = 0
                double indicator = (j == targets[i]) ? 1.0 : 0.0;
                double g = scale * (s - indicator);
                if(is_1d) logits->grad[j] += g;
                else logits->grad_at(i, j) += g;
            }
        }
    };
    return out;
}
