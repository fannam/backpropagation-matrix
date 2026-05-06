#include "nn/Init.hpp"
#include "core/Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace {

// Kiểm tra con trỏ Tensor trước khi ghi dữ liệu.
// Tách helper này để mọi hàm init trả lỗi cùng kiểu và message dễ lần ra hàm gọi.
void require_tensor(const std::shared_ptr<Tensor>& t, const char* name) {
    if (!t) {
        throw std::runtime_error(std::string(name) + ": tensor is null");
    }
}

// Với Linear trong repo này, forward dự kiến là `x @ weight`.
// Nếu `x` có shape [batch, in_features] thì `weight` có shape [in_features, out_features].
// Vì vậy số input đi vào mỗi output chính là số hàng của weight.
double fan_in(const Tensor& t) {
    return static_cast<double>(t.rows);
}

// Số output của layer tuyến tính chính là số cột của weight.
double fan_out(const Tensor& t) {
    return static_cast<double>(t.cols);
}

} // namespace

void zeros_(std::shared_ptr<Tensor> t) {
    require_tensor(t, "zeros_");
    // Chỉ sửa data, không chạm vào grad hay graph autograd.
    std::fill(t->data.begin(), t->data.end(), 0.0);
}

void ones_(std::shared_ptr<Tensor> t) {
    require_tensor(t, "ones_");
    // Dùng cho bias hoặc test khi cần giá trị cố định.
    std::fill(t->data.begin(), t->data.end(), 1.0);
}

void uniform_(std::shared_ptr<Tensor> t, double low, double high, unsigned seed) {
    require_tensor(t, "uniform_");
    // Khoảng lấy mẫu phải hợp lệ để tránh phân phối có tham số đảo ngược.
    if (low > high) {
        throw std::runtime_error("uniform_: low must be <= high");
    }

    // mt19937 với seed cố định cho cùng chuỗi số ngẫu nhiên ở mỗi lần chạy.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(low, high);

    // Ghi trực tiếp từng phần tử trong data để init chạy in-place.
    for (double& value : t->data) {
        value = dist(gen);
    }
}

void normal_(std::shared_ptr<Tensor> t, double mean, double stddev, unsigned seed) {
    require_tensor(t, "normal_");
    // Độ lệch chuẩn âm không có nghĩa thống kê.
    if (stddev < 0.0) {
        throw std::runtime_error("normal_: stddev must be >= 0");
    }
    // Trường hợp suy biến: phân phối chuẩn co lại thành một điểm tại mean.
    if (stddev == 0.0) {
        std::fill(t->data.begin(), t->data.end(), mean);
        return;
    }

    // Seed cố định giúp normal_ deterministic giống uniform_.
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(mean, stddev);

    // Lấy mẫu Gaussian độc lập cho từng phần tử.
    for (double& value : t->data) {
        value = dist(gen);
    }
}

void xavier_uniform_(std::shared_ptr<Tensor> t, unsigned seed) {
    require_tensor(t, "xavier_uniform_");
    // Xavier giữ phương sai tín hiệu tương đối ổn định giữa input và output.
    double bound = std::sqrt(6.0 / (fan_in(*t) + fan_out(*t)));
    uniform_(t, -bound, bound, seed);
}

void kaiming_uniform_(std::shared_ptr<Tensor> t, unsigned seed) {
    require_tensor(t, "kaiming_uniform_");
    // Kaiming dùng fan_in để bù việc ReLU thường loại khoảng một nửa activation âm.
    double bound = std::sqrt(6.0 / fan_in(*t));
    uniform_(t, -bound, bound, seed);
}
