#ifndef INIT_HPP
#define INIT_HPP

#include <memory>
#include "core/Tensor.hpp"

// Các hàm khởi tạo tham số cho Tensor.
// Quy ước dấu gạch dưới cuối tên hàm giống PyTorch: hàm sửa trực tiếp dữ liệu trong `t->data`.
// Các hàm này không tạo node trong đồ thị autograd và không thay đổi `t->grad`.

// Gán toàn bộ phần tử của tensor bằng 0.0.
void zeros_(std::shared_ptr<Tensor> t);

// Gán toàn bộ phần tử của tensor bằng 1.0.
void ones_(std::shared_ptr<Tensor> t);

// Lấy mẫu phân phối đều trong đoạn [low, high].
// `seed` giúp kết quả lặp lại được, thuận tiện cho test và debug.
void uniform_(std::shared_ptr<Tensor> t, double low, double high, unsigned seed);

// Lấy mẫu phân phối chuẩn với trung bình `mean` và độ lệch chuẩn `stddev`.
// Nếu `stddev == 0`, mọi phần tử được gán đúng bằng `mean`.
void normal_(std::shared_ptr<Tensor> t, double mean, double stddev, unsigned seed);

// Xavier/Glorot uniform: phù hợp cho nhiều layer tuyến tính với sigmoid/tanh.
// Bound = sqrt(6 / (fan_in + fan_out)).
void xavier_uniform_(std::shared_ptr<Tensor> t, unsigned seed);

// Kaiming/He uniform: phù hợp hơn cho mạng dùng ReLU hoặc biến thể của ReLU.
// Bound = sqrt(6 / fan_in).
void kaiming_uniform_(std::shared_ptr<Tensor> t, unsigned seed);

#endif
