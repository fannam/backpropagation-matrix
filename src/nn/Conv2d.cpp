#include "nn/Conv2d.hpp"
#include "core/Tensor.hpp"
#include <memory>
#include <vector>

// Conv2d trong repo này là convolution trên một ma trận 2D single-channel.
//
// Quy ước:
// - input  xem như [H, W]
// - weight xem như [kernel_size, kernel_size]
// - output xem như [H_out, W_out]
//
// Hai tham số in_channels và out_channels vẫn có mặt trong API,
// nhưng bản implement matrix-only nên chỉ chấp nhận:
// - in_channels  = 1
// - out_channels = 1
Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    // TODO:
    // 1. Validate tham số:
    //    - in_channels > 0
    //    - out_channels > 0
    //    - kernel_size > 0
    //    - stride > 0
    //    - padding >= 0
    //
    // 2. Reject sớm nếu:
    //    - in_channels != 1
    //    - out_channels != 1
    //    vì layer này chỉ support một ma trận 2D.
    //
    // 3. Gán toàn bộ tham số vào member của object.
    //
    // 4. Khởi tạo weight và bias theo shape 2D tối thiểu:
    //    - weight: [kernel_size, kernel_size]
    //    - bias:   [1, 1]
    //
    // 5. Khởi tạo weight bằng random hợp lý.
    //    Thực tế nên dùng fan-in/fan-out init, ví dụ kaiming/xavier tùy activation.
    //
    // 6. Khởi tạo bias = 0.
}

std::shared_ptr<Tensor> Conv2d::forward(std::shared_ptr<Tensor> x) {
    // TODO:
    // 1. Validate:
    //    - x != nullptr
    //    - weight != nullptr
    //    - bias != nullptr
    //    - x là ma trận 2D hợp lệ
    //
    // 2. Diễn giải input:
    //    - x xem như input [H, W]
    //    - không có chiều batch
    //    - không có chiều channel
    //
    // 3. Tính output shape:
    //    - output_rows = floor((H + 2 * padding - kernel_size) / stride) + 1
    //    - output_cols = floor((W + 2 * padding - kernel_size) / stride) + 1
    //    - validate output_rows > 0 và output_cols > 0
    //
    // 4. Tạo output tensor:
    //    - Tensor::create(output_rows, output_cols, {x, weight, bias}, "conv2d")
    //
    // 5. Forward single-channel:
    //    - với mỗi ô output (i, j):
    //      a. row_start = i * stride - padding
    //      b. col_start = j * stride - padding
    //      c. duyệt kernel (ki, kj)
    //      d. nếu tọa độ input nằm ngoài biên thì coi như 0
    //      e. cộng:
    //         sum += x[in_row, in_col] * weight[ki, kj]
    //      f. out[i, j] = sum + bias
    //
    // 6. Backward cần có 3 nhánh:
    //    - dL/dx
    //    - dL/dweight
    //    - dL/dbias
    //
    // 7. Backward single-channel tối thiểu:
    //    - với mỗi out[i, j], lấy grad_out = out->grad_at(i, j)
    //    - bias.grad += grad_out
    //    - với mỗi phần tử kernel (ki, kj):
    //      a. xác định input thật: in_row, in_col
    //      b. nếu hợp lệ:
    //         weight.grad[ki, kj] += x[in_row, in_col] * grad_out
    //         x.grad[in_row, in_col] += weight[ki, kj] * grad_out
    //
    // 8. Cần giữ forward và backward cùng padding convention:
    //    - ngoài biên input được xem là 0
    //    - phần padding không có gradient riêng
}

std::vector<std::shared_ptr<Tensor>> Conv2d::parameters() {
    // TODO:
    // Trả về toàn bộ trainable parameter theo thứ tự ổn định.
    // Bản matrix-only:
    // - {weight, bias}
    //
    // Nên đảm bảo:
    // - không trả nullptr nếu layer đã khởi tạo đúng
    // - thứ tự không đổi để optimizer dùng ổn định
}
