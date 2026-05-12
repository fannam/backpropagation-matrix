#include "nn/MaxPool2d.hpp"
#include "core/Tensor.hpp"
#include <memory>
#include <utility>

// Constructor scalar: người dùng truyền int như PyTorch.
// Ý nghĩa:
// - kernel_size = k  -> kernel_rows = k, kernel_cols = k
// - stride = s       -> stride_rows = s, stride_cols = s
// - padding = p      -> padding_rows = p, padding_cols = p
//
// Việc cần code:
// 1. Validate kernel_size > 0.
// 2. Validate stride > 0.
// 3. Validate padding >= 0.
// 4. Có thể validate padding < kernel_size để tránh window toàn padding.
// 5. Gán các member *_rows và *_cols bằng cùng một giá trị scalar.
MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding) {
}

// Constructor 2D: người dùng truyền pair giống tuple[int, int] của PyTorch.
// Quy ước:
// - pair.first  = rows dimension
// - pair.second = cols dimension
//
// Ví dụ:
// MaxPool2d({2, 3}, {1, 2}, {0, 1})
// nghĩa là:
// - kernel_rows = 2, kernel_cols = 3
// - stride_rows = 1, stride_cols = 2
// - padding_rows = 0, padding_cols = 1
//
// Việc cần code:
// 1. Tách kernel_size.first/second thành kernel_rows/kernel_cols.
// 2. Tách stride.first/second thành stride_rows/stride_cols.
// 3. Tách padding.first/second thành padding_rows/padding_cols.
// 4. Validate:
//    - kernel_rows > 0 và kernel_cols > 0
//    - stride_rows > 0 và stride_cols > 0
//    - padding_rows >= 0 và padding_cols >= 0
//    - padding_rows < kernel_rows và padding_cols < kernel_cols nếu muốn ràng buộc pooling an toàn.
// 5. Gán vào member của object.
MaxPool2d::MaxPool2d(
    std::pair<int, int> kernel_size,
    std::pair<int, int> stride,
    std::pair<int, int> padding
) {
}

// Forward MaxPool2d.
// Input hiện tại của repo là Tensor 2D:
// - x->rows = input_rows
// - x->cols = input_cols
// Chưa support batch/channel kiểu PyTorch (N, C, H, W).
//
// Output shape:
// - output_rows = floor((input_rows + 2 * padding_rows - kernel_rows) / stride_rows) + 1
// - output_cols = floor((input_cols + 2 * padding_cols - kernel_cols) / stride_cols) + 1
//
// Việc cần code phần forward:
// 1. Validate x != nullptr.
// 2. Tính output_rows, output_cols bằng công thức trên.
// 3. Validate output_rows > 0 và output_cols > 0.
// 4. Tạo out = Tensor::create(output_rows, output_cols, {x}, "max_pool2d").
// 5. Tạo bộ nhớ lưu vị trí max cho từng ô output:
//    - max_rows có size output_rows * output_cols
//    - max_cols có size output_rows * output_cols
// 6. Duyệt từng ô output out[i, j].
// 7. Tính góc trái trên của window trong tọa độ input:
//    - row_start = i * stride_rows - padding_rows
//    - col_start = j * stride_cols - padding_cols
// 8. Duyệt kernel:
//    - in_row = row_start + ki, với ki từ 0 tới kernel_rows - 1
//    - in_col = col_start + kj, với kj từ 0 tới kernel_cols - 1
// 9. Nếu in_row/in_col nằm ngoài input thì bỏ qua.
// 10. Tìm giá trị lớn nhất trong các ô hợp lệ.
// 11. Gán out[i, j] = max_value.
// 12. Lưu tọa độ max vào max_rows/max_cols tại flat index:
//     - flat_index = i * output_cols + j
//
// Padding convention cho MaxPool:
// - Bỏ qua ô ngoài input.
// - Tương đương padding bằng -infinity.
// - Padding không bao giờ thắng max.
//
// Tie convention:
// - Nếu nhiều phần tử bằng max, chọn phần tử max đầu tiên khi duyệt window.
// - Dùng so sánh ">" thay vì ">=" để giữ max đầu tiên.
//
// Backward cần code trong out->_backward:
// Công thức:
// - out[i,j] = max(window)
// - Gradient chỉ đi về phần tử input đã thắng max.
// - Nếu out[i,j] lấy max từ x[r,c]:
//   x.grad[r,c] += out.grad[i,j]
//
// Việc cần code phần backward:
// 1. Capture x, max_rows, max_cols, output_cols vào lambda.
// 2. Với từng out[i,j], lấy flat_index = i * output_cols + j.
// 3. Đọc in_row = max_rows[flat_index], in_col = max_cols[flat_index].
// 4. Cộng:
//    x->grad_at(in_row, in_col) += out->grad_at(i, j)
//
// Lưu ý:
// - Backward MaxPool không chia gradient.
// - Nếu cùng một input là max của nhiều output window, grad sẽ được cộng dồn nhiều lần.
std::shared_ptr<Tensor> MaxPool2d::forward(std::shared_ptr<Tensor> x) {
}
