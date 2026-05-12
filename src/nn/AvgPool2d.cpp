#include "nn/AvgPool2d.hpp"
#include "core/Tensor.hpp"
#include <memory>
#include <utility>
#include <stdexcept>

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
AvgPool2d::AvgPool2d(int kernel_size, int stride, int padding, bool count_include_pad) {
    if(kernel_size <= 0 || stride <= 0 || padding < 0) {
        throw std::runtime_error("AvgPool2d error: Required kernel_size > 0, stride > 0 and padding >= 0.");
    }

    if(padding >= kernel_size) {
        throw std::runtime_error("AvgPool2d error: Required padding < kernel_size.");
    }

    this->kernel_rows       = kernel_size;
    this->kernel_cols       = kernel_size;
    this->stride_rows       = stride;
    this->stride_cols       = stride;
    this->padding_rows      = padding;
    this->padding_cols      = padding;
    this->count_include_pad = count_include_pad;
}

// Constructor 2D: người dùng truyền pair giống tuple[int, int] của PyTorch.
// Quy ước:
// - pair.first  = rows dimension
// - pair.second = cols dimension
//
// Ví dụ:
// AvgPool2d({2, 3}, {1, 2}, {0, 1})
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
AvgPool2d::AvgPool2d(
    std::pair<int, int> kernel_size,
    std::pair<int, int> stride,
    std::pair<int, int> padding,
    bool count_include_pad
) {
    if(kernel_size.first <= 0 || kernel_size.second <= 0 || 
       stride.first <= 0 || stride.second <= 0 ||
       padding.first < 0 || padding.second < 0) {
        throw std::runtime_error("AvgPool2d error: Required kernel_size > 0, stride > 0 and padding >= 0.");
    }

    if(padding.first >= kernel_size.first || padding.second >= kernel_size.second) {
        throw std::runtime_error("AvgPool2d error: Required padding < kernel_size.");
    }

    this->kernel_rows       = kernel_size.first;
    this->kernel_cols       = kernel_size.second;
    this->stride_rows       = stride.first;
    this->stride_cols       = stride.second;
    this->padding_rows      = padding.first;
    this->padding_cols      = padding.second;
    this->count_include_pad = count_include_pad;
}

// Forward AvgPool2d.
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
// 4. Tạo out = Tensor::create(output_rows, output_cols, {x}, "avg_pool2d").
// 5. Duyệt từng ô output out[i, j].
// 6. Tính góc trái trên của window trong tọa độ input:
//    - row_start = i * stride_rows - padding_rows
//    - col_start = j * stride_cols - padding_cols
// 7. Duyệt kernel:
//    - in_row = row_start + ki, với ki từ 0 tới kernel_rows - 1
//    - in_col = col_start + kj, với kj từ 0 tới kernel_cols - 1
// 8. Nếu in_row/in_col nằm ngoài input thì bỏ qua.
// 9. Cộng sum các ô hợp lệ, đếm count các ô hợp lệ.
// 10. Gán out[i, j] = sum / count.
//
// Padding convention đang khuyên dùng:
// - Bỏ qua ô ngoài input.
// - Chia cho count ô hợp lệ.
// - Nghĩa là không coi padding là số 0 trong mẫu trung bình.
//
// Backward cần code trong out->_backward:
// Công thức:
// - out[i,j] = sum(valid x trong window) / count
// - d(out[i,j]) / d(x[r,c]) = 1 / count nếu x[r,c] nằm trong window hợp lệ
// - Vì vậy:
//   x.grad[r,c] += out.grad[i,j] / count
//
// Việc cần code phần backward:
// 1. Capture x và các tham số pooling rows/cols vào lambda.
// 2. Lặp lại đúng cùng window như forward.
// 3. Tính lại count ô hợp lệ.
// 4. grad_share = out->grad_at(i, j) / count.
// 5. Cộng grad_share vào mọi x->grad_at(in_row, in_col) hợp lệ trong window.
//
// Lưu ý:
// - Forward và backward phải dùng cùng padding convention.
// - Nếu forward chia theo count hợp lệ, backward cũng phải chia theo count hợp lệ.
// - Nếu sau này muốn giống PyTorch hơn, có thể thêm option count_include_pad.
std::shared_ptr<Tensor> AvgPool2d::forward(std::shared_ptr<Tensor> x) {
    if(!x){
        throw std::runtime_error("AvgPool2d error: Input is null.");
    }

    if(x->rows <= 0 || x->cols <= 0){
        throw std::runtime_error("AvgPool2d error: Input must have positive dimensions.");
    }

    if(x->rows + 2 * padding_rows < kernel_rows || x->cols + 2 * padding_cols < kernel_cols){
        throw std::runtime_error("AvgPool2d error: Kernel size must be smaller than input size.");
    }

    int out_rows = (x->rows + 2 * padding_rows - kernel_rows) / stride_rows + 1;
    int out_cols = (x->cols + 2 * padding_cols - kernel_cols) / stride_cols + 1;
    auto out = Tensor::create(out_rows, out_cols, {x}, "avg_pool2d");

    // Snapshot các tham số để capture vào lambda (tránh capture this).
    int k_rows = kernel_rows;
    int k_cols = kernel_cols;
    int s_rows = stride_rows;
    int s_cols = stride_cols;
    int p_rows = padding_rows;
    int p_cols = padding_cols;
    bool include_pad = count_include_pad;
    // denominator cố định khi include_pad = true
    int kernel_area = k_rows * k_cols;

    for(int i = 0; i < out_rows; ++i){
        for(int j = 0; j < out_cols; ++j){
            int start_row = i * s_rows - p_rows;
            int start_col = j * s_cols - p_cols;
            double sum  = 0.0;
            int    count = 0;
            for(int ki = 0; ki < k_rows; ++ki){
                for(int kj = 0; kj < k_cols; ++kj){
                    int in_row = start_row + ki;
                    int in_col = start_col + kj;
                    if(in_row >= 0 && in_row < x->rows && in_col >= 0 && in_col < x->cols){
                        sum += x->at(in_row, in_col);
                        ++count;
                    }
                }
            }
            // count_include_pad = true  → chia kernel_area (coi padding = 0, giống PyTorch default)
            // count_include_pad = false → chia số ô thực sự có giá trị
            double denom = include_pad ? static_cast<double>(kernel_area)
                                       : static_cast<double>(count);
            out->at(i, j) = (count > 0) ? sum / denom : 0.0;
        }
    }

    out->_backward = [x, out, k_rows, k_cols, s_rows, s_cols, p_rows, p_cols,
                      include_pad, kernel_area](Tensor* /* self */) {
        int in_rows = x->rows;
        int in_cols = x->cols;
        int out_rows = out->rows;
        int out_cols = out->cols;

        for(int i = 0; i < out_rows; ++i){
            for(int j = 0; j < out_cols; ++j){
                int start_row = i * s_rows - p_rows;
                int start_col = j * s_cols - p_cols;
                // Tính lại count ô hợp lệ để dùng đúng denominator như forward.
                int count = 0;
                for(int ki = 0; ki < k_rows; ++ki){
                    for(int kj = 0; kj < k_cols; ++kj){
                        int in_row = start_row + ki;
                        int in_col = start_col + kj;
                        if(in_row >= 0 && in_row < in_rows && in_col >= 0 && in_col < in_cols){
                            ++count;
                        }
                    }
                }
                if(count == 0) continue;
                double denom = include_pad ? static_cast<double>(kernel_area)
                                           : static_cast<double>(count);
                double grad_share = out->grad_at(i, j) / denom;
                for(int ki = 0; ki < k_rows; ++ki){
                    for(int kj = 0; kj < k_cols; ++kj){
                        int in_row = start_row + ki;
                        int in_col = start_col + kj;
                        if(in_row >= 0 && in_row < in_rows && in_col >= 0 && in_col < in_cols){
                            x->grad_at(in_row, in_col) += grad_share;
                        }
                    }
                }
            }
        }
    };
    return out;
}
