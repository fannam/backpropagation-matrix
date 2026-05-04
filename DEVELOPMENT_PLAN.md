# Ke hoach phat trien step-by-step

## 1. Muc tieu

Muc tieu cua repo la phat trien tu mot autograd engine C++ nho cho Tensor 2D thanh mot mini deep learning library co the:

- Bieu dien tensor 2D voi `data`, `grad`, shape va graph tinh toan.
- Ho tro forward va backward cho cac phep toan co ban.
- Co test ro rang cho moi operator.
- Co loss, layer va optimizer toi thieu de train duoc mot model nho.
- Co example de nguoi dung chay va hieu luong tinh toan.

Trang thai hien tai:

- Build bang `make`.
- Binary hien tai la `test_app`.
- `src/main.cpp` dang dong vai tro test harness thu cong.
- Da co: add, subtract qua negate, hadamard multiply/divide, scalar ops, transpose, matmul.
- Da co activations: ReLU, LeakyReLU, SiLU, GeLU, sigmoid, tanh, log_softmax.
- `log_softmax` da implement nhung chua duoc test trong `main.cpp`.

## 2. Nguyen tac phat trien

- Moi thay doi operator phai co test forward va backward.
- Uu tien dung `double` nhat quan, tranh cast ve `float` neu khong can thiet.
- Khi them API moi, phai co example hoac test the hien cach dung.
- Giua cac buoc phai dam bao `make run` pass.
- Khong refactor lon khi chua co test bao ve hanh vi hien co.

## 3. Phase 0 - Chuan hoa baseline

Muc tieu: biet chac code hien tai dang dung o dau va co cach kiem tra lap lai.

### Step 0.1 - Ghi README ngan

Tao `README.md` gom:

- Ten project.
- Cach build: `make`.
- Cach chay test hien tai: `make run`.
- Mo ta ngan ve `Tensor`, operator va autograd.

Tieu chi hoan thanh:

- Nguoi moi clone repo biet cach build va chay trong duoi 1 phut.

### Step 0.2 - Giu lai baseline test hien tai

Chay:

```bash
make clean
make run
```

Tieu chi hoan thanh:

- Build thanh cong.
- Tat ca test hien tai pass.

### Step 0.3 - Ghi nhan han che hien tai

Them vao README hoac file roadmap:

- Chua co test framework rieng.
- `main.cpp` dang tron demo va test.
- Chua co reduction ops nhu `sum`, `mean`.
- Chua co loss, layer, optimizer.
- Chua co numerical gradient check.

## 4. Phase 1 - Tach test khoi main

Muc tieu: bien repo thanh cau truc co the mo rong test de dang.

### Step 1.1 - Tao thu muc tests

Tao cau truc:

```text
tests/
  test_helpers.hpp
  test_ops.cpp
  test_activations.cpp
```

Noi dung du kien:

- `test_helpers.hpp`: `make_tensor`, `close_enough`, `expect_vector_close`, helper in loi.
- `test_ops.cpp`: add, multiply, divide, scalar ops, transpose, matmul.
- `test_activations.cpp`: relu, leaky_relu, silu, gelu, sigmoid, tanh, log_softmax.

Tieu chi hoan thanh:

- Helper khong con nam trong `src/main.cpp`.
- Moi file test chi chua cac test lien quan.

### Step 1.2 - Doi Makefile de build test binary

Cap nhat `Makefile` theo huong:

- Source library nam trong `src/core`, `src/operators`, `src/activations`.
- Test source nam trong `tests`.
- Target mac dinh build `test_app`.
- `make run` chay `./test_app`.
- `make clean` xoa object va binary.

Tieu chi hoan thanh:

- `make run` van pass.
- Them test moi khong can sua `src/main.cpp`.

### Step 1.3 - Chuyen `src/main.cpp` thanh demo nho

Co 2 lua chon:

- Lua chon A: bo `src/main.cpp` khoi build test, de test binary co `tests/main.cpp`.
- Lua chon B: doi `src/main.cpp` thanh example don gian, va build bang target rieng.

Khuyen nghi: Lua chon A.

Tieu chi hoan thanh:

- Test va demo khong con bi tron.

## 5. Phase 2 - Cung co Tensor core

Muc tieu: lam `Tensor` an toan va de dung hon truoc khi them nhieu operator.

### Step 2.1 - Them constructor tu values

Them API:

```cpp
static std::shared_ptr<Tensor> create(
    int rows,
    int cols,
    const std::vector<double>& values,
    std::string label = ""
);
```

Tieu chi hoan thanh:

- Neu `values.size() != rows * cols`, throw `std::runtime_error`.
- Test tao tensor dung shape.
- Test tao tensor sai size.

### Step 2.2 - Validate shape

Trong constructor `Tensor(int r, int c, ...)`, them check:

- `r > 0`
- `c > 0`

Tieu chi hoan thanh:

- Tao tensor shape hop le pass.
- Tao tensor rows/cols <= 0 throw exception.

### Step 2.3 - Them const accessors

Them:

```cpp
const double& at(int i, int j) const;
const double& grad_at(int i, int j) const;
```

Tieu chi hoan thanh:

- Code doc du lieu tu `const Tensor&` duoc.
- Test compile va pass.

### Step 2.4 - Them bounds checking

Trong `at()` va `grad_at()`, check:

- `0 <= i < rows`
- `0 <= j < cols`

Tieu chi hoan thanh:

- Access hop le pass.
- Access out-of-bounds throw exception ro rang.

### Step 2.5 - Them `zero_grad()`

Them method:

```cpp
void zero_grad();
```

Tieu chi hoan thanh:

- Sau backward, goi `zero_grad()` dua grad ve 0.
- Test xac nhan co the backward lai sau khi reset grad.

### Step 2.6 - Them `backward(seed_grad)`

Hien tai `backward()` mac dinh set grad output bang toan 1. Nen them:

```cpp
void backward(const std::vector<double>& seed_grad);
```

Tieu chi hoan thanh:

- Neu seed size sai, throw exception.
- Neu seed size dung, output gradient bang seed.
- Test voi `add`, `matmul`, `log_softmax` co seed khong dong deu.

## 6. Phase 3 - Hoan thien test operator hien co

Muc tieu: bao ve behavior hien tai truoc khi them feature moi.

### Step 3.1 - Test `log_softmax`

Them test cho:

- Vector row, vi du shape `1x3`.
- Vector column, vi du shape `3x1`.
- Matrix, vi du shape `2x3`, apply theo tung row.
- Backward voi seed gradient khong phai toan 1.

Tieu chi hoan thanh:

- Forward dung log-softmax on dinh so hoc.
- Backward dung cong thuc: `dL/dx_i = dL/dy_i - softmax_i * sum_j dL/dy_j`.

### Step 3.2 - Test error paths

Them test cho:

- Add hai tensor khac shape.
- Hadamard multiply/divide khac shape.
- Matmul sai kich thuoc.
- Chia cho 0.
- `log_op` voi input <= 0.
- `gelu` voi `approximate` khong hop le.

Tieu chi hoan thanh:

- Error message ro rang.
- Khong crash bang `assert` trong runtime path.

### Step 3.3 - Them numerical gradient check

Tao helper:

```cpp
double finite_difference(
    std::function<double()> fn,
    double& x,
    double eps = 1e-6
);
```

Ap dung cho:

- `hadamard_mul`
- `hadamard_div`
- `matmul`
- `sigmoid`
- `tanh`
- `gelu`
- `log_softmax`

Tieu chi hoan thanh:

- Analytical gradient gan finite-difference gradient trong tolerance.
- Cac test nay chay duoc bang `make run`.

## 7. Phase 4 - Them reduction ops

Muc tieu: co du primitive de tao loss scalar.

### Step 4.1 - Them `sum`

API du kien:

```cpp
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a);
```

Forward:

- Output shape `1x1`.
- Data la tong tat ca phan tu.

Backward:

- Moi phan tu input nhan gradient cua output.

Tieu chi hoan thanh:

- Test forward/backward voi tensor `2x3`.

### Step 4.2 - Them `mean`

API du kien:

```cpp
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a);
```

Forward:

- Output shape `1x1`.
- Data la trung binh tat ca phan tu.

Backward:

- Moi phan tu input nhan `out_grad / num_elements`.

Tieu chi hoan thanh:

- Test forward/backward voi tensor `2x3`.

### Step 4.3 - Them row-wise reductions neu can

Chi them sau khi scalar reductions da on:

```cpp
std::shared_ptr<Tensor> sum_rows(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mean_rows(std::shared_ptr<Tensor> a);
```

Tieu chi hoan thanh:

- Shape output ro rang.
- Backward broadcast gradient dung.

## 8. Phase 5 - Them loss functions

Muc tieu: co the train model bang loss scalar.

### Step 5.1 - Them MSE loss

API du kien:

```cpp
std::shared_ptr<Tensor> mse_loss(
    std::shared_ptr<Tensor> prediction,
    std::shared_ptr<Tensor> target
);
```

Cong thuc:

```text
mean((prediction - target)^2)
```

Can them truoc:

- `mean`
- hadamard multiply da co
- subtract da co qua operator overload

Tieu chi hoan thanh:

- Forward dung.
- Backward dung voi prediction.
- Target co grad hay khong can quyet dinh ro trong design.

### Step 5.2 - Them NLL loss

Dung voi output cua `log_softmax`.

API du kien:

```cpp
std::shared_ptr<Tensor> nll_loss(
    std::shared_ptr<Tensor> log_probs,
    const std::vector<int>& targets
);
```

Tieu chi hoan thanh:

- Ho tro batch shape `batch_size x num_classes`.
- Output shape `1x1`.
- Gradient chi tac dong vao class dung.

### Step 5.3 - Them cross entropy loss

API du kien:

```cpp
std::shared_ptr<Tensor> cross_entropy(
    std::shared_ptr<Tensor> logits,
    const std::vector<int>& targets
);
```

Co the implement bang:

```text
nll_loss(log_softmax(logits), targets)
```

Tieu chi hoan thanh:

- Forward dung.
- Backward dung.
- Test batch nho `2x3`.

## 9. Phase 6 - Them layer va optimizer

Muc tieu: train duoc model don gian.

### Step 6.1 - Them Linear layer

Tao:

```text
include/nn/Linear.hpp
src/nn/Linear.cpp
```

API du kien:

```cpp
class Linear {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    Linear(int in_features, int out_features);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::vector<std::shared_ptr<Tensor>> parameters();
};
```

Forward:

```text
x @ weight + bias
```

Can quyet dinh bias broadcast:

- Cach nhanh: chi ho tro batch size 1 luc dau.
- Cach tot hon: them operator add bias theo row.

Khuyen nghi: them bias broadcast theo row.

Tieu chi hoan thanh:

- Test shape output.
- Test backward co grad cho input, weight, bias.

### Step 6.2 - Them random initialization

Them helper:

```cpp
void uniform_(std::shared_ptr<Tensor> t, double low, double high, unsigned seed);
```

Tieu chi hoan thanh:

- Cung seed tao cung data.
- Gia tri nam trong `[low, high]`.

### Step 6.3 - Them SGD optimizer

Tao:

```text
include/optim/SGD.hpp
src/optim/SGD.cpp
```

API du kien:

```cpp
class SGD {
public:
    SGD(std::vector<std::shared_ptr<Tensor>> params, double lr);
    void step();
    void zero_grad();
};
```

Tieu chi hoan thanh:

- `step()` cap nhat `param->data -= lr * param->grad`.
- `zero_grad()` reset grad cho moi param.
- Test mot parameter scalar/tensor nho.

## 10. Phase 7 - Example train duoc

Muc tieu: chung minh library dung duoc ngoai unit test.

### Step 7.1 - Linear regression example

Tao:

```text
examples/linear_regression.cpp
```

Bai toan:

```text
y = 2x + 1
```

Model:

```text
Linear(1, 1)
MSE loss
SGD
```

Tieu chi hoan thanh:

- Loss giam sau nhieu epoch.
- Weight gan 2, bias gan 1.

### Step 7.2 - Classification example

Tao:

```text
examples/tiny_classification.cpp
```

Model:

```text
Linear(input_dim, num_classes)
cross_entropy
SGD
```

Tieu chi hoan thanh:

- Forward/backward chay on.
- Accuracy tren dataset nho tang ro.

## 11. Phase 8 - Cai thien API va cau truc project

Muc tieu: code de dung va de maintain hon.

### Step 8.1 - Gom public headers

Them:

```text
include/autograd.hpp
```

Noi dung:

```cpp
#include "core/Tensor.hpp"
#include "operators/BinaryOps.hpp"
#include "operators/UnaryOps.hpp"
#include "operators/MatrixOps.hpp"
#include "activations/Activations.hpp"
```

Tieu chi hoan thanh:

- Example chi can include `autograd.hpp`.

### Step 8.2 - Chuan hoa namespace

Can nhac dua code vao namespace:

```cpp
namespace bpm {
    ...
}
```

Tieu chi hoan thanh:

- Khong gay conflict voi `std::tanh` hoac ten ham pho bien.
- Cap nhat toan bo test va example.

### Step 8.3 - Doi build system neu can

Khi project lon hon, can nhac chuyen tu Makefile don sang CMake.

Chi lam khi:

- Co nhieu binary test/example.
- Makefile bat dau kho maintain.

Tieu chi hoan thanh:

- `cmake -S . -B build`
- `cmake --build build`
- `ctest --test-dir build`

## 12. Technical debt can xu ly som

### 12.1 - Dung `double` nhat quan

Hien co nhieu doan dung `float` trong activation.

Can lam:

- Doi `float a_val` thanh `double a_val`.
- Doi constant `0.044715f` thanh `0.044715`.
- Doi `exp(...)` thanh `std::exp(...)`.

Tieu chi hoan thanh:

- Test activation van pass.
- Numerical gradient gan hon.

### 12.2 - Loai bo `assert` trong runtime validation

Trong `gelu`, invalid `approximate` nen throw exception thay vi assert.

Tieu chi hoan thanh:

- Build release va debug deu co behavior giong nhau.

### 12.3 - Chia cho 0 va log domain

Can them check:

- `hadamard_div`: neu divisor bang 0 thi throw.
- `scalar_div_tensor`: neu tensor element bang 0 thi throw.
- `log_op`: neu input <= 0 thi throw.

Tieu chi hoan thanh:

- Co test cho moi case.

### 12.4 - Backward nhieu lan

Can quyet dinh semantics:

- Option A: gradients accumulate nhu PyTorch.
- Option B: `backward()` reset graph grads truoc moi lan chay.

Khuyen nghi: Option A, nhung phai document ro va co `zero_grad()`.

Tieu chi hoan thanh:

- Test backward hai lan cho thay grad accumulate.
- Test `zero_grad()` reset dung.

## 13. Thu tu commit khuyen nghi

1. Them `README.md` va `DEVELOPMENT_PLAN.md`.
2. Tach helpers test ra `tests/test_helpers.hpp`.
3. Tach test ops va activation ra `tests/`.
4. Cap nhat `Makefile` cho test binary.
5. Them test `log_softmax`.
6. Them `Tensor::zero_grad()` va test.
7. Them `Tensor::create(rows, cols, values)` va validate shape.
8. Them `backward(seed_grad)` va test.
9. Chuan hoa `double` trong activation.
10. Them error-path tests va thay `assert` bang exception.
11. Them `sum` va `mean`.
12. Them `mse_loss`.
13. Them `Linear`.
14. Them `SGD`.
15. Them example linear regression.

## 14. Definition of Done cho moi phase

Moi phase chi nen coi la xong khi:

- Code build duoc bang `make`.
- `make run` pass.
- Feature moi co test forward va backward.
- Error path quan trong co test.
- README hoac example duoc cap nhat neu API thay doi.
- Khong co object/binary moi bi track trong git.

## 15. Checklist lenh kiem tra

Dung checklist nay truoc khi ket thuc moi buoc:

```bash
make clean
make
make run
git status --short
```

Ket qua mong doi:

- `make` thanh cong.
- `make run` pass tat ca test.
- `git status --short` chi hien cac file minh chu dong sua.

