# Coding Rules & Conventions

## 1. Loop Index Types: `int` vs `size_t`

To ensure safety, prevent unsigned underflow bugs (especially with padding and strides), and avoid compiler warnings (`-Wsign-compare`), please adhere to the following rules when writing `for` loops:

### Rule 1: Iterate over spatial dimensions (`rows`, `cols`, `kernel_size`, etc.) using `int`
Since the tensor dimensions (`Tensor::rows` and `Tensor::cols`) are defined as `int`, any loops traversing spatial coordinates must use `int`. This prevents dangerous underflow errors when coordinates can temporarily be negative (e.g., `i * stride - padding`).

**Do this:**
```cpp
// CORRECT
for(int i = 0; i < tensor->rows; ++i) {
    for(int j = 0; j < tensor->cols; ++j) {
        // Safe to do index arithmetic like i * stride - padding
    }
}
```

**Don't do this:**
```cpp
// INCORRECT: causes warnings and risk of underflow
for(size_t i = 0; i < tensor->rows; ++i) {
    for(size_t j = 0; j < tensor->cols; ++j) {
        // If (i * stride - padding) < 0, it wraps around to a massive positive number!
    }
}
```

### Rule 2: Iterate over 1D vector arrays (`std::vector::size()`) using `size_t`
The `.size()` method of C++ standard containers (`std::vector`) returns `std::size_t`. To avoid compiler warnings about signed/unsigned mismatch, loops bounded by `.size()` should use `size_t`.

**Do this:**
```cpp
// CORRECT
for(size_t i = 0; i < tensor->data.size(); ++i) {
    tensor->data[i] = ...;
}
```

**Don't do this:**
```cpp
// INCORRECT: triggers -Wsign-compare compiler warnings
for(int i = 0; i < tensor->data.size(); ++i) {
    tensor->data[i] = ...;
}
```

*(If you strictly need an `int` for mathematical calculations inside a `size()` loop, you can cast it: `int(tensor->data.size())` or use `int total = tensor->rows * tensor->cols; for(int i = 0; i < total; ++i)`).*
