# Sarvam-Research-Fellow-Assignment---Harshwardhan

Repository for Sarvam Research Fellowship Application - Implementing Einops from scratch

# Einops Rearrange Implementation (NumPy)

## 1. Overview

A from-scratch implementation of the core `rearrange` operation inspired by the `einops` library.
The primary goal is to replicate the flexible tensor manipulation capabilities of `einops.rearrange` using only Python and NumPy, without importing the original `einops` library.

Currently it supports only numpy ndarray

## 2. Core Functionality

The `rearrange` function is written such as we can manipulate NumPy ndarrays using intuitive, readable pattern strings. Supported operations include:

- **Reshaping:** Changing tensor dimensions.
- **Transposition:** Reordering axes.
- **Splitting:** Dividing an axis into multiple new axes.
- **Merging:** Combining multiple axes into one.
- **Repeating:** Duplicating data along a new or existing dimension.

## 3. Implementation Approach

The core logic follows a two-stage pipeline designed for robustness:

1.  **Checker (Validation Stage):**

    - This initial stage receives the user's input (`tensor`, `pattern`, `**axes_lengths`).
    - It performs crucial upfront validation:
      - Checks if the input `tensor` is a NumPy ndarray.
      - Validates the basic syntax of the `pattern` string.
      - Performs preliminary checks on dimension compatibility (e.g., number of axes vs. pattern elements).
      - Ensures provided `axes_lengths` are syntactically valid.
    - **Crucially, if any validation fails, this stage raises a detailed, informative `Exception` immediately, preventing further processing.**

2.  **Executor (Processing Stage):**
    - This stage only runs if the Checker stage passes successfully.
    - It invokes a **Parser** to deeply analyze the `pattern` string, identifying specific operations (split, merge, transpose, repeat) and axis relationships.
    - It performs **semantic validation** that requires combining information from the tensor's shape, the parsed pattern, and `axes_lengths` (e.g., checking if an axis length is divisible for a split operation).
    - It calculates the target shape and determines the precise sequence of NumPy operations (`reshape`, `transpose`, `repeat`/`tile`, etc.) needed.
    - It executes these NumPy operations to transform the tensor.
    - It returns the final, rearranged NumPy ndarray.

<!-- rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray
tensor (np.ndarray): The input NumPy array to manipulate.pattern (str): The einops-style pattern defining the transformation (e.g., 'b h w c -> b c h w').**axes_lengths (Dict[str, int]): Optional keyword arguments specifying the sizes of new axes introduced on the right side of the pattern or dimensions involved in splitting.5. Example Usageimport numpy as np
# Assuming your implementation is in 'my_einops.py'
# from my_einops import rearrange -->

### Workflow Diagram

![Workflow Diagram](diagram.svg)

_(This diagram illustrates the high-level flow. The Executor stage internally handles parsing and detailed semantic checks.)_

## 4. Main Function

### `rearrange`

```python

# --- Placeholder function for demonstration ---
def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    print(f"Rearranging with pattern: {pattern}, lengths: {axes_lengths}")
    # In the real implementation, actual rearrangement happens here
    # Returning input shape for demonstration of call
    print(f"Input shape: {tensor.shape}")
    # This is just a dummy return, replace with actual result
    return tensor # Replace with actual rearranged tensor

# --- Examples ---

# Transpose
print("--- Transpose ---")
x_t = np.random.rand(3, 4)
result_t = rearrange(x_t, 'h w -> w h')
# Expected result shape: (4, 3)

# Split an axis
print("\n--- Split Axis ---")
x_s = np.random.rand(12, 10)
result_s = rearrange(x_s, '(h w) c -> h w c', h=3)
# Expected result shape: (3, 4, 10)

# Merge axes
print("\n--- Merge Axes ---")
x_m = np.random.rand(3, 4, 5)
result_m = rearrange(x_m, 'a b c -> (a b) c')
# Expected result shape: (12, 5)

# Repeat an axis
print("\n--- Repeat Axis ---")
x_r = np.random.rand(3, 1, 5)
result_r = rearrange(x_r, 'a 1 c -> a b c', b=4)
# Expected result shape: (3, 4, 5)

# Handle batch dimensions (...)
print("\n--- Ellipsis ---")
x_e = np.random.rand(2, 3, 4, 5)
result_e = rearrange(x_e, '... h w -> ... (h w)')
# Expected result shape: (2, 3, 20)
```

# Install dependencies

pip install numpy
