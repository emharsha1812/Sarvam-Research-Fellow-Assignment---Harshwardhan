# Einops Rearrange Implementation (NumPy) - Sarvam Research Fellowship Assignment

## 1. Overview

This project presents a Python implementation of the `rearrange` operation, drawing inspiration from the `einops` library. The implementation currently targets `numpy.ndarray` objects exclusively

## 2. Functionality

The core of this module is the `rearrange` function:

```python
def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
```

This function accepts a NumPy tensor, a pattern string describing the rearrangement, and optional keyword arguments specifying the lengths of newly introduced axes.

It supports the following tensor manipulations through the pattern string:

- **Transposition:** Reordering existing axes by changing their position between the left-hand side (LHS) and right-hand side (RHS) of the `->` separator.
- **Reshaping (Merging Axes):** Combining multiple axes from the LHS into a single axis on the RHS using parentheses `()`. For example, `a b c -> (a b) c`.
- **Reshaping (Splitting an Axis):** Dividing a single axis from the LHS into multiple axes on the RHS using parentheses `()`. The lengths of the new axes are inferred or specified via `axes_lengths`. For example, `(h w) c -> h w c`.
- **Adding Dimensions:** Introducing new axes of length 1 by placing `1` on the RHS. For example, `h w -> h 1 w`.
- **Removing Dimensions:** Removing axes of length 1 by omitting the `1` from the RHS (or including it within a composition on the LHS). For example, `h 1 w -> h w`.
- **Repeating Axes:** Introducing new axes with specified lengths on the RHS. This can be done using numeric literals (e.g., `h w -> h w 3` repeats the last dimension 3 times) or named axes whose lengths are provided in `axes_lengths` (e.g., `h w -> h w b`, where `b=4` is passed) [cite: EI/my_einops.py].
- **Ellipsis Handling:** Using `...` to represent any number of dimensions not explicitly mentioned in the pattern, allowing operations on specific trailing or leading dimensions regardless of the tensor's rank [cite: EI/my_einops.py].

## 3. Pattern String Syntax

The `pattern` string follows conventions inspired by `einops`:

- **Separator:** `->` divides the pattern into a left-hand side (LHS) describing the input tensor's axes and a right-hand side (RHS) describing the output tensor's axes.
- **Identifiers:** Space-separated names (e.g., `batch`, `height`, `width`, `channels`) represent tensor axes. Each unique identifier must correspond to a dimension of the same size wherever it appears on the LHS and RHS (unless part of a composition).
- **Composition:** Parentheses `()` group axes for merging on the RHS (e.g., `(h w)`) or splitting on the LHS (e.g., `(h w)`). Nested parentheses are not supported
- **Ellipsis:** `...` represents one or more dimensions not explicitly named. It can appear at most once on the LHS and, if present, must also appear on the RHS
- **Anonymous Axes (Length 1):** The number `1` can be used to add or remove dimensions of size one.
- **Repetition Axes:** Numbers greater than 1 on the RHS indicate repetition (e.g., `h w -> h w 3`). New named axes on the RHS whose lengths are provided in `axes_lengths` also function as repetition axes (e.g., `h w -> h rep w` with `rep=4`).
- **`axes_lengths`:** A dictionary providing integer lengths for axes that are newly introduced on the RHS (named repetitions) or axes involved in LHS splitting where their size cannot be inferred from the input tensor dimension alone

## 4. Implementation Details & Workflow

The workflow diagram is as follows:

![Workflow Diagram](diagram.svg) _(Diagram illustrating the processing stages)_

1.  **Initial Validation (`Checker` Stage):**

        - Mapped to the `_validate_input` function [cite: EI/my_einops.py].
        - Performs preliminary checks on the inputs _before_ attempting complex parsing or execution.
        - Verifies the tensor is a `numpy.ndarray`.
        - Checks for basic pattern string well-formedness (contains one `->`, balanced parentheses).
        - Validates the format of keys and values provided in `axes_lengths` (identifiers, positive integers).

> Purpose: Fail fast with clear errors for fundamental input mistakes, preventing wasted computation. Errors raised: `EinopsError`.

1.  **Parsing, Semantic Validation & Planning (`Executor Stage 1: Semantic Validator`):**

        - Mapped to the `_parse_pattern` function [cite: EI/my_einops.py].
        - Performs in-depth analysis of the pattern string in relation to the input tensor's shape and `axes_lengths`.
        - Parses LHS and RHS expressions, identifying identifiers, compositions, ellipsis, and anonymous/repeat axes.
        - Resolves axis lengths: Matches LHS axes to tensor dimensions, infers lengths where possible (e.g., in compositions), and incorporates provided `axes_lengths`.
        - Performs _semantic_ validation:
          - Checks for consistency of axis lengths across LHS and RHS.
          - Verifies tensor rank compatibility with the pattern.
          - Ensures dimension sizes are divisible for splitting operations.
          - Validates rules for ellipsis and anonymous/repeat axes (e.g., numbers > 1 only allowed on RHS for repeat).
          - Confirms that all axes on LHS appear on RHS for `rearrange`, and any new axes on RHS have lengths defined
        - Generates an execution plan (`ParsedPattern` object) detailing the required sequence of NumPy operations (reshapes, transpositions, repeats), target shapes, and indices

> Purpose: Ensure the requested rearrangement is logically possible and mathematically consistent with the input tensor _before_ attempting execution. Errors raised: `EinopsError`.

1.  **Execution (`Executor Stage 2: Operator`):** - Mapped to the `_execute_rearrangement` function [cite: EI/my_einops.py]. - Takes the original tensor and the validated execution plan (`ParsedPattern`). - Performs the sequence of NumPy operations as defined in the plan: - Initial reshape based on LHS composition/anonymous axes. - Transpose axes according to the reordering specified between LHS and RHS. - Insert and repeat new dimensions based on numeric literals or named axes from `axes_lengths`. - Final reshape based on RHS composition/anonymous axes to achieve the target output shape.
    > Purpose: Efficiently execute the planned tensor manipulations using NumPy backend operations. Errors raised: Catches potential NumPy errors during execution (e.g., unexpected shape mismatch despite checks, though unlikely) and wraps them in `EinopsError`.

## 5. Error Handling

A custom exception class, `EinopsError(ValueError)`, is used for all validation and execution errors related to the rearrangement logic ]. The goal is to provide informative error messages that include context, such as the pattern string, input shape, and the specific reason for the failure, aiding in debugging. Errors are categorized based on the stage they occur in (input validation, semantic validation, execution).

## 6. Usage Examples

```python
import numpy as np
from my_einops import rearrange

# --- Examples ---

# Transpose
print("--- Transpose ---")
x_t = np.random.rand(3, 4)
# result_t = rearrange(x_t, 'h w -> w h')
print(f"Input shape: {x_t.shape}, Pattern: 'h w -> w h'")
# Expected result shape: (4, 3)

# Split an axis
print("\n--- Split Axis ---")
x_s = np.random.rand(12, 10)
# result_s = rearrange(x_s, '(h w) c -> h w c', h=3)
print(f"Input shape: {x_s.shape}, Pattern: '(h w) c -> h w c', h=3")
# Expected result shape: (3, 4, 10)

# Merge axes
print("\n--- Merge Axes ---")
x_m = np.random.rand(3, 4, 5)
# result_m = rearrange(x_m, 'a b c -> (a b) c')
print(f"Input shape: {x_m.shape}, Pattern: 'a b c -> (a b) c'")
# Expected result shape: (12, 5)

# Repeat an axis (using named axis length)
print("\n--- Repeat Axis (Named) ---")
x_r_named = np.random.rand(3, 1, 5)
# result_r_named = rearrange(x_r_named, 'a 1 c -> a b c', b=4)
print(f"Input shape: {x_r_named.shape}, Pattern: 'a 1 c -> a b c', b=4")
# Expected result shape: (3, 4, 5)

# Repeat an axis (using numeric literal)
print("\n--- Repeat Axis (Numeric) ---")
x_r_num = np.random.rand(3, 5)
# result_r_num = rearrange(x_r_num, 'a c -> a 4 c')
print(f"Input shape: {x_r_num.shape}, Pattern: 'a c -> a 4 c'")
# Expected result shape: (3, 4, 5)

# Handle batch dimensions (...)
print("\n--- Ellipsis ---")
x_e = np.random.rand(2, 3, 4, 5)
# result_e = rearrange(x_e, '... h w -> ... (h w)')
print(f"Input shape: {x_e.shape}, Pattern: '... h w -> ... (h w)'")
# Expected result shape: (2, 3, 20)


```

## 7. Dependencies

This implementation requires only NumPy

```bash
pip install numpy
```
