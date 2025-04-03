The objective of the test is to implement the `rearrange` operation of the einops library

This function should support the following operations

- [ ] Reshaping of a tensor
- [ ] Transpose of a tensor
- [ ] Splitting of an axis of a tensor
- [ ] Merging of an axis of a tensor
- [ ] Repeating of an axis of a tensor

### Obj

So here is what i have to do
From the Einops library i have to implement the rearrange and repeat functions
For that i also have to build a good parser that can understand and parse and then perform those operations
Along with that I have to take care of test cases, edge cases and maybe build a good feedback system on errors
Documentation should be upto date too
Keeping in mind that my implementation need to only work with Numpy Ndarrays at the moment, others can wait
Also in Einops both rearrange and repeat are different implementations
but for my use case i need to keep both of them as same

Here is what i am planning to do
I will create a class called Rearrange and inside it i will define
different methods for that class. these different methods will implement different things that are mentioned such as

1. Reshaping
2. Transpose
3. Splitting
4. Merging of axis
5. Repeating of axis

But how will the class know what i want to do?
I guess i have to handle that in the parser.
So the objective along with that will be to build an efficient parser that can parse and then give instructins to perform those operations. ( maybe i should checkout how einops does it)

##### tips

I think it would be good if i time the performances of both einops and my own implementation
maybe if i can get mine to work faster it would be amazing
the original parserClass in the einops library does not use Regex instead it uses manual logic, (sigh that would be so hard to code)
instead what we can do is use regex or some other string parsing library that maybe can potentially make it easier for us

Also maybe for errors i should first check the dimensions
I think i will keep the flow like this
So a query comes and it first gets verified if its possible or not( dimensionalirty check) -- > next it goes to some helper function which invokes the neccessary functions and methods to make it possible (parser and other) and it will be the one who returns it finally to the user. It also checks for parsing errors --> then if everything is alright then it goes to the function which actually does all the work --> then that function returns the list directly to the user

#### Additives

Apart from the mentioned, I would also try to add the features that are currently NOT part of the einops libary. For that I went through the 'Issues' section of the repository and listed all the features suggestions and tried to implement them one by one

- [ ] [Match arbitrary number of dimensions “a \*b”](https://github.com/arogozhnikov/einops/issues/369)
- [ ]

Day 1 : I am going through the extensive documentation of Einops library and checking why the library is useful, in what way it speeds up things.
I am also looking at how they have implemented their famous functions like rearrange, reduce. My main target is rearrange function so i will use that.
Also motivation for why this library exists i.e which Einstein notation is popular

##### Reason No 1 : Much more verbose and direct

y = x.view(x.shape[0], -1)
y = rearrange(x, 'b c h w -> b (c h w)')

#### Important Notations

#### How rearrange works in the Einops

To explain how the `rearrange` function works in `einops` and the functions it uses, I'll create a workflow chart in textual format:

### Workflow Chart for `rearrange` Function

1. **`rearrange` Function Call**

   - **Location:** `einops/array_api.py`
   - **Signature:** `def rearrange(tensor: Tensor, pattern: str, **axes_lengths) -> Tensor`
   - **Function:** Calls `reduce` with `reduction="rearrange"`

2. **`reduce` Function Call**

   - **Location:** `einops/array_api.py`
   - **Signature:** `def reduce(tensor: Tensor, pattern: str, reduction: Reduction, **axes_lengths: int) -> Tensor`
   - **Function:** Prepares the transformation recipe and applies it.

3. **`_prepare_transformation_recipe` Function Call**

   - **Location:** `einops/einops.py`
   - **Signature:** `def _prepare_transformation_recipe(pattern: str, reduction: Reduction, axes_names: Tuple[str, ...], ndim: int) -> TransformRecipe`
   - **Function:** Parses the pattern string and prepares a `TransformRecipe`.

4. **`_apply_recipe_array_api` Function Call**
   - **Location:** `einops/einops.py`
   - **Signature:** `def _apply_recipe_array_api(xp, recipe: TransformRecipe, tensor: Tensor, reduction_type: Reduction, axes_lengths: HashableAxesLengths) -> Tensor`
   - **Function:** Applies the `TransformRecipe` to the tensor.

### Detailed Steps

1. **rearrange(tensor, pattern, \*\*axes_lengths)**

   - Called by the user to rearrange a tensor.
   - Calls `reduce(tensor, pattern, reduction="rearrange", **axes_lengths)`.

2. **reduce(tensor, pattern, "rearrange", \*\*axes_lengths)**

   - Prepares the transformation recipe by calling `_prepare_transformation_recipe(pattern, "rearrange", tuple(axes_lengths), tensor.ndim)`.
   - Applies the recipe to the tensor by calling `_apply_recipe_array_api(xp, recipe, tensor, "rearrange", tuple(axes_lengths))`.

3. **\_prepare_transformation_recipe(pattern, "rearrange", axes_names, ndim)**

   - Parses the pattern string into `ParsedExpression` objects.
   - Validates the pattern and checks for consistency.
   - Creates a `TransformRecipe` that includes:
     - Initial and final shapes.
     - Axes reordering.
     - Reduced and added axes.

4. **\_apply_recipe_array_api(xp, recipe, tensor, "rearrange", axes_lengths)**
   - Reshapes the tensor according to the initial shapes in the recipe.
   - Permutes the dimensions of the tensor as specified.
   - Applies any reduction operations if needed.
   - Expands the dimensions and broadcasts the tensor if new axes are added.
   - Reshapes the tensor to its final shape as specified in the recipe.

### Dependencies and Interactions

- **`ParsedExpression` Class**

  - Parses and validates the pattern string.
  - Used by `_prepare_transformation_recipe`.

- **`TransformRecipe` Class**
  - Represents the transformation recipe.
  - Created by `_prepare_transformation_recipe`.

### Summary

The `rearrange` function in `einops` works by calling the `reduce` function with a specific reduction type. The `reduce` function prepares a transformation recipe using `_prepare_transformation_recipe` and applies this recipe using `_apply_recipe_array_api`. The process involves parsing the pattern, validating it, and transforming the tensor step-by-step according to the recipe.

You can refer to the following files for the complete implementation:

- [`einops/array_api.py`](https://github.com/arogozhnikov/einops/blob/main/einops/array_api.py)
- [`einops/einops.py`](https://github.com/arogozhnikov/einops/blob/main/einops/einops.py)

#### TLDR of above

1. The current implementation—whether in the standard API or array‑API variant—typically follows these steps:

2. Initial Reshape: Converts the input tensor into an intermediate shape that separates grouped axes.

3. Transpose (Permutation): Reorders the dimensions as specified by the parsed pattern.

4. Final Reshape: Combines or splits dimensions to produce the final desired shape.

Here are common error conditions and messages found in the original `einops` library, primarily raised as `EinopsError`:

**1. Basic Pattern Syntax Errors (Likely in `_validate_input` or early `_parse_pattern`)**

- **Missing Separator:** Pattern doesn't contain `->`.
  - `einops` Check: `if '->' not in pattern:` (`_validate_input` in your blueprint) or `pattern.split('->')` fails (`_parse_pattern`).
  - Example Message: `"Pattern must contain '->' separator."` (or similar from `split` failure).
- **Invalid Characters:** Characters other than alphanumeric, underscore, parentheses, space, ellipsis are used.
  - `einops` Check: Iterates through characters in `ParsedExpression`.
  - Example Message: `"Unknown character '{}'"`
- **Unbalanced Parentheses:** Mismatched `(` and `)`.
  - `einops` Check: Counts `(` vs `)` (`_validate_input`) and checks state during parsing (`ParsedExpression`).
  - Example Message: `"Pattern has unbalanced parentheses."` or `"Brackets are not balanced"` or `"Imbalanced parentheses in expression: \"{}\""`

**2. Ellipsis Misuse (Likely in `_parse_pattern`)**

- **Dots Outside Ellipsis:** Using `.` characters not part of `...`.
  - `einops` Check: Checks `.` count vs `...` count in `ParsedExpression`.
  - Example Message: `"Expression may contain dots only inside ellipsis (...)"`
- **Multiple Ellipses:** More than one `...` in the expression.
  - `einops` Check: Counts `...` in `ParsedExpression`.
  - Example Message: `"Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor "`
- **Ellipsis on RHS only:** `...` appears on the right side but not the left.
  - `einops` Check: `if not left.has_ellipsis and rght.has_ellipsis:` in `_prepare_transformation_recipe`.
  - Example Message: `"Ellipsis found in right side, but not left side of a pattern {}"`
- **Ellipsis in Parentheses (LHS):** `(...)` on the left side is disallowed.
  - `einops` Check: `if left.has_ellipsis and left.has_ellipsis_parenthesized:` in `_prepare_transformation_recipe`.
  - Example Message: `"Ellipsis inside parenthesis in the left side is not allowed: {}"`

**3. Axis Identifier Errors (Likely in `_parse_pattern`)**

- **Invalid Identifier:** Axis name is not a valid Python identifier, or starts/ends with `_` (unless it _is_ `_` and allowed).
  - `einops` Check: Uses `str.isidentifier()` and custom checks in `ParsedExpression.check_axis_name_return_reason`.
  - Example Message: `"Invalid axis identifier: {}\n{}"` (includes reason).
- **Duplicate Identifier:** Same axis name used multiple times on one side (LHS or RHS).
  - `einops` Check: Uses sets (`self.identifiers`) in `ParsedExpression`.
  - Example Message: `"Indexing expression contains duplicate dimension \"{}\""`
- **Invalid Anonymous Axis:** Using `1` or non-positive integers for anonymous axes.
  - `einops` Check: In `AnonymousAxis.__init__`.
  - Example Message: `"No need to create anonymous axis of length 1..."` or `"Anonymous axis should have positive length, not {}"`
- **Non-unitary Anonymous Axes (Rearrange):** Using numbers other than `1` in `rearrange`.
  - `einops` Check: `if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:` in `_prepare_transformation_recipe`.
  - Example Message: `"Non-unitary anonymous axes are not supported in rearrange (exception is length 1)"`

**4. Composition / Decomposition Errors (Likely in `_parse_pattern`)**

- **Nested Parentheses:** Patterns like `a (b (c d)) e`.
  - `einops` Check: Tracks nesting level in `ParsedExpression`.
  - Example Message: `"Axis composition is one-level (brackets inside brackets not allowed)"`
- **Multiple Unknowns in Composition:** Cannot infer dimensions if multiple axes within a single composition `(a b c)` are unknown.
  - `einops` Check: `if len(unknown) > 1:` in `_prepare_transformation_recipe`.
  - Example Message: `"Could not infer sizes for {}"`
- **Shape Mismatch (Divisibility):** An axis cannot be evenly split according to the pattern and known dimensions.
  - `einops` Check: `if length % known_product != 0:` in `_reconstruct_from_shape_uncached`.
  - Example Message: `"Shape mismatch, can't divide axis of length {} in chunks of {}"`
- **Shape Mismatch (Exact):** Product of known axes in a composition doesn't match the input dimension length when no unknowns are present.
  - `einops` Check: `if length != known_product:` in `_reconstruct_from_shape_uncached`.
  - Example Message: `"Shape mismatch, {} != {}"`

**5. Semantic / Consistency Errors (Likely in `_parse_pattern`)**

- **Axis Mismatch (Rearrange):** Axes found only on LHS or RHS for `rearrange`.
  - `einops` Check: `set.symmetric_difference(left.identifiers, rght.identifiers)` in `_prepare_transformation_recipe`.
  - Example Message: `"Identifiers only on one side of expression (should be on both): {}"`
- **Axis Mismatch (Repeat/Reduce):** Checks for unexpected axes on LHS (repeat) or RHS (reduce).
  - `einops` Check: `set.difference(...)` in `_prepare_transformation_recipe`.
  - Example Messages: `"Unexpected identifiers on the left side of repeat: {}"` or `"Unexpected identifiers on the right side of reduce {}: {}"`
- **Missing Axis Size (Repeat):** New axes introduced on RHS of `repeat` don't have their size specified in `axes_lengths`.
  - `einops` Check: Checks `rght.identifiers` against `left.identifiers` and `axes_names` in `_prepare_transformation_recipe`.
  - Example Message: `"Specify sizes for new axes in repeat: {}"`
- **Dimension Number Mismatch:** Number of dimensions in tensor doesn't match the number specified in the pattern (considering ellipsis).
  - `einops` Check: Compares `ndim` vs `len(left.composition)` in `_prepare_transformation_recipe`.
  - Example Message: `"Wrong shape: expected {} dims. Received {}-dim tensor."` (or similar message for ellipsis case).

**6. `axes_lengths` Argument Errors (Likely in `_validate_input` or `_parse_pattern`)**

- **Invalid Name:** Key in `axes_lengths` is not a valid identifier.
  - `einops` Check: `isinstance(name, str) or not name.isidentifier()` (`_validate_input` in your blueprint).
  - Example Message: `"Axis name '{}' in axes_lengths is not a valid identifier."`
- **Invalid Value:** Value in `axes_lengths` is not a positive integer.
  - `einops` Check: `isinstance(length, int) or length <= 0` (`_validate_input` in your blueprint).
  - Example Message: `"Length for axis '{}' must be a positive integer, got {}."`
- **Unused Axis Name:** An axis name provided in `axes_lengths` doesn't appear in the pattern where it's needed (e.g., for a new axis on RHS).
  - `einops` Check: `if elementary_axis not in axis_name2known_length:` in `_prepare_transformation_recipe`.
  - Example Message: `"Axis {} is not used in transform"`

**Recommendation:**

You can define similar checks within your `_validate_input` and `_parse_pattern` functions. When a check fails, raise your custom `EinopsError` with a descriptive message, potentially mirroring some of the messages above. This makes debugging much easier for the user. The main `rearrange` function in `einops` also wraps the internal calls in a try-except block to add more context (like the full pattern, input shape, and axes_lengths) to any `EinopsError` raised from within. You might consider doing the same.
