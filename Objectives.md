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
