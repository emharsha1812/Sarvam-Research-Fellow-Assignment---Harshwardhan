# my_einops.py

import numpy as np
from typing import Dict, List, Tuple, Any, Set, Union
import dataclasses
import re
import math

# --- 1. Custom Exception ---
class EinopsError(ValueError):
    """Custom exception for errors during einops operations."""
    pass

# --- 2. Data Structure for Parsed Pattern ---
# (Dataclasses remain the same)
@dataclasses.dataclass
class ParsedExpressionData:
    raw_axes: List[Union[str, List[str]]]
    identifiers: Set[str]
    has_ellipsis: bool
    has_composition: bool
    has_anonymous_axes: bool
    has_trivial_anonymous_axis: bool

@dataclasses.dataclass
class ParsedPattern:
    lhs_expression: ParsedExpressionData
    decomposed_lhs_axes: List[str]
    rhs_expression: ParsedExpressionData
    decomposed_rhs_axes_final: List[str] # Includes placeholders like _repeat_N, _anon_1_, and new named axes
    resolved_axes_lengths: Dict[str, int] # Includes LHS, axes_lengths, inferred, and NEW named axes
    # Consolidate repeat info (numeric literals AND new named axes)
    repeat_axes_info: Dict[str, int] # Map axis name (_repeat_N or 'b') to length
    needs_reshaping_input: bool
    needs_repeating: bool
    needs_transposing: bool
    needs_reshaping_output: bool
    shape_after_lhs_reshape: Tuple[int, ...]
    transpose_indices: Tuple[int, ...]
    final_shape: Tuple[int, ...]

# --- 3. Main Public Function ---
# (Remains the same)
def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths: int) -> np.ndarray:
    """ Rearranges a NumPy ndarray based on the provided einops-style pattern. """
    try:
        _validate_input(tensor, pattern, axes_lengths)
        parsed_pattern = _parse_pattern(pattern, tensor.shape, axes_lengths)
        result = _execute_rearrangement(tensor, parsed_pattern)
        return result
    except EinopsError as e:
        context = f'Error processing pattern "{pattern}"'
        try:
            if isinstance(tensor, np.ndarray): context += f" for input shape {tensor.shape}"
        except: pass
        context += f" with axes_lengths={axes_lengths}."
        raise EinopsError(f"{context}\n -> {e}") from e
    except Exception as e:
        context = f'Unexpected error processing pattern "{pattern}"'
        try:
            if isinstance(tensor, np.ndarray): context += f" for input shape {tensor.shape}"
        except: pass
        context += f" with axes_lengths={axes_lengths}."
        raise RuntimeError(f"{context}\n -> {e.__class__.__name__}: {e}") from e

# --- 4. Internal Helper Functions ---

# --- 4a. Validation (Checker Logic) ---
# (Remains the same)
def _validate_input(tensor: np.ndarray, pattern: str, axes_lengths: Dict[str, int]) -> None:
    """ Performs initial syntax and type checks. """
    if not isinstance(tensor, np.ndarray): raise EinopsError("Input tensor must be a NumPy ndarray.")
    if not isinstance(pattern, str): raise EinopsError("Pattern must be a string.")
    if '->' not in pattern: raise EinopsError("Pattern must contain '->' separator.")
    if pattern.count('->') > 1: raise EinopsError("Pattern must contain exactly one '->' separator.")
    if pattern.count('(') != pattern.count(')'): raise EinopsError(f"Pattern has unbalanced parentheses: '{pattern}'")
    for name, length in axes_lengths.items():
        if not isinstance(name, str) or not name.isidentifier(): raise EinopsError(f"Axis name '{name}' in axes_lengths is not a valid identifier.")
        if name.startswith('_') or name.endswith('_'): raise EinopsError(f"Axis name '{name}' in axes_lengths should not start or end with underscore.")
        if not isinstance(length, int) or length <= 0: raise EinopsError(f"Length for axis '{name}' must be a positive integer, got {length}.")

# --- 4b. Parsing Helpers ---
# (Remains the same)
def _parse_expression(expression_str: str) -> ParsedExpressionData:
    """ Parses one side of the pattern (LHS or RHS) using tokenization. """
    raw_axes: List[Union[str, List[str]]] = []
    identifiers: Set[str] = set()
    has_ellipsis = False; has_composition = False; has_anonymous_axes = False; has_trivial_anonymous_axis = False
    current_composition: List[str] = None; in_composition = False; paren_level = 0
    processed_expression = expression_str.replace('(', ' ( ').replace(')', ' ) ')
    tokens = processed_expression.split()
    for token in tokens:
        if not token: continue
        if token == '...':
            if has_ellipsis: raise EinopsError(f"Pattern side '{expression_str}' has multiple ellipses (...)")
            if paren_level > 0: current_composition.append('...')
            else: raw_axes.append('...')
            has_ellipsis = True
        elif token == '(':
            if paren_level > 0: raise EinopsError(f"Nested parentheses are not allowed in pattern: '{expression_str}'")
            paren_level += 1; in_composition = True; current_composition = []; has_composition = True
        elif token == ')':
            if paren_level == 0: raise EinopsError(f"Unbalanced parentheses (extra closing) in pattern: '{expression_str}'")
            paren_level -= 1; in_composition = False
            if current_composition is not None: raw_axes.append(list(current_composition))
            current_composition = None
        elif token.isidentifier():
            if token.startswith('_') or token.endswith('_'): raise EinopsError(f"Axis name '{token}' should not start or end with underscore.")
            current_scope_identifiers = set(current_composition) if in_composition else identifiers
            if token in current_scope_identifiers: raise EinopsError(f"Duplicate identifier '{token}' in expression '{expression_str}'")
            if in_composition: current_composition.append(token)
            else: raw_axes.append(token)
            identifiers.add(token)
        elif token.isdigit():
            num_val = int(token)
            if num_val <= 0: raise EinopsError(f"Numeric axis must be positive, found '{token}' in '{expression_str}'")
            axis_repr = str(num_val)
            if num_val == 1: has_trivial_anonymous_axis = True
            else: has_anonymous_axes = True
            if num_val != 1:
                 current_scope_identifiers = set(current_composition) if in_composition else identifiers | {ax for ax in raw_axes if isinstance(ax, str)}
                 if axis_repr in current_scope_identifiers: raise EinopsError(f"Duplicate identifier '{axis_repr}' in expression '{expression_str}'")
            if in_composition: current_composition.append(axis_repr)
            else: raw_axes.append(axis_repr)
        else:
            if '.' in token: raise EinopsError("Invalid token '.' found outside ellipsis (...)")
            raise EinopsError(f"Invalid token '{token}' found during parsing of '{expression_str}'")
    if paren_level != 0: raise EinopsError(f"Unbalanced parentheses in pattern: '{expression_str}'")
    processed_raw_axes = []
    for axis_group in raw_axes:
        if isinstance(axis_group, list):
            if len(axis_group) == 1: processed_raw_axes.append(axis_group[0])
            else: processed_raw_axes.append(axis_group)
        else: processed_raw_axes.append(axis_group)
    has_actual_composition = any(isinstance(ax, list) and len(ax) > 1 for ax in processed_raw_axes)
    return ParsedExpressionData(
        raw_axes=processed_raw_axes, identifiers=identifiers, has_ellipsis=has_ellipsis,
        has_composition=has_actual_composition, has_anonymous_axes=has_anonymous_axes,
        has_trivial_anonymous_axis=has_trivial_anonymous_axis
    )

# --- 4c. Main Parsing Logic ---
def _parse_pattern(pattern: str, tensor_shape: Tuple[int, ...], axes_lengths: Dict[str, int]) -> ParsedPattern:
    """ Parses pattern, validates semantics, builds execution plan. """
    lhs_str, rhs_str = pattern.split('->'); lhs_str = lhs_str.strip(); rhs_str = rhs_str.strip()
    lhs_data = _parse_expression(lhs_str); rhs_data = _parse_expression(rhs_str)
    if not lhs_data.has_ellipsis and rhs_data.has_ellipsis: raise EinopsError(f"Ellipsis found in right side, but not left side of pattern '{pattern}'")

    # --- Stage 2: Resolve LHS Dimensions ---
    # (No changes needed in this stage compared to v8)
    resolved_axes_lengths: Dict[str, int] = axes_lengths.copy()
    decomposed_lhs_axes: List[str] = []
    current_dim_index = 0; ellipsis_axes_names: List[str] = []; ellipsis_start_index_in_shape = -1
    if lhs_data.has_ellipsis:
        non_ellipsis_dims_lhs = len([ax for ax in lhs_data.raw_axes if ax != '...'])
        if len(tensor_shape) < non_ellipsis_dims_lhs: raise EinopsError(f"Input tensor has {len(tensor_shape)} dimensions, but pattern requires at least {non_ellipsis_dims_lhs} (excluding ellipsis). Pattern: '{lhs_str}'")
        ellipsis_ndim = len(tensor_shape) - non_ellipsis_dims_lhs
        ellipsis_axes_names = [f"_ellipsis_{i}" for i in range(ellipsis_ndim)]
        try: ellipsis_marker_index_in_raw = lhs_data.raw_axes.index('...'); ellipsis_start_index_in_shape = ellipsis_marker_index_in_raw
        except ValueError: pass
        for i, name in enumerate(ellipsis_axes_names): resolved_axes_lengths[name] = tensor_shape[ellipsis_start_index_in_shape + i]
    temp_dim_index_for_ellipsis = 0; raw_axes_index = 0
    while raw_axes_index < len(lhs_data.raw_axes):
        axis_group = lhs_data.raw_axes[raw_axes_index]; is_ellipsis_group = (axis_group == '...')
        if is_ellipsis_group:
            decomposed_lhs_axes.extend(ellipsis_axes_names); temp_dim_index_for_ellipsis += len(ellipsis_axes_names)
        else:
            if lhs_data.has_ellipsis and raw_axes_index >= ellipsis_start_index_in_shape: current_shape_dim_index = ellipsis_start_index_in_shape + temp_dim_index_for_ellipsis; temp_dim_index_for_ellipsis += 1
            else: current_shape_dim_index = current_dim_index
            if current_shape_dim_index >= len(tensor_shape): raise EinopsError(f"Pattern '{lhs_str}' has more axes than input tensor rank {len(tensor_shape)}")
            dim_size = tensor_shape[current_shape_dim_index]
            if isinstance(axis_group, list): # Composition
                current_composition_axes = []; known_axes = {}; unknown_axes = []; has_one_in_comp = False
                for ax in axis_group:
                    if ax == '1': has_one_in_comp = True; continue
                    if ax.isdigit(): raise EinopsError(f"Numeric literal '{ax}' > 1 is not allowed in LHS composition '{axis_group}'")
                    if ax in resolved_axes_lengths: known_axes[ax] = resolved_axes_lengths[ax]
                    else: unknown_axes.append(ax)
                    current_composition_axes.append(ax)
                if len(unknown_axes) > 1: raise EinopsError(f"Could not infer sizes for multiple axes {unknown_axes} in composition '{axis_group}' from dimension {current_shape_dim_index} with size {dim_size}")
                known_product = math.prod(known_axes.values()) if known_axes else 1
                if has_one_in_comp and len(unknown_axes) == 0 and dim_size != known_product : raise EinopsError(f"Shape mismatch: Dimension {current_shape_dim_index} has size {dim_size}, but composition '{axis_group}' including '1' requires product {known_product}")
                if len(unknown_axes) == 1:
                    unknown_axis = unknown_axes[0]; effective_dim_size_for_calc = dim_size
                    if known_product == 0: raise EinopsError(f"Cannot infer axis size for '{unknown_axis}' when known product is zero in composition '{axis_group}'")
                    if effective_dim_size_for_calc % known_product != 0: raise EinopsError(f"Shape mismatch: Dimension {current_shape_dim_index} of size {effective_dim_size_for_calc} is not divisible by known axes product {known_product} for composition '{axis_group}'")
                    inferred = effective_dim_size_for_calc // known_product
                    if unknown_axis in resolved_axes_lengths and resolved_axes_lengths[unknown_axis] != inferred: raise EinopsError(f"Axis '{unknown_axis}' length mismatch: provided/inferred as {resolved_axes_lengths[unknown_axis]}, calculated as {inferred} from shape.")
                    resolved_axes_lengths[unknown_axis] = inferred; known_axes[unknown_axis] = inferred
                elif len(unknown_axes) == 0:
                    required_product = known_product
                    if dim_size != required_product: raise EinopsError(f"Shape mismatch: Dimension {current_shape_dim_index} has size {dim_size}, but composition {axis_group} requires product {required_product}")
                decomposed_lhs_axes.extend(current_composition_axes)
            elif isinstance(axis_group, str): # Single axis or '1'
                if axis_group == '1':
                    if dim_size != 1: raise EinopsError(f"Shape mismatch: Dimension {current_shape_dim_index} has size {dim_size}, but pattern specifies '1'")
                elif axis_group.isdigit(): raise EinopsError(f"Numeric literal '{axis_group}' > 1 is not allowed on LHS of rearrange pattern.")
                else: # Named axis
                    axis_name = axis_group
                    if axis_name in resolved_axes_lengths and resolved_axes_lengths[axis_name] != dim_size: raise EinopsError(f"Axis '{axis_name}' length mismatch: provided as {resolved_axes_lengths[axis_name]}, dimension {current_shape_dim_index} has size {dim_size}.")
                    resolved_axes_lengths[axis_name] = dim_size; decomposed_lhs_axes.append(axis_name)
            else: raise EinopsError(f"Internal parsing error: Unexpected element '{axis_group}' in parsed LHS.")
            current_dim_index += 1
        raw_axes_index += 1
    num_pattern_groups = len(lhs_data.raw_axes)
    expected_rank = num_pattern_groups - 1 + len(ellipsis_axes_names) if lhs_data.has_ellipsis else num_pattern_groups
    if expected_rank != len(tensor_shape): raise EinopsError(f"Pattern '{lhs_str}' expects rank {expected_rank} but tensor has rank {len(tensor_shape)}.")

    # --- Stage 3: Resolve RHS / Repeats ---
    decomposed_rhs_axes_final: List[str] = []
    # FIX: Initialize repeat_axes_info here
    repeat_axes_info: Dict[str, int] = {} # Handles BOTH numeric AND new named axes
    repeat_counter = 0
    final_shape_list: List[int] = []

    for axis_group in rhs_data.raw_axes:
        if axis_group == '...':
            decomposed_rhs_axes_final.extend(ellipsis_axes_names)
            final_shape_list.extend(resolved_axes_lengths[ax] for ax in ellipsis_axes_names)
        elif isinstance(axis_group, list): # Composition
            group_axes = []; group_len_prod = 1
            for axis_name in axis_group:
                if axis_name == '1':
                    anon_name = f"_anon_1_{len(decomposed_rhs_axes_final) + len(group_axes)}"; group_axes.append(anon_name); group_len_prod *= 1
                elif axis_name.isdigit(): # Numeric literal inside composition
                     repeat_len = int(axis_name); repeat_axis_name = f"_repeat_{repeat_counter}"
                     repeat_axes_info[repeat_axis_name] = repeat_len # Store numeric repeat info
                     group_axes.append(repeat_axis_name); group_len_prod *= repeat_len; repeat_counter += 1
                elif axis_name in resolved_axes_lengths: # Existing axis from LHS
                    group_axes.append(axis_name); group_len_prod *= resolved_axes_lengths[axis_name]
                # FIX: Allow new named axes if length provided
                elif axis_name in axes_lengths:
                     resolved_axes_lengths[axis_name] = axes_lengths[axis_name] # Add to resolved lengths
                     repeat_axes_info[axis_name] = axes_lengths[axis_name] # Add to repeats
                     group_axes.append(axis_name); group_len_prod *= axes_lengths[axis_name]
                else:
                     # This axis is unknown AND not provided in axes_lengths
                     raise EinopsError(f"Unknown axis '{axis_name}' found on RHS composition '{axis_group}' and size not provided.")
            decomposed_rhs_axes_final.extend(group_axes)
            if len(group_axes) > 0: final_shape_list.append(group_len_prod)
        elif isinstance(axis_group, str): # Single axis, '1', or numeric literal
            if axis_group == '1':
                anon_name = f"_anon_1_{len(decomposed_rhs_axes_final)}"
                decomposed_rhs_axes_final.append(anon_name); final_shape_list.append(1)
            elif axis_group.isdigit(): # Numeric literal for repetition
                repeat_len = int(axis_group); repeat_axis_name = f"_repeat_{repeat_counter}"
                repeat_axes_info[repeat_axis_name] = repeat_len # Store numeric repeat info
                decomposed_rhs_axes_final.append(repeat_axis_name); final_shape_list.append(repeat_len); repeat_counter += 1
            else: # Named axis
                axis_name = axis_group
                if axis_name in resolved_axes_lengths: # Existing axis from LHS
                     decomposed_rhs_axes_final.append(axis_name); final_shape_list.append(resolved_axes_lengths[axis_name])
                # FIX: Allow new named axes if length provided
                elif axis_name in axes_lengths:
                     resolved_axes_lengths[axis_name] = axes_lengths[axis_name] # Add to resolved lengths
                     repeat_axes_info[axis_name] = axes_lengths[axis_name] # Add to repeats
                     decomposed_rhs_axes_final.append(axis_name); final_shape_list.append(axes_lengths[axis_name])
                else:
                     # This axis is unknown AND not provided in axes_lengths
                     raise EinopsError(f"Unknown axis '{axis_name}' found on RHS and size not provided.")
        else: raise EinopsError(f"Internal parsing error: Unexpected element '{axis_group}' in parsed RHS.")

    final_shape = tuple(final_shape_list)

    # --- Stage 4: Final Validation & Plan Generation ---
    if lhs_data.has_anonymous_axes: raise EinopsError(f"Numeric literal > 1 is not allowed on LHS of rearrange pattern.")
    rhs_anon_axes_names = {ax for ax_g in rhs_data.raw_axes for ax in (ax_g if isinstance(ax_g, list) else [ax_g]) if isinstance(ax, str) and ax.isdigit() and ax != '1'}
    # Allow anonymous on RHS only if they are part of repeat_axes_info (numeric repeats)
    # Named repeats (like 'b=4') don't contribute to rhs_data.has_anonymous_axes
    if rhs_data.has_anonymous_axes and not any(k.startswith("_repeat_") for k in repeat_axes_info):
         raise EinopsError(f"Non-unitary anonymous axes (numbers > 1) are only allowed on RHS for repetition. Pattern: '{pattern}'")

    # FIX: Relaxed Axis Mismatch Check (allow extra RHS if in axes_lengths)
    lhs_atomic_identifiers = set(decomposed_lhs_axes)
    # Consider all identifiers on RHS except placeholders
    rhs_atomic_identifiers_all = {ax for ax in decomposed_rhs_axes_final if not ax.startswith(('_repeat_', '_anon_1_'))}

    missing_on_rhs = lhs_atomic_identifiers - rhs_atomic_identifiers_all
    extra_on_rhs = rhs_atomic_identifiers_all - lhs_atomic_identifiers

    if missing_on_rhs: # Reduction not allowed in rearrange
        raise EinopsError(f"Axis mismatch: Axes {missing_on_rhs} missing on RHS. Pattern: '{pattern}'")

    # Extra axes on RHS are allowed ONLY if their size was provided in axes_lengths
    # (which should have been added to repeat_axes_info during RHS resolution)
    unspecified_extra_axes = {ax for ax in extra_on_rhs if ax not in repeat_axes_info}
    if unspecified_extra_axes:
        raise EinopsError(f"Axis mismatch: Axes {unspecified_extra_axes} extra on RHS and size not provided in axes_lengths. Pattern: '{pattern}'")
    # --- End of Relaxed Check ---

    # --- Calculate Execution Plan ---
    needs_reshaping_input = lhs_data.has_composition or lhs_data.has_trivial_anonymous_axis
    needs_reshaping_output = rhs_data.has_composition or rhs_data.has_trivial_anonymous_axis or any(ax.startswith('_anon_1_') for ax in decomposed_rhs_axes_final)
    needs_repeating = bool(repeat_axes_info) # True if numeric OR named repeats exist

    shape_after_lhs_reshape = tuple(resolved_axes_lengths[ax] for ax in decomposed_lhs_axes)

    # Transpose indices calculation: Map original LHS axes to their order on RHS *before* new axes are inserted
    source_axes_order = decomposed_lhs_axes
    source_indices = {name: i for i, name in enumerate(source_axes_order)}
    # Target order should only include axes originating from the LHS
    target_order_for_transpose = [ax for ax in decomposed_rhs_axes_final if ax in source_indices]

    transpose_indices = []
    temp_transpose_needed = False
    if set(source_axes_order) != set(target_order_for_transpose): raise EinopsError(f"Internal error: Mismatch between source axes ({set(source_axes_order)}) and target axes for transpose ({set(target_order_for_transpose)}).")
    if source_axes_order != target_order_for_transpose:
        temp_transpose_needed = True
        try: transpose_indices = tuple(source_axes_order.index(ax) for ax in target_order_for_transpose)
        except ValueError as e: raise EinopsError(f"Internal error calculating transpose order: {e}")
    needs_transposing = temp_transpose_needed

    # --- Create ParsedPattern Object ---
    parsed_info = ParsedPattern(
        lhs_expression=lhs_data, decomposed_lhs_axes=decomposed_lhs_axes,
        rhs_expression=rhs_data, decomposed_rhs_axes_final=decomposed_rhs_axes_final,
        resolved_axes_lengths=resolved_axes_lengths, repeat_axes_info=repeat_axes_info,
        needs_reshaping_input=needs_reshaping_input, needs_repeating=needs_repeating,
        needs_transposing=needs_transposing, needs_reshaping_output=needs_reshaping_output,
        shape_after_lhs_reshape=shape_after_lhs_reshape,
        transpose_indices=tuple(transpose_indices),
        final_shape=final_shape,
    )
    return parsed_info


# --- 4d. Execution (Executor Logic) ---
def _execute_rearrangement(tensor: np.ndarray, plan: ParsedPattern) -> np.ndarray:
    """ Executes the rearrangement using NumPy operations based on the parsed plan. """
    current_tensor = tensor
    current_shape = tensor.shape
    try:
        # 1. Initial Reshape
        if plan.needs_reshaping_input:
            target_shape = plan.shape_after_lhs_reshape
            if math.prod(current_shape) != math.prod(target_shape): raise EinopsError(f"Internal error: Cannot reshape {current_shape} to {target_shape}, size mismatch.")
            current_tensor = current_tensor.reshape(target_shape)
            current_shape = current_tensor.shape

        # 2. Transpose (Reorder existing axes before adding new ones)
        if plan.needs_transposing:
            if len(plan.transpose_indices) != len(current_shape): raise EinopsError(f"Internal error: Transpose indices length {len(plan.transpose_indices)} != tensor rank {len(current_shape)}")
            current_tensor = np.transpose(current_tensor, axes=plan.transpose_indices)
            current_shape = current_tensor.shape
            # Tensor axes now match the order required by RHS, excluding new axes

        # 3. Repeat/Add New Axes
        if plan.needs_repeating or any(ax.startswith("_anon_1_") for ax in plan.decomposed_rhs_axes_final):
            temp_tensor = current_tensor
            insert_locations = [] # Store (target_index, axis_name_to_insert)
            # Find where new axes should be inserted based on the final RHS order
            current_axis_index_in_transposed = 0
            for i, final_axis_name in enumerate(plan.decomposed_rhs_axes_final):
                is_new_axis = final_axis_name.startswith(('_repeat_', '_anon_1_')) or (final_axis_name in plan.repeat_axes_info)
                if is_new_axis:
                    insert_locations.append((i, final_axis_name))
                else:
                    current_axis_index_in_transposed += 1 # Existing axis, move to next slot

            # Expand dimensions first, inserting axes from left-to-right
            if insert_locations:
                 # Sort insertions by index to do them correctly
                 insert_locations.sort(key=lambda x: x[0])
                 num_inserted = 0
                 for insert_pos, axis_name in insert_locations:
                      actual_insert_pos = insert_pos # Index in the *final* structure
                      temp_tensor = np.expand_dims(temp_tensor, axis=actual_insert_pos)
                      num_inserted += 1

            # Apply repeats now that axes exist
            current_shape_after_expand = temp_tensor.shape
            repeat_tensor = temp_tensor
            for insert_pos, axis_name in insert_locations:
                 # Repeat numeric literals OR named new axes
                 if axis_name in plan.repeat_axes_info:
                      repeat_len = plan.repeat_axes_info[axis_name]
                      if repeat_len > 1: # No need to repeat if length is 1
                           repeat_tensor = np.repeat(repeat_tensor, repeat_len, axis=insert_pos)
                 # else: _anon_1_ axis already has size 1 from expand_dims

            current_tensor = repeat_tensor
            current_shape = current_tensor.shape

        # 4. Final Reshape
        if plan.needs_reshaping_output or current_shape != plan.final_shape:
            target_shape = plan.final_shape
            if math.prod(current_shape) != math.prod(target_shape): raise EinopsError(f"Internal error: Cannot reshape {current_shape} to {target_shape}, size mismatch before final reshape.")
            current_tensor = current_tensor.reshape(target_shape)
            current_shape = current_tensor.shape

        # Final sanity check
        if current_tensor.shape != plan.final_shape:
           raise EinopsError(f"Internal error: Final shape mismatch. Expected {plan.final_shape}, got {current_shape}")

    except ValueError as ve: raise EinopsError(f"NumPy error during execution: {ve}. Current shape during error: {current_shape}") from ve
    except IndexError as ie:
         op = "transpose" if plan.needs_transposing else "execution"
         raise EinopsError(f"Indexing error during {op}: {ie}. Current shape: {current_shape}. Indices: {plan.transpose_indices if plan.needs_transposing else 'N/A'}") from ie
    return current_tensor


# --- Optional: Example Usage within the module ---
if __name__ == '__main__':
    print("Running example usage:")
    # --- Test Cases ---
    # (run_test function remains the same)
    def run_test(test_name, tensor, pattern, lengths, expected_shape=None, expect_error=None, check_values=None, expected_values=None):
        print(f"\n--- {test_name} ---")
        print(f"Pattern: '{pattern}', Lengths: {lengths}")
        result = None
        try:
            if isinstance(tensor, np.ndarray): print("Input shape:", tensor.shape)
            result = rearrange(tensor, pattern, **lengths)
            if expect_error:
                print(f"!!! ERROR: Expected EinopsError ({expect_error}), but got result.")
                print(f"    Output shape (unexpected): {result.shape}")
            else:
                print(f"Output shape: {result.shape} (Expected: {expected_shape})")
                if expected_shape:
                   assert result.shape == expected_shape, f"Shape mismatch: Got {result.shape}, expected {expected_shape}"
                   print("    Shape check PASSED.")
                if check_values and expected_values is not None:
                    assert np.array_equal(result, expected_values), f"Value mismatch:\nGot:\n{result}\nExpected:\n{expected_values}"
                    print("    Value check PASSED.")
        except EinopsError as e:
            if expect_error:
                print(f"OK: Caught expected EinopsError.")
                if expect_error not in str(e):
                     print(f"    WARN: Error message mismatch. Expected substring '{expect_error}', Got '{e}'")
                else:
                     print(f"    Message contains expected text: '{expect_error}'")
            else:
                print(f"!!! ERROR: Caught unexpected EinopsError: {e}")
        except Exception as e:
             print(f"!!! ERROR: Caught unexpected {type(e).__name__}: {e}")

    # --- Basic Validation Tests ---
    run_test("Invalid Tensor", [1, 2, 3], 'a -> a', {}, expect_error="Input tensor must be a NumPy ndarray")
    run_test("No Separator", np.zeros(1), 'a b c', {}, expect_error="Pattern must contain '->' separator")
    run_test("Multiple Separators", np.zeros(1), 'a -> b -> c', {}, expect_error="Pattern must contain exactly one '->' separator")
    run_test("Invalid Structure", np.zeros(1), 'a . . -> b', {}, expect_error="Invalid token '.'")
    run_test("Unbalanced Parens 1", np.zeros(1), '(a -> b', {}, expect_error="Unbalanced parentheses")
    run_test("Unbalanced Parens 2", np.zeros(1), 'a) -> b', {}, expect_error="Unbalanced parentheses")
    run_test("Invalid axes_lengths Name", np.zeros(1), 'a -> b', {'1b': 2}, expect_error="not a valid identifier")
    run_test("Invalid axes_lengths Value", np.zeros(1), 'a -> b', {'b': 0}, expect_error="must be a positive integer")
    run_test("Invalid axes_lengths Underscore", np.zeros(1), 'a -> b', {'_b': 2}, expect_error="should not start or end with underscore")

    # --- Parsing/Semantic Tests ---
    run_test("Dots outside Ellipsis", np.zeros((2,3)), 'a . b -> a b', {}, expect_error="Invalid token '.'")
    run_test("Multiple Ellipsis LHS", np.zeros((2,3)), '... a ... -> a', {}, expect_error="multiple ellipses")
    run_test("Ellipsis RHS only", np.zeros((2,3)), 'a b -> ... a b', {}, expect_error="Ellipsis found in right side, but not left")
    run_test("Nested Parens", np.zeros((2,3)), 'a (b (c)) -> a b c', {}, expect_error="Nested parentheses")
    run_test("Duplicate ID LHS", np.zeros((2,2)), 'a a -> a', {}, expect_error="Duplicate identifier 'a'")
    run_test("Duplicate ID RHS", np.zeros((2,)), 'a -> a a', {}, expect_error="Duplicate identifier 'a'")
    run_test("Invalid ID Underscore", np.zeros((2,)), '_a -> a', {}, expect_error="should not start or end with underscore")
    run_test("Unknown Axis RHS No Length", np.zeros((2,3)), 'a b -> a c', {}, expect_error="Axes {'c'} extra on RHS and size not provided")
    run_test("Axis Mismatch", np.zeros((2,3)), 'a b -> a', {}, expect_error="Axes {'b'} missing on RHS")
    run_test("Non-unitary Anonymous LHS", np.zeros((2,3)), 'a 2 -> a', {}, expect_error="Numeric literal '2' > 1 is not allowed on LHS")
    run_test("Non-unitary Anonymous RHS (No Repeat)", np.zeros((2,3)), 'a b -> a 2', {}, expect_error="Non-unitary anonymous axes (numbers > 1) are only allowed on RHS for repetition")

    run_test("Split Axis Fail (Div)", np.arange(10).reshape(5, 2), '(h w) c -> h w c', {'h': 3}, expect_error="is not divisible by known axes product")
    run_test("Multiple Unknowns", np.arange(10).reshape(5, 2), '(h w) c -> h w c', {}, expect_error="Could not infer sizes for multiple axes")
    run_test("Shape Rank Mismatch", np.arange(10).reshape(5, 2), 'a b c -> a b c', {}, expect_error="Pattern 'a b c' expects rank 3 but tensor has rank 2")
    run_test("Shape Mismatch Anon 1", np.arange(10).reshape(5, 2), 'a 1 -> a', {}, expect_error="Dimension 1 has size 2, but pattern specifies '1'")
    run_test("Shape Mismatch Composition", np.arange(10).reshape(5, 2), '(h w) c -> h w c', {'h': 2, 'w': 2}, expect_error="Dimension 0 has size 5, but composition ['h', 'w'] requires product 4")

    # --- Tests Expected to Pass ---
    run_test("Simple Reshape", np.arange(12).reshape(3, 4), 'h w -> (h w)', {}, expected_shape=(12,), check_values=True, expected_values=np.arange(12))
    run_test("Simple Reshape 2", np.arange(12), '(h w) -> h w', {'h':3}, expected_shape=(3, 4), check_values=True, expected_values=np.arange(12).reshape(3,4))
    run_test("Transpose", np.arange(12).reshape(3, 4), 'h w -> w h', {}, expected_shape=(4, 3), check_values=True, expected_values=np.arange(12).reshape(3,4).T)
    run_test("Split Axis", np.arange(12).reshape(6, 2), '(h w) c -> h w c', {'h': 3}, expected_shape=(3, 2, 2), check_values=True, expected_values=np.arange(12).reshape(3,2,2))
    run_test("Merge Axes", np.arange(12).reshape(2, 3, 2), 'a b c -> (a b) c', {}, expected_shape=(6, 2), check_values=True, expected_values=np.arange(12).reshape(6,2))
    run_test("Split & Merge", np.arange(24).reshape(12, 2), '(h w) c -> h (w c)', {'h':3}, expected_shape=(3, 8), check_values=True, expected_values=np.arange(24).reshape(3,8))
    run_test("Merge & Split", np.arange(24).reshape(6, 4), '(a b) c -> a (b c)', {'a':2}, expected_shape=(2, 12), check_values=True, expected_values=np.arange(24).reshape(2,12))
    run_test("All Merge", np.arange(24).reshape(2, 3, 4), 'a b c -> (a b c)', {}, expected_shape=(24,), check_values=True, expected_values=np.arange(24))
    run_test("All Split", np.arange(24), '(a b c) -> a b c', {'a':2, 'b':3}, expected_shape=(2, 3, 4), check_values=True, expected_values=np.arange(24).reshape(2,3,4))

    # --- Ellipsis Tests ---
    run_test("Ellipsis Front", np.arange(24).reshape(2,3,4), '... w -> ... w', {}, expected_shape=(2,3,4))
    run_test("Ellipsis Middle", np.arange(24).reshape(2,3,4), 'b ... w -> b ... w', {}, expected_shape=(2,3,4))
    run_test("Ellipsis End", np.arange(24).reshape(2,3,4), 'b h ... -> b h ...', {}, expected_shape=(2,3,4))
    run_test("Ellipsis Transpose 1", np.arange(24).reshape(2,3,4), 'b ... w -> ... w b', {}, expected_shape=(3,4,2))
    run_test("Ellipsis Transpose 2", np.arange(24).reshape(2,3,4), 'b h ... -> ... h b', {}, expected_shape=(4,3,2))
    run_test("Ellipsis Split", np.arange(60).reshape(2,3,10), 'b ... (h w) -> b ... h w', {'h':2}, expected_shape=(2,3,2,5))
    run_test("Ellipsis Merge", np.arange(60).reshape(2,3,2,5), 'b ... h w -> b ... (h w)', {}, expected_shape=(2,3,10))

    # --- Anon Axis '1' Tests ---
    run_test("Anon Axis 1 Squeeze", np.arange(6).reshape(2,1,3), 'a 1 c -> a c', {}, expected_shape=(2,3))
    run_test("Anon Axis 1 Unsqueeze", np.arange(6).reshape(2,3), 'a c -> a 1 c', {}, expected_shape=(2,1,3))
    run_test("Anon Axis 1 Transpose", np.arange(6).reshape(1,2,3), '1 b c -> b 1 c', {}, expected_shape=(2,1,3))
    run_test("Anon Axis 1 Composition LHS", np.arange(3).reshape(3, 1), '(a 1) -> a', {'a': 3}, expected_shape=(3,))
    run_test("Anon Axis 1 Composition RHS", np.arange(6).reshape(2,3), 'a b -> (a 1 b)', {}, expected_shape=(6,))

    # --- Repeat Tests (Numeric and Named) ---
    print("\n--- REPEAT TESTS ---")
    run_test("Repeat Literal Simple", np.array([1,2]), 'a -> a 3', {}, expected_shape=(2,3), check_values=True, expected_values=np.array([[1,1,1],[2,2,2]]))
    run_test("Repeat Literal End", np.array([[1,2],[3,4]]), 'a b -> a b 2', {}, expected_shape=(2,2,2), check_values=True, expected_values=np.array([[[1,1],[2,2]],[[3,3],[4,4]]]))
    run_test("Repeat Literal Middle", np.array([[1,2],[3,4]]), 'a b -> a 2 b', {}, expected_shape=(2,2,2), check_values=True, expected_values=np.array([[[1,2],[1,2]],[[3,4],[3,4]]]))
    run_test("Repeat Literal Start", np.array([[1,2],[3,4]]), 'a b -> 3 a b', {}, expected_shape=(3,2,2), check_values=True, expected_values=np.array([[[1,2],[3,4]],[[1,2],[3,4]],[[1,2],[3,4]]]))
    run_test("Repeat Literal Transpose", np.array([[1,2],[3,4]]), 'a b -> b 2 a', {}, expected_shape=(2,2,2), check_values=True, expected_values=np.array([[[1,1],[3,3]],[[2,2],[4,4]]]))
    run_test("Repeat Literal Merge", np.array([[1,2],[3,4]]), 'a b -> (a 2 b)', {}, expected_shape=(8,), check_values=True, expected_values=np.array([1,2,1,2,3,4,3,4]))
    run_test("Repeat Literal Multiple", np.array([1,2]), 'a -> 2 a 3', {}, expected_shape=(2,2,3), check_values=True, expected_values=np.array([[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]]]))
    # This one should now pass because we allow new named axes if length is provided
    run_test("Repeat Named Axis", np.array([[1,2],[3,4]]), 'a b -> a rpt b', {'rpt': 2}, expected_shape=(2,2,2), check_values=True, expected_values=np.array([[[1,2],[1,2]],[[3,4],[3,4]]]))
    # User's original failing test case - should pass now
    run_test("Repeat User Case", np.random.rand(3, 1, 5), 'a 1 c -> a b c', {'b': 4}, expected_shape=(3,4,5))
    print("\n--- REPEAT TESTS ---")
    run_test("Repeat Literal Simple", np.array([1,2]), 'a -> a 3', {}, expected_shape=(2,3), check_values=True, expected_values=np.array([[1,1,1],[2,2,2]]))
    run_test("Repeat Literal End", np.array([[1,2],[3,4]]), 'a b -> a b 2', {}, expected_shape=(2,2,2), check_values=True, expected_values=np.array([[[1,1],[2,2]],[[3,3],[4,4]]]))
    run_test("Repeat Literal Middle", np.array([[1,2],[3,4]]), 'a b -> a 2 b', {}, expected_shape=(2,2,2), check_values=True, expected_values=np.array([[[1,2],[1,2]],[[3,4],[3,4]]]))
    run_test("Repeat Literal Start", np.array([[1,2],[3,4]]), 'a b -> 3 a b', {}, expected_shape=(3,2,2), check_values=True, expected_values=np.array([[[1,2],[3,4]],[[1,2],[3,4]],[[1,2],[3,4]]]))
    run_test("Repeat Literal Transpose", np.array([[1,2],[3,4]]), 'a b -> b 2 a', {}, expected_shape=(2,2,2), check_values=True, expected_values=np.array([[[1,1],[3,3]],[[2,2],[4,4]]])) # Fixed expected value
    run_test("Repeat Literal Merge", np.array([[1,2],[3,4]]), 'a b -> (a 2 b)', {}, expected_shape=(8,), check_values=True, expected_values=np.array([1,2,1,2,3,4,3,4]))
    run_test("Repeat Literal Multiple", np.array([1,2]), 'a -> 2 a 3', {}, expected_shape=(2,2,3), check_values=True, expected_values=np.array([[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]]]))
    # Should pass now
    run_test("Repeat Named Axis", np.array([[1,2],[3,4]]), 'a b -> a rpt b', {'rpt': 2}, expected_shape=(2,2,2), check_values=True, expected_values=np.array([[[1,2],[1,2]],[[3,4],[3,4]]]))
    # Should pass now
    run_test("Repeat User Case", np.random.rand(3, 1, 5), 'a 1 c -> a b c', {'b': 4}, expected_shape=(3,4,5))

    print("\n--- Test Execution Finished ---")

