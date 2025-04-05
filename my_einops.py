import numpy as np
from typing import Dict, List, Tuple, Any, Set, Union
import dataclasses
import math
import pytest

# --- 1. Custom Exception ---
class EinopsError(ValueError):
    """
    Custom exception class for errors encountered during einops-style operations.
    Inherits from ValueError for semantic grouping of value-related issues.
    """
    pass

# --- 2. Data Structures for Parsed Pattern ---
@dataclasses.dataclass
class ParsedExpressionData:
    """
    Holds parsed information about one side (LHS or RHS) of the einops pattern.
    """
    raw_axes: List[Union[str, List[str]]]
    identifiers: Set[str]
    has_ellipsis: bool
    has_composition: bool
    has_anonymous_axes: bool
    has_trivial_anonymous_axis: bool

@dataclasses.dataclass
class ParsedPattern:
    """
    Represents the fully parsed and validated einops pattern, containing
    the execution plan for the rearrangement.
    """
    lhs_expression: ParsedExpressionData
    decomposed_lhs_axes: List[str]
    rhs_expression: ParsedExpressionData
    decomposed_rhs_axes_final: List[str]
    resolved_axes_lengths: Dict[str, int]
    repeat_axes_info: Dict[str, int]
    needs_reshaping_input: bool
    needs_repeating: bool
    needs_transposing: bool
    needs_reshaping_output: bool
    shape_after_lhs_reshape: Tuple[int, ...]
    transpose_indices: Tuple[int, ...]
    final_shape: Tuple[int, ...]

# --- 3. Main Public Function ---
def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths: int) -> np.ndarray:
    """
    Rearranges a NumPy ndarray based on the provided einops-style pattern.
    """
    try:
        # 1. Perform basic input validation (types, pattern structure)
        _validate_input(tensor, pattern, axes_lengths)

        # 2. Parse the pattern, validate semantics, and create an execution plan
        parsed_pattern = _parse_pattern(pattern, tensor.shape, axes_lengths)

        # 3. Execute the plan using NumPy operations
        result = _execute_rearrangement(tensor, parsed_pattern)

        return result
    except EinopsError as e:
        raise e
    except Exception as e:
        try:
            if isinstance(tensor, np.ndarray):
                context += f" for input shape {tensor.shape}"
        except:
            pass
        context += f" with axes_lengths={axes_lengths}."
        raise RuntimeError(f"{context}\n -> {e.__class__.__name__}: {e}") from e

# --- 4. Internal Helper Functions ---

# --- 4a. Validation (Checker Logic) ---
def _validate_input(tensor: np.ndarray, pattern: str, axes_lengths: Dict[str, int]) -> None:
    """
    Performs initial syntax and type checks on the inputs.
    """
    if not isinstance(tensor, np.ndarray):
        raise EinopsError("Input tensor must be a NumPy ndarray.")
    if not isinstance(pattern, str):
        raise EinopsError("Pattern must be a string.")
    if '->' not in pattern:
        raise EinopsError("Pattern must contain '->' separator.")
    if pattern.count('->') > 1:
        raise EinopsError("Pattern must contain exactly one '->' separator.")
    if pattern.count('(') != pattern.count(')'):
        raise EinopsError(f"Pattern has unbalanced parentheses: '{pattern}'")

    # Validate names and values provided in axes_lengths
    for name, length in axes_lengths.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise EinopsError(f"Axis name '{name}' in axes_lengths is not a valid identifier.")
        # Disallow leading/trailing underscores for user-provided names
        if name.startswith('_') or name.endswith('_'):
            raise EinopsError(f"Axis name '{name}' in axes_lengths should not start or end with underscore.")
        if not isinstance(length, int) or length <= 0:
            raise EinopsError(f"Length for axis '{name}' must be a positive integer, got {length}.")

# --- 4b. Parsing Helpers ---
def _parse_expression(expression_str: str) -> ParsedExpressionData:
    """
    Parses one side (LHS/RHS) of the pattern string into structured data.
    """
    raw_axes: List[Union[str, List[str]]] = []
    identifiers: Set[str] = set()
    has_ellipsis = False
    has_composition = False
    has_anonymous_axes = False
    has_trivial_anonymous_axis = False
    current_composition: List[str] = None
    in_composition = False
    paren_level = 0

    processed_expression = expression_str.replace('(', ' ( ').replace(')', ' ) ')
    tokens = processed_expression.split()

    for token in tokens:
        if not token:
            continue
        if token == '...':
            if has_ellipsis:
                raise EinopsError(f"Pattern side '{expression_str}' has multiple ellipses (...)")
            if paren_level > 0:
                if current_composition is None:
                    raise EinopsError(
                        f"Internal parsing error: In composition state without active list near '{token}' in '{expression_str}'."
                    )
                current_composition.append('...')
            else:
                raw_axes.append('...')
            has_ellipsis = True
        elif token == '(':
            if paren_level > 0:
                raise EinopsError(
                    f"Nested parentheses are not allowed in pattern: '{expression_str}'"
                )
            paren_level += 1
            in_composition = True
            current_composition = []
            has_composition = True
        elif token == ')':
            if paren_level == 0:
                raise EinopsError(
                    f"Unbalanced parentheses (extra closing) in pattern: '{expression_str}'"
                )
            paren_level -= 1
            in_composition = False
            if current_composition is not None:
                raw_axes.append(list(current_composition))
            current_composition = None
        elif token.isidentifier():
            if token.startswith('_') or token.endswith('_'):
                raise EinopsError(
                    f"Axis name '{token}' should not start or end with underscore."
                )
            current_scope_identifiers = (
                set(current_composition) if in_composition else identifiers
            )
            if token in current_scope_identifiers:
                # Allow duplicate '1' inside composition, else fail
                if not (in_composition and token == '1'):
                    raise EinopsError(
                        f"Duplicate identifier '{token}' in expression '{expression_str}'"
                    )
            if in_composition:
                current_composition.append(token)
            else:
                raw_axes.append(token)
            identifiers.add(token)
        elif token.isdigit():
            num_val = int(token)
            if num_val <= 0:
                raise EinopsError(
                    f"Numeric axis must be positive, found '{token}' in '{expression_str}'"
                )
            axis_repr = str(num_val)
            if num_val == 1:
                has_trivial_anonymous_axis = True
            else:
                has_anonymous_axes = True

            if num_val != 1:
                current_scope_identifiers = (
                    set(current_composition)
                    if in_composition
                    else identifiers | {ax for ax in raw_axes if isinstance(ax, str)}
                )
                if axis_repr in current_scope_identifiers:
                    raise EinopsError(
                        f"Duplicate identifier '{axis_repr}' (numeric literal treated as identifier) in expression '{expression_str}'"
                    )

            if in_composition:
                if current_composition is None:
                    raise EinopsError(
                        f"Internal parsing error: In composition state without active list near '{token}' in '{expression_str}'."
                    )
                current_composition.append(axis_repr)
            else:
                raw_axes.append(axis_repr)
        else:
            if '.' in token:
                raise EinopsError("Invalid token '.' found outside ellipsis (...)")
            raise EinopsError(f"Invalid token '{token}' found during parsing of '{expression_str}'")

    if paren_level != 0:
        raise EinopsError(f"Unbalanced parentheses in pattern: '{expression_str}'")

    # Simplify compositions of length 1
    processed_raw_axes = []
    for axis_group in raw_axes:
        if isinstance(axis_group, list):
            if len(axis_group) == 1:
                processed_raw_axes.append(axis_group[0])
            else:
                processed_raw_axes.append(axis_group)
        else:
            processed_raw_axes.append(axis_group)

    has_actual_composition = any(
        isinstance(ax, list) and len(ax) > 1 for ax in processed_raw_axes
    )

    return ParsedExpressionData(
        raw_axes=processed_raw_axes,
        identifiers=identifiers,
        has_ellipsis=has_ellipsis,
        has_composition=has_actual_composition,
        has_anonymous_axes=has_anonymous_axes,
        has_trivial_anonymous_axis=has_trivial_anonymous_axis,
    )

# --- 4c. Main Parsing Logic ---
def _parse_pattern(pattern: str, tensor_shape: Tuple[int, ...], axes_lengths: Dict[str, int]) -> ParsedPattern:
    lhs_str, rhs_str = pattern.split('->')
    lhs_str = lhs_str.strip()
    rhs_str = rhs_str.strip()

    lhs_data = _parse_expression(lhs_str)
    rhs_data = _parse_expression(rhs_str)

    if not lhs_data.has_ellipsis and rhs_data.has_ellipsis:
        raise EinopsError(f"Ellipsis found in right side, but not left side of pattern '{pattern}'")

    resolved_axes_lengths: Dict[str, int] = axes_lengths.copy()
    decomposed_lhs_axes: List[str] = []
    current_dim_index = 0
    ellipsis_axes_names: List[str] = []
    ellipsis_start_index_in_shape = -1

    if lhs_data.has_ellipsis:
        non_ellipsis_dims_lhs = 0
        for ax_group in lhs_data.raw_axes:
            if ax_group != '...':
                non_ellipsis_dims_lhs += 1

        if len(tensor_shape) < non_ellipsis_dims_lhs:
            raise EinopsError(
                f"Input tensor has {len(tensor_shape)} dimensions, but pattern requires at least {non_ellipsis_dims_lhs} explicit axes (excluding ellipsis). Pattern: '{lhs_str}'"
            )
        ellipsis_ndim = len(tensor_shape) - non_ellipsis_dims_lhs
        ellipsis_axes_names = [f"_ellipsis_{i}" for i in range(ellipsis_ndim)]

        try:
            ellipsis_marker_index_in_raw = lhs_data.raw_axes.index('...')
            ellipsis_start_index_in_shape = ellipsis_marker_index_in_raw
        except ValueError:
            raise EinopsError(
                f"Internal error: Ellipsis flag set but '...' not found in raw LHS axes: {lhs_data.raw_axes}"
            )

        for i, name in enumerate(ellipsis_axes_names):
            resolved_axes_lengths[name] = tensor_shape[ellipsis_start_index_in_shape + i]

    temp_dim_index_for_ellipsis = 0
    raw_axes_index = 0
    while raw_axes_index < len(lhs_data.raw_axes):
        axis_group = lhs_data.raw_axes[raw_axes_index]
        is_ellipsis_group = (axis_group == '...')

        if is_ellipsis_group:
            decomposed_lhs_axes.extend(ellipsis_axes_names)
            temp_dim_index_for_ellipsis += len(ellipsis_axes_names)
        else:
            if lhs_data.has_ellipsis and raw_axes_index > ellipsis_marker_index_in_raw:
                num_groups_after_ellipsis = raw_axes_index - (ellipsis_marker_index_in_raw + 1)
                current_shape_dim_index = ellipsis_start_index_in_shape + ellipsis_ndim + num_groups_after_ellipsis
            else:
                current_shape_dim_index = current_dim_index

            if current_shape_dim_index >= len(tensor_shape):
                raise EinopsError(
                    f"Rank mismatch: Pattern '{lhs_str}' has more axes than input tensor rank {len(tensor_shape)}"
                )
            dim_size = tensor_shape[current_shape_dim_index]

            if isinstance(axis_group, list):
                current_composition_axes = []
                known_axes = {}
                unknown_axes = []
                has_one_in_comp = False

                for ax in axis_group:
                    if ax == '1':
                        has_one_in_comp = True
                        continue
                    elif ax.isdigit():
                        raise EinopsError(
                            f"Numeric literal '{ax}' > 1 is not allowed in LHS composition '{axis_group}'"
                        )
                    elif ax in resolved_axes_lengths:
                        known_axes[ax] = resolved_axes_lengths[ax]
                    else:
                        unknown_axes.append(ax)
                    current_composition_axes.append(ax)

                if len(unknown_axes) > 1:
                    raise EinopsError(
                        f"Could not infer sizes for multiple axes {unknown_axes} in composition '{axis_group}' mapping to dimension {current_shape_dim_index} (size {dim_size}). Provide lengths via axes_lengths."
                    )
                known_product = math.prod(known_axes.values()) if known_axes else 1

                if has_one_in_comp and len(unknown_axes) == 0 and dim_size != known_product:
                    raise EinopsError(
                        f"Shape mismatch: Dimension {current_shape_dim_index} (size {dim_size}) does not match product of known axes ({known_product}) in composition '{axis_group}' which includes '1'."
                    )

                if len(unknown_axes) == 1:
                    unknown_axis = unknown_axes[0]
                    effective_dim_size_for_calc = dim_size
                    if known_product == 0:
                        raise EinopsError(
                            f"Cannot infer axis size for '{unknown_axis}' when known product is zero in composition '{axis_group}'"
                        )
                    if effective_dim_size_for_calc % known_product != 0:
                        raise EinopsError(
                            f"Shape mismatch: Dimension {current_shape_dim_index} (size {effective_dim_size_for_calc}) is not divisible by product of known axes ({known_product}) for composition '{axis_group}'. Cannot infer '{unknown_axis}'."
                        )
                    inferred = effective_dim_size_for_calc // known_product
                    if unknown_axis in resolved_axes_lengths and resolved_axes_lengths[unknown_axis] != inferred:
                        raise EinopsError(
                            f"Axis '{unknown_axis}' length mismatch: Provided/inferred earlier as {resolved_axes_lengths[unknown_axis]}, but calculated as {inferred} from shape dimension {current_shape_dim_index} (size {dim_size}) and composition '{axis_group}'."
                        )
                    resolved_axes_lengths[unknown_axis] = inferred
                    known_axes[unknown_axis] = inferred

                elif len(unknown_axes) == 0:
                    required_product = known_product
                    if dim_size != required_product:
                        if not (has_one_in_comp and dim_size == 1 and required_product == 1):
                            raise EinopsError(
                                f"Shape mismatch: Dimension {current_shape_dim_index} (size {dim_size}) does not match required product ({required_product}) for composition {axis_group}."
                            )
                decomposed_lhs_axes.extend(current_composition_axes)
            elif isinstance(axis_group, str):
                if axis_group == '1':
                    if dim_size != 1:
                        raise EinopsError(
                            f"Shape mismatch: Dimension {current_shape_dim_index} (size {dim_size}) corresponds to pattern axis '1', but size is not 1."
                        )
                elif axis_group.isdigit():
                    raise EinopsError(
                        f"Numeric literal '{axis_group}' > 1 is not allowed on LHS of rearrange pattern."
                    )
                    
                else:
                    axis_name = axis_group
                    if axis_name in resolved_axes_lengths and resolved_axes_lengths[axis_name] != dim_size:
                        raise EinopsError(
                            f"Axis '{axis_name}' length mismatch: Provided/inferred earlier as {resolved_axes_lengths[axis_name]}, but dimension {current_shape_dim_index} has size {dim_size}."
                        )
                    resolved_axes_lengths[axis_name] = dim_size
                    decomposed_lhs_axes.append(axis_name)
            else:
                raise EinopsError(
                    f"Internal parsing error: Unexpected element type '{type(axis_group)}' ('{axis_group}') in parsed LHS."
                )
            current_dim_index += 1
        raw_axes_index += 1

    num_pattern_groups_lhs = len(lhs_data.raw_axes)
    expected_rank_from_pattern = (
        (num_pattern_groups_lhs - 1 + len(ellipsis_axes_names))
        if lhs_data.has_ellipsis
        else num_pattern_groups_lhs
    )
    if expected_rank_from_pattern != len(tensor_shape):
        raise EinopsError(
            f"Rank mismatch: Pattern '{lhs_str}' implies rank {expected_rank_from_pattern}, but tensor has rank {len(tensor_shape)}."
        )

    decomposed_rhs_axes_final: List[str] = []
    repeat_axes_info: Dict[str, int] = {}
    repeat_counter = 0
    final_shape_list: List[int] = []

    for axis_group in rhs_data.raw_axes:
        if axis_group == '...':
            decomposed_rhs_axes_final.extend(ellipsis_axes_names)
            final_shape_list.extend(resolved_axes_lengths[ax] for ax in ellipsis_axes_names)
        elif isinstance(axis_group, list):
            group_axes = []
            group_len_prod = 1
            for axis_name in axis_group:
                if axis_name == '1':
                    anon_name = f"_anon_1_{len(decomposed_rhs_axes_final) + len(group_axes)}"
                    group_axes.append(anon_name)
                    group_len_prod *= 1
                elif axis_name.isdigit():
                    repeat_len = int(axis_name)
                    repeat_axis_name = f"_repeat_{repeat_counter}"
                    repeat_axes_info[repeat_axis_name] = repeat_len
                    group_axes.append(repeat_axis_name)
                    group_len_prod *= repeat_len
                    repeat_counter += 1
                elif axis_name in resolved_axes_lengths:
                    group_axes.append(axis_name)
                    group_len_prod *= resolved_axes_lengths[axis_name]
                elif axis_name in axes_lengths:
                    # This is a NEW named axis that the user wants to repeat
                    resolved_axes_lengths[axis_name] = axes_lengths[axis_name]
                    repeat_axes_info[axis_name] = axes_lengths[axis_name]
                    decomposed_rhs_axes_final.append(axis_name)
                    final_shape_list.append(axes_lengths[axis_name])
                else:
                    pass
            decomposed_rhs_axes_final.extend(group_axes)
            if len(group_axes) > 0:
                final_shape_list.append(group_len_prod)
        elif isinstance(axis_group, str):
            if axis_group == '1':
                anon_name = f"_anon_1_{len(decomposed_rhs_axes_final)}"
                decomposed_rhs_axes_final.append(anon_name)
                final_shape_list.append(1)
            elif axis_group.isdigit():
                repeat_len = int(axis_group)
                repeat_axis_name = f"_repeat_{repeat_counter}"
                repeat_axes_info[repeat_axis_name] = repeat_len
                decomposed_rhs_axes_final.append(repeat_axis_name)
                final_shape_list.append(repeat_len)
                repeat_counter += 1
            else:
                if axis_group in resolved_axes_lengths:
                    decomposed_rhs_axes_final.append(axis_group)
                    final_shape_list.append(resolved_axes_lengths[axis_group])
                elif axis_group in axes_lengths:
                    resolved_axes_lengths[axis_group] = axes_lengths[axis_group]
                    repeat_axes_info[axis_group] = axes_lengths[axis_group]
                    decomposed_rhs_axes_final.append(axis_group)
                    final_shape_list.append(axes_lengths[axis_group])
                else:
                    pass
        else:
            raise EinopsError(
                f"Internal parsing error: Unexpected element type '{type(axis_group)}' ('{axis_group}') in parsed RHS."
            )

    final_shape = tuple(final_shape_list)

    lhs_set = set(decomposed_lhs_axes)
    rhs_set_all_atomic = {ax for ax in decomposed_rhs_axes_final if not ax.startswith('_anon_1_')}
    repeat_set = set(repeat_axes_info.keys())

    extra_on_rhs = rhs_set_all_atomic - lhs_set
    if extra_on_rhs != repeat_set:
        unaccounted_extra = extra_on_rhs - repeat_set
        raise EinopsError(f"Axes {unaccounted_extra} appear on RHS but not LHS")

    missing_on_rhs = lhs_set - rhs_set_all_atomic
    if missing_on_rhs:
        raise EinopsError(f"Axes {missing_on_rhs} present on LHS but missing on RHS. Reduction is not supported by rearrange.")

    needs_reshaping_input = lhs_data.has_composition or lhs_data.has_trivial_anonymous_axis
    needs_repeating = bool(repeat_axes_info)
    needs_reshaping_output = rhs_data.has_composition or any(
        ax.startswith('_anon_1_') for ax in decomposed_rhs_axes_final
    )
    shape_after_lhs_reshape = tuple(resolved_axes_lengths[ax] for ax in decomposed_lhs_axes)

    transpose_indices = []
    needs_transposing = False
    source_axes_order = decomposed_lhs_axes
    target_order_for_transpose = [ax for ax in decomposed_rhs_axes_final if ax in lhs_set]

    if source_axes_order != target_order_for_transpose:
        if set(source_axes_order) != set(target_order_for_transpose):
            raise EinopsError(
                f"Internal error: Mismatch between source axes ({set(source_axes_order)}) "
                f"and target axes for transpose ({set(target_order_for_transpose)})."
            )
        needs_transposing = True
        try:
            transpose_indices = tuple(source_axes_order.index(ax) for ax in target_order_for_transpose)
        except ValueError as e:
            raise EinopsError(
                f"Internal error calculating transpose order: {e}"
            )

    parsed_info = ParsedPattern(
        lhs_expression=lhs_data,
        decomposed_lhs_axes=decomposed_lhs_axes,
        rhs_expression=rhs_data,
        decomposed_rhs_axes_final=decomposed_rhs_axes_final,
        resolved_axes_lengths=resolved_axes_lengths,
        repeat_axes_info=repeat_axes_info,
        needs_reshaping_input=needs_reshaping_input,
        needs_repeating=needs_repeating,
        needs_transposing=needs_transposing,
        needs_reshaping_output=needs_reshaping_output,
        shape_after_lhs_reshape=shape_after_lhs_reshape,
        transpose_indices=tuple(transpose_indices),
        final_shape=final_shape,
    )
    return parsed_info

# --- 4d. Execution (Executor Logic) ---
def _execute_rearrangement(tensor: np.ndarray, plan: ParsedPattern) -> np.ndarray:
    current_tensor = tensor
    current_shape = tensor.shape
    try:
        if plan.needs_reshaping_input:
            target_shape = plan.shape_after_lhs_reshape
            if math.prod(current_shape) != math.prod(target_shape):
                raise EinopsError(
                    f"Internal error during initial reshape: Cannot reshape {current_shape} "
                    f"(size {math.prod(current_shape)}) to {target_shape} "
                    f"(size {math.prod(target_shape)}), element count mismatch."
                )
            current_tensor = current_tensor.reshape(target_shape)
            current_shape = current_tensor.shape

        if plan.needs_transposing:
            if len(plan.transpose_indices) != len(current_shape):
                raise EinopsError(
                    f"Internal error during transpose: Transpose indices length "
                    f"{len(plan.transpose_indices)} != tensor rank {len(current_shape)}. "
                    f"Current shape: {current_shape}"
                )
            current_tensor = np.transpose(current_tensor, axes=plan.transpose_indices)
            current_shape = current_tensor.shape

        needs_axis_insertion = plan.needs_repeating or any(
            ax.startswith("_anon_1_") for ax in plan.decomposed_rhs_axes_final
        )
        if needs_axis_insertion:
            temp_tensor = current_tensor
            insert_locations = []

            lhs_axes_set = set(plan.decomposed_lhs_axes)
            current_axis_index_in_transposed = 0
            for i, final_axis_name in enumerate(plan.decomposed_rhs_axes_final):
                is_new_axis = (
                    final_axis_name.startswith(('_repeat_', '_anon_1_'))
                    or (final_axis_name in plan.repeat_axes_info and final_axis_name not in lhs_axes_set)
                )
                if is_new_axis:
                    insert_locations.append((i, final_axis_name))
                else:
                    current_axis_index_in_transposed += 1

            if insert_locations:
                insert_locations.sort(key=lambda x: x[0])
                num_inserted = 0
                for insert_pos, axis_name in insert_locations:
                    actual_insert_pos = insert_pos
                    temp_tensor = np.expand_dims(temp_tensor, axis=actual_insert_pos)
                    num_inserted += 1

            current_shape_after_expand = temp_tensor.shape
            repeat_tensor = temp_tensor
            for insert_pos, axis_name in insert_locations:
                if axis_name in plan.repeat_axes_info:
                    repeat_len = plan.repeat_axes_info[axis_name]
                    if repeat_len > 1:
                        repeat_tensor = np.repeat(repeat_tensor, repeat_len, axis=insert_pos)

            current_tensor = repeat_tensor
            current_shape = current_tensor.shape

        if current_shape != plan.final_shape:
            if plan.needs_reshaping_output or current_shape != plan.final_shape:
                target_shape = plan.final_shape
                if math.prod(current_shape) != math.prod(target_shape):
                    raise EinopsError(
                        f"Internal error before final reshape: Cannot reshape {current_shape} "
                        f"(size {math.prod(current_shape)}) to {target_shape} "
                        f"(size {math.prod(target_shape)}), element count mismatch."
                    )
                current_tensor = current_tensor.reshape(target_shape)
                current_shape = current_tensor.shape

        if current_tensor.shape != plan.final_shape:
            raise EinopsError(
                f"Internal error: Final shape mismatch after all operations. "
                f"Expected {plan.final_shape}, but got {current_shape}"
            )

    except ValueError as ve:
        raise EinopsError(f"NumPy ValueError during execution: {ve}. Current shape during error: {current_shape}") from ve
    except IndexError as ie:
        op = "transpose" if plan.needs_transposing else "repeat/expand"
        indices = plan.transpose_indices if plan.needs_transposing else 'N/A'
        raise EinopsError(
            f"NumPy IndexError during {op}: {ie}. Current shape: {current_shape}. Indices/Axis info: {indices}"
        ) from ie
    except Exception as e:
        raise EinopsError(
            f"Unexpected NumPy error during execution: {type(e).__name__}: {e}. Current shape: {current_shape}"
        ) from e

    return current_tensor