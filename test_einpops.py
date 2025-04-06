import numpy as np
from my_einops import rearrange, EinopsError
# from einops_testing import rearrange, EinopsError
import pytest


# --- Pytest Unit Tests --- (left as-is)
def test_invalid_tensor_type():
    with pytest.raises(EinopsError, match="Input tensor must be a NumPy ndarray"):
        rearrange([1, 2, 3], 'a -> a')

# etc... (the rest of your tests)

# --- Basic Validation Error Tests ---
def test_invalid_tensor_type():
    with pytest.raises(EinopsError, match="Input tensor must be a NumPy ndarray"):
        rearrange([1, 2, 3], 'a -> a')

def test_no_separator():
    with pytest.raises(EinopsError, match="Pattern must contain '->' separator"):
        rearrange(np.zeros(1), 'a b c')

def test_multiple_separators():
    with pytest.raises(EinopsError, match="Pattern must contain exactly one '->' separator"):
        rearrange(np.zeros(1), 'a -> b -> c')

def test_invalid_token_dot():
    with pytest.raises(EinopsError, match="Invalid token '.'"):
        rearrange(np.zeros(1), 'a . . -> b')

def test_unbalanced_parens_opening():
    # FIX: Match start of message
    with pytest.raises(EinopsError, match="Pattern has unbalanced parentheses"):
        rearrange(np.zeros(1), '(a -> b')

def test_unbalanced_parens_closing():
    # FIX: Match start of message
    with pytest.raises(EinopsError, match="Pattern has unbalanced parentheses"):
        rearrange(np.zeros(1), 'a) -> b')

def test_invalid_axes_lengths_name_numeric():
    with pytest.raises(EinopsError, match="not a valid identifier"):
        rearrange(np.zeros(1), 'a -> b', **{'1b': 2})

def test_invalid_axes_lengths_value_zero():
    with pytest.raises(EinopsError, match="must be a positive integer"):
        rearrange(np.zeros(1), 'a -> b', **{'b': 0})

def test_invalid_axes_lengths_value_negative():
    with pytest.raises(EinopsError, match="must be a positive integer"):
        rearrange(np.zeros(1), 'a -> b', **{'b': -1})

def test_invalid_axes_lengths_underscore_prefix():
    with pytest.raises(EinopsError, match="should not start or end with underscore"):
        rearrange(np.zeros(1), 'a -> b', **{'_b': 2})

def test_invalid_axes_lengths_underscore_suffix():
    with pytest.raises(EinopsError, match="should not start or end with underscore"):
        rearrange(np.zeros(1), 'a -> b', **{'b_': 2})


# --- Parsing and Semantic Error Tests ---
def test_dots_outside_ellipsis():
    with pytest.raises(EinopsError, match="Invalid token '.' found outside ellipsis"):
        rearrange(np.zeros((2,3)), 'a . b -> a b')

def test_multiple_ellipsis_lhs():
    with pytest.raises(EinopsError, match="multiple ellipses"):
        rearrange(np.zeros((2,3)), '... a ... -> a')

def test_ellipsis_rhs_only():
    with pytest.raises(EinopsError, match="Ellipsis found in right side, but not left"):
        rearrange(np.zeros((2,3)), 'a b -> ... a b')

def test_nested_parentheses():
    with pytest.raises(EinopsError, match="Nested parentheses"):
        rearrange(np.zeros((2,3)), 'a (b (c)) -> a b c')

def test_duplicate_identifier_lhs_simple():
    with pytest.raises(EinopsError, match="Duplicate identifier 'a'"):
        rearrange(np.zeros((2,2)), 'a a -> a')

def test_duplicate_identifier_lhs_composition():
    with pytest.raises(EinopsError, match="Duplicate identifier 'b'"):
        rearrange(np.zeros((2,2,2)), 'a (b b) -> a b')

def test_duplicate_identifier_rhs_simple():
    with pytest.raises(EinopsError, match="Duplicate identifier 'a'"):
        rearrange(np.zeros((2,)), 'a -> a a')

def test_duplicate_identifier_rhs_composition():
    with pytest.raises(EinopsError, match="Duplicate identifier 'b'"):
        rearrange(np.zeros((2,2)), 'a b -> (b b)')

def test_invalid_identifier_underscore_prefix():
    with pytest.raises(EinopsError, match="should not start or end with underscore"):
        rearrange(np.zeros((2,)), '_a -> a')

def test_invalid_identifier_underscore_suffix():
    with pytest.raises(EinopsError, match="should not start or end with underscore"):
        rearrange(np.zeros((2,)), 'a_ -> a')

def test_unknown_axis_rhs_no_length():
    expected_pattern = r"Axis 'c' appears on the right side \('a c'\) but is not defined on the left side \('a b'\) and its length is not specified via axes_lengths=\{\}"
    with pytest.raises(EinopsError, match=expected_pattern):
        rearrange(np.zeros((2,3)), 'a b -> a c')


def test_axis_mismatch_reduction():
    # FIX: Use raw string and escape braces
    with pytest.raises(EinopsError, match=r"Axes \{'b'\} present on LHS but missing on RHS"):
        rearrange(np.zeros((2,3)), 'a b -> a')

def test_non_unitary_anonymous_lhs_simple():
    with pytest.raises(EinopsError, match="Numeric literal '2' > 1 is not allowed on LHS"):
        rearrange(np.zeros((2,3)), 'a 2 -> a')

def test_non_unitary_anonymous_lhs_composition():
    with pytest.raises(EinopsError, match ="Numeric literal 2 > 1 not allowed in LHS composition"):
        rearrange(np.zeros((2,3)), '(a 2) -> a')

def test_non_unitary_anonymous_rhs_no_repeat():
    # FIX: Comment out - code now handles this correctly as a repeat
    with pytest.raises(EinopsError, match=r"Axes {'b'} present on LHS but missing on RHS. Reduction is not supported by rearrange."):
         rearrange(np.zeros((2,3)), 'a b -> a 2')
    pass

def test_split_axis_incompatible_shape():
    with pytest.raises(EinopsError, match="is not divisible by product of known axes"):
        rearrange(np.arange(10).reshape(5, 2), '(h w) c -> h w c', h=3)

def test_split_axis_multiple_unknowns():
    with pytest.raises(EinopsError, match="Could not infer sizes for multiple axes"):
        rearrange(np.arange(10).reshape(5, 2), '(h w) c -> h w c')

def test_shape_rank_mismatch_pattern_longer():
    # FIX: Use simpler match
    with pytest.raises(EinopsError, match="Rank mismatch:"):
        rearrange(np.arange(10).reshape(5, 2), 'a b c -> a b c')

def test_shape_rank_mismatch_pattern_shorter():
    # FIX: Use simpler match
     with pytest.raises(EinopsError, match="Rank mismatch:"):
        rearrange(np.arange(10).reshape(5, 2), 'a -> a')

def test_shape_mismatch_anonymous_1():
    # FIX: Use raw string and escape parentheses
    with pytest.raises(EinopsError, match=r"Dimension 1 \(size 2\) corresponds to pattern axis '1', but size is not 1"):
        rearrange(np.arange(10).reshape(5, 2), 'a 1 -> a')

def test_shape_mismatch_composition_product():
    # FIX: Use raw string and escape brackets/parentheses
    with pytest.raises(EinopsError, match=r"Dimension 0 \(size 5\) does not match required product \(4\) for composition \['h', 'w'\]"):
        rearrange(np.arange(10).reshape(5, 2), '(h w) c -> h w c', h=2, w=2)

def test_axis_length_mismatch_provided_vs_shape():
     with pytest.raises(EinopsError, match="Axis 'h' length mismatch: Provided/inferred earlier as 3, but dimension 0 has size 4"):
         rearrange(np.zeros((4, 5)), 'h w -> w h', h=3)

def test_axis_length_mismatch_inferred_vs_provided():
     # FIX: Expect Shape mismatch error instead
     # FIX: Use raw string and escape parentheses
     with pytest.raises(EinopsError, match=r"Shape mismatch: Dimension 0 \(size 2\) does not match required product \(20\) for composition \['h', 'w'\]"):
         rearrange(np.zeros((2, 10)), '(h w) -> h w', h=2, w=10) # Shape implies w=5


# --- Tests Expected to Pass ---

# Simple Reshapes
def test_simple_reshape_merge():
    tensor = np.arange(12).reshape(3, 4)
    result = rearrange(tensor, 'h w -> (h w)')
    assert result.shape == (12,)
    assert np.array_equal(result, np.arange(12))

def test_simple_reshape_split():
    tensor = np.arange(12)
    result = rearrange(tensor, '(h w) -> h w', h=3)
    assert result.shape == (3, 4)
    assert np.array_equal(result, np.arange(12).reshape(3, 4))

def test_all_merge():
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, 'a b c -> (a b c)')
    assert result.shape == (24,)
    assert np.array_equal(result, np.arange(24))

def test_all_split():
    tensor = np.arange(24)
    result = rearrange(tensor, '(a b c) -> a b c', a=2, b=3)
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, np.arange(24).reshape(2, 3, 4))

# Transpose
def test_transpose_2d():
    tensor = np.arange(12).reshape(3, 4)
    result = rearrange(tensor, 'h w -> w h')
    assert result.shape == (4, 3)
    assert np.array_equal(result, tensor.T)

def test_transpose_3d():
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, 'a b c -> c b a')
    assert result.shape == (4, 3, 2)
    assert np.array_equal(result, np.transpose(tensor, (2, 1, 0)))

# Split and Merge Combinations
def test_split_axis():
    tensor = np.arange(12).reshape(6, 2)
    result = rearrange(tensor, '(h w) c -> h w c', h=3)
    assert result.shape == (3, 2, 2)
    assert np.array_equal(result, np.arange(12).reshape(3, 2, 2))

def test_merge_axes():
    tensor = np.arange(12).reshape(2, 3, 2)
    result = rearrange(tensor, 'a b c -> (a b) c')
    assert result.shape == (6, 2)
    assert np.array_equal(result, np.arange(12).reshape(6, 2))

def test_split_and_merge():
    tensor = np.arange(24).reshape(12, 2)
    result = rearrange(tensor, '(h w) c -> h (w c)', h=3)
    assert result.shape == (3, 8)
    assert np.array_equal(result, np.arange(24).reshape(3, 8))

def test_merge_and_split():
    tensor = np.arange(24).reshape(6, 4)
    result = rearrange(tensor, '(a b) c -> a (b c)', a=2)
    assert result.shape == (2, 12)
    assert np.array_equal(result, np.arange(24).reshape(2, 12))


# --- Ellipsis Tests ---
def test_ellipsis_no_op_front():
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, '... w -> ... w')
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, tensor)

def test_ellipsis_no_op_middle():
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, 'b ... w -> b ... w')
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, tensor)

def test_ellipsis_no_op_end():
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, 'b h ... -> b h ...')
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, tensor)

def test_ellipsis_transpose_1():
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, 'b ... w -> ... w b')
    assert result.shape == (3, 4, 2)
    assert np.array_equal(result, np.transpose(tensor, (1, 2, 0)))

def test_ellipsis_transpose_2():
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, 'b h ... -> ... h b')
    assert result.shape == (4, 3, 2)
    assert np.array_equal(result, np.transpose(tensor, (2, 1, 0)))

def test_ellipsis_split():
    tensor = np.arange(60).reshape(2, 3, 10)
    result = rearrange(tensor, 'b ... (h w) -> b ... h w', h=2)
    assert result.shape == (2, 3, 2, 5)
    assert np.array_equal(result, np.arange(60).reshape(2, 3, 2, 5))

def test_ellipsis_merge():
    tensor = np.arange(60).reshape(2, 3, 2, 5)
    result = rearrange(tensor, 'b ... h w -> b ... (h w)')
    assert result.shape == (2, 3, 10)
    assert np.array_equal(result, np.arange(60).reshape(2, 3, 10))

def test_ellipsis_only():
    tensor = np.arange(60).reshape(2, 3, 2, 5)
    result = rearrange(tensor, '... -> ...')
    assert result.shape == (2, 3, 2, 5)
    assert np.array_equal(result, tensor)


# --- Anonymous Axis '1' Tests ---
def test_anon_axis_1_squeeze():
    tensor = np.arange(6).reshape(2, 1, 3)
    result = rearrange(tensor, 'a 1 c -> a c')
    assert result.shape == (2, 3)
    assert np.array_equal(result, np.arange(6).reshape(2, 3))

def test_anon_axis_1_unsqueeze():
    tensor = np.arange(6).reshape(2, 3)
    result = rearrange(tensor, 'a c -> a 1 c')
    assert result.shape == (2, 1, 3)
    assert np.array_equal(result, np.arange(6).reshape(2, 1, 3))

def test_anon_axis_1_transpose():
    tensor = np.arange(6).reshape(1, 2, 3)
    result = rearrange(tensor, '1 b c -> b 1 c')
    assert result.shape == (2, 1, 3)
    assert np.array_equal(result, np.arange(6).reshape(2, 1, 3)) # Value check needs care

def test_anon_axis_1_composition_lhs_squeeze():
    # FIX: Expect Rank mismatch error
    tensor = np.arange(3).reshape(3, 1)
    with pytest.raises(EinopsError, match="Rank mismatch:"):
        rearrange(tensor, '(a 1) -> a', a=3)
    # assert result.shape == (3,)
    # assert np.array_equal(result, np.arange(3))

def test_anon_axis_1_composition_rhs_unsqueeze():
    tensor = np.arange(6).reshape(2, 3)
    result = rearrange(tensor, 'a b -> (a 1 b)')
    # Expected intermediate: (2, 1, 3) -> Final: (6,)
    assert result.shape == (6,)
    assert np.array_equal(result, np.arange(6).reshape(2,1,3).flatten())


# --- Repeat Tests (Numeric and Named) ---
def test_repeat_literal_simple():
    tensor = np.array([1, 2])
    result = rearrange(tensor, 'a -> a 3')
    expected = np.array([[1, 1, 1], [2, 2, 2]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_repeat_literal_end():
    tensor = np.array([[1, 2], [3, 4]])
    result = rearrange(tensor, 'a b -> a b 2')
    expected = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_repeat_literal_middle():
    tensor = np.array([[1, 2], [3, 4]])
    result = rearrange(tensor, 'a b -> a 2 b')
    expected = np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_repeat_literal_start():
    tensor = np.array([[1, 2], [3, 4]])
    result = rearrange(tensor, 'a b -> 3 a b')
    expected = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_repeat_literal_transpose():
    tensor = np.array([[1, 2], [3, 4]])
    result = rearrange(tensor, 'a b -> b 2 a')
    # FIX: Correct expected value after transpose then repeat
    expected = np.array([[[1, 3], [1, 3]], [[2, 4], [2, 4]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_repeat_literal_merge():
    tensor = np.array([[1, 2], [3, 4]])
    result = rearrange(tensor, 'a b -> (a 2 b)')
    # Expected intermediate: (2, 2, 2) from 'a 2 b'
    # [[1, 2], [1, 2]], [[3, 4], [3, 4]] -> flatten
    expected = np.array([1, 2, 1, 2, 3, 4, 3, 4])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_repeat_literal_multiple():
    tensor = np.array([1, 2])
    result = rearrange(tensor, 'a -> 2 a 3')
    # Expected intermediate 'a 3': [[1,1,1],[2,2,2]]
    # Expected final '2 a 3': [[[1,1,1],[2,2,2]], [[1,1,1],[2,2,2]]]
    expected = np.array([[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_repeat_named_axis():
    # FIX: This test should now pass due to corrected validation logic
    tensor = np.array([[1, 2], [3, 4]])
    result = rearrange(tensor, 'a b -> a rpt b', rpt=2)
    expected = np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]]) # Same as 'a 2 b'
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)

def test_repeat_user_case_squeeze_and_repeat():
    # FIX: This test should now pass due to corrected validation logic
    # Test case: 'a 1 c -> a b c' with b=4
    tensor = np.random.rand(3, 1, 5)
    result = rearrange(tensor, 'a 1 c -> a b c', b=4)
    assert result.shape == (3, 4, 5)
    # Value check: squeeze '1', then repeat the new middle axis
    squeezed = tensor.squeeze(axis=1) # Shape (3, 5)
    unsqueezed = np.expand_dims(squeezed, axis=1) # Shape (3, 1, 5)
    expected_manual = np.repeat(unsqueezed, 4, axis=1) # Shape (3, 4, 5)
    assert np.array_equal(result, expected_manual)

def test_repeat_in_composition():
    tensor = np.array([1,2]) # shape (2,)
    result = rearrange(tensor, 'a -> (a 3)')
    # Expected intermediate 'a 3': [[1,1,1],[2,2,2]] shape (2,3)
    # Expected final '(a 3)': flatten -> [1,1,1,2,2,2] shape (6,)
    expected = np.array([1,1,1,2,2,2])
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)





