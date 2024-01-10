# -*- coding: utf-8 -*-
"""Collection of utility functions for numpy and sparse arrays."""
from __future__ import annotations

# Built-Ins
import logging
from typing import Collection, Iterable, Optional, Sequence, overload

# Third Party
import numba as nb
import numpy as np
import sparse

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
# ## Private functions ## #
@nb.njit(fastmath=True, nogil=True, cache=True)
def _get_unique_idxs_and_counts(
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """Get the index positions of the start of each group and their counts.

    NOTE: Only works on sorted groups!

    Taken from pydata/sparse code -> _calc_counts_invidx(). See here:
    https://github.com/pydata/sparse/blob/10612ea939c72f427a0b7d58cec72db04b5e98ae/sparse/_coo/core.py#L1521

    Parameters
    ----------
    groups:
        The groups that each element in an array of the same shape belongs to.
        This should be a sorted 1d array.

    Returns
    -------
    unique_idxs:
        The index of the first element where each group is found.
    """
    # Init
    inv_idx = [0]
    counts: list[int] = []

    # Return early if groups is empty
    if len(groups) == 0:
        return (
            np.array(inv_idx, dtype=groups.dtype),
            np.array(counts, dtype=groups.dtype),
        )

    # Iterator over list getting unique IDs and counts
    last_group = groups[0]
    for i in range(1, len(groups)):
        if groups[i] != last_group:
            counts.append(i - inv_idx[-1])
            inv_idx.append(i)
            last_group = groups[i]
    counts.append(len(groups) - inv_idx[-1])

    return np.array(inv_idx, dtype=groups.dtype), np.array(counts, dtype=groups.dtype)


@nb.njit
def _is_sorted(array: np.ndarray) -> bool:  # pragma: no cover
    """Check is a numpy array is sorted."""
    # TODO(BT): Write a public available function which checks types etc...
    for i in range(array.size - 1):
        if array[i + 1] < array[i]:
            return False
    return True


@nb.njit
def _1d_is_ones(array: np.ndarray) -> bool:  # pragma: no cover
    """Check is a numpy array is only 1s."""
    # Disabling pylint warning, see https://github.com/PyCQA/pylint/issues/2910
    for i in nb.prange(array.size):  # pylint: disable=not-an-iterable
        if array[i] != 1:
            return False
    return True


def _is_in_order_sequential(array: Sequence[int] | np.ndarray) -> bool:
    """Check if a numpy array is both sequential an in order."""
    if not isinstance(array, np.ndarray):
        array = np.array(list(array))
    if not _is_sorted(array):
        return False
    return _1d_is_ones(np.diff(array))


def _sparse_unsorted_axis_swap(
    sparse_array: sparse.COO,
    new_axis_order: Iterable[int],
) -> sparse.COO:
    """Fast function for a partial sparse array transpose.

    A faster version of calling `sparse_array.transpose(new_axis_order)`, as
    the return coordinates are not re-sorted.

    WARNING: This function does not sort the sparse matrix indices again and
    might break the internal sparse.COO functionality which assumes the
    coordinate values are sorted.
    USE AT YOUR OWN RISK.

    This function also does not contain any error checking as it is designed
    to be fast over all else. Be careful when using this function to ensure
    correct values are always given.

    Parameters
    ----------
    sparse_array:
        The array to transpose

    new_axis_order:
        A tuple or list which contains a permutation of [0,1,..,N-1] where N
        is the number of axes of a. The i’th axis of the returned array will
        correspond to the axis numbered `axes[i]` of the input.

    Returns
    -------
    transposed_array:
        `sparse_array` with its axes permuted, without the coordinates being
        sorted.

    See Also
    --------
    `_sort_2d_sparse_coords_and_data`
    `_2d_sparse_sorted_axis_swap`
    """
    array = sparse_array.copy()
    array.coords = array.coords[new_axis_order, :]
    array.shape = tuple(array.shape[ax] for ax in new_axis_order)
    return array


def _sort_2d_sparse_coords_and_data(sparse_array: sparse.COO) -> sparse.COO:
    """Quickly sort the coordinates and data of a 2D sparse array.

    Used to replace the built-in sort method, as this is optimised for 2D
    arrays and will only work for them.

    Parameters
    ----------
    sparse_array:
        The 2D sparse matrix to sort.

    Returns
    -------
    sorted_array:
        `sparse_array` with its coordinates and data sorted.

    See Also
    --------
    `_sparse_unsorted_axis_swap`
    `_2d_sparse_sorted_axis_swap`

    Notes
    -----
    If an N-dimensional version is needed you're probably better off using the
    internal sparse.COO sort method defined here:
    https://github.com/pydata/sparse/blob/10612ea939c72f427a0b7d58cec72db04b5e98ae/sparse/_coo/core.py#L1239
    It uses `np.ravel_multi_index()` to create a 1D index which it then sorts.
    """
    # Only works on 2D sparse arrays
    n_dims = len(sparse_array.coords)
    if n_dims != 2:
        raise ValueError(
            "_sort_2d_sparse_coords_and_data() only works on 2D sparse arrays. "
            f"got {n_dims}D array instead."
        )

    # Stack and sort the arrays
    stacked = np.vstack((sparse_array.coords, sparse_array.data))
    idx = stacked[0, :].argsort()
    stacked = np.take(stacked, idx, axis=1)

    # Stick sorted arrays back into the sparse matrix
    array = sparse_array.copy()
    array.coords = stacked[:2].astype(sparse_array.coords.dtype)
    array.data = stacked[2].astype(sparse_array.data.dtype)
    return array


def _2d_sparse_sorted_axis_swap(
    sparse_array: sparse.COO,
    new_axis_order: Iterable[int],
) -> sparse.COO:
    """Fast function for a 2D sparse array transpose.

    A faster version of calling `sparse_array.transpose(new_axis_order)`.

    WARNING: This function avoids some N-Dimensional code within sparse.COO,
    to run faster. It changes internal values of the sparse matrix without
    the class error checking.
    USE AT YOUR OWN RISK.

    This function also does not contain any error checking as it is designed
    to be fast over all else. Be careful when using this function to ensure
    correct values are always given.

    Parameters
    ----------
    sparse_array:
        The array to transpose

    new_axis_order:
        A tuple or list which contains a permutation of [0,1,..,N-1] where N
        is the number of axes of a. The i’th axis of the returned array will
        correspond to the axis numbered `axes[i]` of the input.

    Returns
    -------
    transposed_array:
        `sparse_array` with its axes permuted.

    See Also
    --------
    `_sparse_unsorted_axis_swap`
    `_sort_2d_sparse_coords_and_data`
    """
    array = _sparse_unsorted_axis_swap(
        sparse_array=sparse_array,
        new_axis_order=new_axis_order,
    )
    return _sort_2d_sparse_coords_and_data(array)


def _flatten_some_sparse_axis_with_transpose(
    array: sparse.COO,
    flatten_axis: Sequence[int],
):
    """Flatten some axis in a sparse array.

    Will always work, no matter which axis are chosen
    """
    # Figure out how to transpose, and reverse
    other_dims = list(set(range(array.ndim)) - set(flatten_axis))
    axis_swap = list(flatten_axis) + other_dims
    axis_swap_reverse = [axis_swap.index(x) for x in range(len(axis_swap))]

    # Transpose and get the flat coordinates out
    array = array.transpose(axis_swap)
    get_vals = list(range(len(flatten_axis)))
    flat_coord = np.ravel_multi_index(
        np.take(array.coords, get_vals, axis=0),
        np.take(array.shape, get_vals),
    )
    flat_coord.sort()
    return flat_coord, array, axis_swap_reverse


def _flatten_some_sparse_axis_without_transpose(
    array: sparse.COO,
    flatten_axis: Sequence[int],
):
    """Flatten some axis in a sparse array.

    Will only work when flat axis are sequential and in order
    """
    flat_coord = np.ravel_multi_index(
        np.take(array.coords, flatten_axis, axis=0),
        np.take(array.shape, flatten_axis),
    )
    flat_coord.sort()
    return flat_coord


def _broadcast_array_to_array(
    array: sparse.COO,
    target_array: sparse.COO,
    array_dims: int | Sequence[int],
) -> sparse.COO:
    """Broadcast a sparse array into a larger sparse array."""
    # Validate inputs
    if isinstance(array_dims, int):
        array_dims = [array_dims]
    assert isinstance(array_dims, Sequence)

    if len(array_dims) != array.ndim:
        raise ValueError(
            "array_dims must have the same number of values as array has "
            "dimensions.\n"
            f"Expected {array.ndim} values, got {len(array_dims)} values "
            f"instead: {array_dims}"
        )

    validate_axis(array_dims, target_array.ndim)

    # Flatten the given array as a baseline
    flat_array_coord = np.ravel_multi_index(array.coords, array.shape)

    # Only need to transpose (slow!) if array dims are not sequential
    if _is_in_order_sequential(array_dims) and array_dims[0] == 0:
        axis_swap_reverse = None
        flat_target_coord = _flatten_some_sparse_axis_without_transpose(
            array=target_array, flatten_axis=array_dims
        )
    else:
        return_vals = _flatten_some_sparse_axis_with_transpose(
            array=target_array, flatten_axis=array_dims
        )
        flat_target_coord, target_array, axis_swap_reverse = return_vals

    # Find the index of any values in array but not in target
    array_unique = np.unique(flat_array_coord)
    target_unique = np.unique(flat_target_coord)
    missing_values = list(set(array_unique) - set(target_unique))
    missing_idx = np.searchsorted(flat_array_coord, np.array(missing_values))

    # Make sure we've got a reference in target for all array values
    invalid_values = set(target_unique) - set(array_unique)
    if len(invalid_values) > 0:
        raise ValueError(
            f"Unable to broadcast `array` of shape {array.shape} to "
            f"`target_array` of shape {target_array.shape}."
        )

    # Broadcast the array data to the target coordinates
    _, counts = _get_unique_idxs_and_counts(flat_target_coord)
    broadcast_data = np.delete(array.data, missing_idx)
    broadcast_data = np.repeat(broadcast_data, counts)

    # Build the return array and transpose back to original order
    final_array = sparse.COO(
        data=broadcast_data,
        coords=target_array.coords,
        shape=target_array.shape,
        fill_value=target_array.fill_value,
        sorted=True,
    )

    if axis_swap_reverse is not None:
        return final_array.transpose(axis_swap_reverse)
    return final_array


def _broadcast_scalar_to_array(
    fill_value: int | float,
    target_array: sparse.COO,
) -> sparse.COO:
    """Broadcast a scalar value into a sparse array."""
    return sparse.COO(
        data=np.full_like(target_array.data, fill_value),
        coords=target_array.coords,
        shape=target_array.shape,
        fill_value=target_array.fill_value,
        sorted=True,
    )


# ## Public functions ## #
def validate_axis(
    axis: Collection[int],
    n_dims: int,
    name: str = "axis",
) -> None:
    """Validate axis values against a number of dimensions.

    Parameters
    ----------
    axis:
        The axis values to validate

    n_dims:
        The number of dimensions in the matrix that axis should be validated against

    name:
        The name to write out in the error message, if generated

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If any axis values are negative, duplicated, or too high.
    """
    if min(axis) < 0:
        raise ValueError(f"{name} values cannot be negative. {axis} are not valid axes.")

    if len(axis) > len(set(axis)):
        raise ValueError(f"{name} values are not unique. {axis} are not valid axes.")

    if max(axis) >= n_dims:
        raise ValueError(
            f"{name} values too high. {axis} are not valid axes for an array with {n_dims} dimensions."
        )


def remove_sparse_nan_values(
    array: np.ndarray | sparse.COO,
    fill_value: int | float = 0,
) -> np.ndarray | sparse.COO:
    """Remove any NaN values from a dense or sparse array."""
    if isinstance(array, np.ndarray):
        return np.nan_to_num(array, nan=fill_value)

    # Must be sparse and need special infill
    return sparse.COO(
        coords=array.coords,
        data=np.nan_to_num(array.data, nan=fill_value),
        fill_value=np.nan_to_num(array.fill_value, nan=fill_value),
        shape=array.shape,
    )


def broadcast_sparse_matrix(
    array: sparse.COO | int | float,
    target_array: sparse.COO,
    array_dims: Optional[int | Sequence[int]] = None,
) -> sparse.COO:
    """Expand an array to a target sparse matrix, matching target sparsity.

    Parameters
    ----------
    array:
        Input array.

    target_array:
        Target array to broadcast to. The return matrix will be as sparse
        and the same shape as this matrix.

    array_dims:
        The dimensions of `target_array` which correspond to array.

    Returns
    -------
    expanded_array:
        Array expanded to be the same shape and sparsity as target_array
    """
    if isinstance(array, sparse.COO):
        if array_dims is None:
            raise ValueError("array_dims must be given if array is not a scalar value.")

        return _broadcast_array_to_array(
            array=array,
            target_array=target_array,
            array_dims=array_dims,
        )

    # Assume scalar
    return _broadcast_scalar_to_array(
        fill_value=array,
        target_array=target_array,
    )


@overload
def sparse_sum(
    sparse_array: sparse.COO,
    axis: Iterable[int] | int,
) -> sparse.COO:
    ...  # pragma: no cover


@overload
def sparse_sum(
    sparse_array: sparse.COO,
    axis=None,
) -> float:
    ...  # pragma: no cover


def sparse_sum(
    sparse_array: sparse.COO,
    axis: Optional[Iterable[int] | int] = None,
) -> sparse.COO | float:
    """Faster sum for a sparse.COO matrix.

    Converts the sum to a 2D operation and then optimises functionality for
    2D matrices, avoiding sparse.COO N-dimensional code.

    Parameters
    ----------
    sparse_array:
        The sparse array to sum.

    axis:
        The axis to sum `sparse_array` across.

    Returns
    -------
    sum:
        The sum of `sparse_matrix` elements over the given axis
    """
    # Validate given axis
    if axis is None:
        axis = list(range(sparse_array.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)

    # Init
    keep_axis = tuple(sorted(set(range(len(sparse_array.shape))) - set(axis)))
    final_shape = np.take(np.array(sparse_array.shape), keep_axis)
    remove_shape = np.take(np.array(sparse_array.shape), axis)

    # ## # Swap array into 2D where axis 1 needs reducing ## #
    # Basically a transpose, but quicker if we do it ourselves
    array = _sparse_unsorted_axis_swap(
        sparse_array=sparse_array, new_axis_order=list(keep_axis) + axis
    )

    # Return a float if summing all axis
    if keep_axis == tuple():
        return sparse_array.data.sum()

    # Reshape into the 2d array
    array = array.reshape(
        (
            np.prod(final_shape, dtype=np.intp),
            np.prod(remove_shape, dtype=np.intp),
        )
    )

    # Sort the coords and data - sparse sum requires this to be in order
    array = _sort_2d_sparse_coords_and_data(array)

    # Optimised sum across 2 dimensions
    unique_idxs, _ = _get_unique_idxs_and_counts(array.coords[0])
    result = np.add.reduceat(array.data, unique_idxs)

    # Reshape back to original final shape
    final_array = sparse.COO(
        data=result,
        coords=array.coords[:1, unique_idxs],
        shape=(array.shape[0],),
        has_duplicates=False,
        sorted=True,
        prune=True,
        fill_value=0,
    )
    return final_array.reshape(final_shape)
