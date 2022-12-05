# -*- coding: utf-8 -*-
"""Collection of utility functions for numpy and sparse arrays """
from __future__ import annotations

# Built-Ins
import logging

from typing import Iterable
from typing import Sequence

# Third Party
import sparse
import numpy as np
import numba as nb

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #


# # # FUNCTIONS # # #
# ## Private functions ## #
@nb.njit(fastmath=True, nogil=True, cache=True)
def _get_unique_idxs_and_counts(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the index positions of the start of each group and their counts

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
def _is_sorted(array: np.ndarray) -> bool:
    """Check is a numpy array is sorted."""
    # TODO(BT): Write a public available function which checks types etc...
    for i in range(array.size-1):
        if array[i+1] < array[i]:
            return False
    return True


@nb.njit(parallel=True)
def _1d_is_ones(array: np.ndarray) -> bool:
    """Check is a numpy array is only 1s."""
    for i in nb.prange(array.size):
        if array[i] != 1:
            return False
    return True


def _is_in_order_sequential(array: np.ndarray) -> bool:
    """Check if a numpy array is both sequential an in order."""
    if not _is_sorted(array):
        return False
    return _1d_is_ones(np.diff(array))


def _2d_sparse_sum_axis_1(
    sparse_shape: tuple[int, ...],
    sparse_coords: np.ndarray,
    sparse_data: np.ndarray,
) -> np.ndarray:
    """Internal function of sparse_sum(). Used with numba for speed"""
    unique_idxs, _ = _get_unique_idxs_and_counts(sparse_coords[0])
    result = np.add.reduceat(sparse_data, unique_idxs)

    return sparse.COO(
        data=result,
        coords=sparse_coords[:1, unique_idxs],
        shape=(sparse_shape[0],),
        has_duplicates=False,
        sorted=True,
        prune=True,
        fill_value=0,
    )


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
    """Quickly sort the coordinates and data of a 2D sparse array

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


# ## Public functions ## #
def remove_sparse_nan_values(
    array: np.ndarray | sparse.COO,
    fill_value: int | float = 0,
) -> np.ndarray | sparse.COO:
    """Remove any NaN values from a dense or sparse array"""
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
    array: sparse.COO,
    target_array: sparse.COO,
    array_dims: int | Sequence[int],
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
    # Validate inputs
    if isinstance(array_dims, int):
        array_dims = [array_dims]
    assert isinstance(array_dims, Sequence)

    # Flatten the given array as a baseline
    flat_array_coord = np.ravel_multi_index(array.coords, array.shape)
    flat_array_coord.sort()

    # TODO(): Don't need to mess about with transpose if array dims is sequential!
    # _is_in_order_sequential(dims)
    # TODO(): Faster if convert to 2D??

    # Get just the dimensions from target that are in array and flatten them
    other_dims = list(set(range(target_array.ndim)) - set(array_dims))

    axis_swap = list(array_dims) + other_dims
    axis_swap_reverse = [axis_swap.index(x) for x in range(len(axis_swap))]

    target_array = target_array.transpose(axis_swap)
    get_vals = list(range(len(array_dims)))
    flat_target_coord = np.ravel_multi_index(
        target_array.coords[get_vals, :],
        np.take(target_array.shape, get_vals),
    )
    flat_target_coord.sort()

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
    return sparse.COO(
        data=broadcast_data,
        coords=target_array.coords,
        shape=target_array.shape,
        fill_value=target_array.fill_value,
        sorted=True,
    ).transpose(axis_swap_reverse)


def broadcast_sparse_matrix_old(
    array: sparse.COO,
    target_array: sparse.COO,
    expand_axis: int | Iterable[int],
) -> sparse.COO:
    """Expand an array to a target sparse matrix, matching target sparsity.

    Parameters
    ----------
    array:
        Input array.

    target_array:
        Target array to broadcast to. The return matrix will be as sparse
        and the same shape as this matrix.

    expand_axis:
        Position in the expanded axes where the new axis (or axes) is placed.

    Returns
    -------
    expanded_array:
        Array expanded to be the same shape and sparsity as target_array
    """
    # Validate inputs
    if isinstance(expand_axis, int):
        expand_axis = [expand_axis]
    assert isinstance(expand_axis, Iterable)

    # Init
    n_vals = len(array.data)

    # Expand the dimension to match target shape
    new_coords = list()
    build_shape = list()
    old_coords_idx = 0
    for i in range(len(target_array.shape)):
        if i in expand_axis:
            new_coords.append(np.zeros(n_vals))
            build_shape.append(1)
        else:
            new_coords.append(array.coords[old_coords_idx])
            build_shape.append(array.shape[old_coords_idx])
            old_coords_idx += 1

    expanded = sparse.COO(
        coords=np.array(new_coords, dtype=int),
        data=array.data,
        fill_value=array.fill_value,
        shape=tuple(build_shape),
    )

    sparse_locations = target_array != 0
    return expanded * sparse_locations


def sparse_sum(sparse_array: sparse.COO, axis: Iterable[int]) -> sparse.COO:
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
        The sum of `sparse_matrix` elements over the given axiss
    """
    # Init
    axis = list(axis)
    keep_axis = tuple(sorted(set(range(len(sparse_array.shape))) - set(axis)))
    final_shape = np.take(np.array(sparse_array.shape), keep_axis)
    remove_shape = np.take(np.array(sparse_array.shape), axis)

    # ## # Swap array into 2D where axis 1 needs reducing ## #
    # Basically a transpose, but quicker if we do it ourselves
    array = _sparse_unsorted_axis_swap(
        sparse_array=sparse_array, new_axis_order=list(keep_axis) + axis
    )

    # Reshape into the 2d array
    array = array.reshape(
        (
            np.prod(final_shape, dtype=np.intp),
            np.prod(remove_shape, dtype=np.intp),
        )
    )

    # Sort the coords and data - sparse sum requires this to be in order
    array = _sort_2d_sparse_coords_and_data(array)

    # Do the sum which has been optimised for 2d arrays
    final_array = _2d_sparse_sum_axis_1(
        sparse_shape=array.shape,
        sparse_coords=array.coords,
        sparse_data=array.data,
    )
    return final_array.reshape(final_shape)
