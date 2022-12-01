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

    # Get just the dimensions from target that are in array and flatten them
    flat_target_coord = np.ravel_multi_index(
        target_array.coords[array_dims, :],
        np.take(target_array.shape, array_dims),
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
    return sparse.COO(
        data=broadcast_data,
        coords=target_array.coords,
        shape=target_array.shape,
        fill_value=target_array.fill_value,
        sorted=True,
    )


def sparse_sum(sparse_array: sparse.COO, axis: Iterable[int]) -> sparse.COO:
    """Faster sum for a sparse.COO matrix

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
    new_axes = list(keep_axis) + axis
    array = sparse_array.copy()
    array.coords = array.coords[new_axes, :]
    array.shape = tuple(array.shape[ax] for ax in new_axes)

    # Reshape into the 2d array
    array = array.reshape(
        (
            np.prod(final_shape, dtype=np.intp),
            np.prod(remove_shape, dtype=np.intp),
        )
    )

    # Sort the coords and data
    # This would have been done already if we used sparse_array.transpose().
    # It does actually work out faster in the long run!
    stacked = np.vstack((array.coords, array.data))
    idx = stacked[0, :].argsort()
    stacked = np.take(stacked, idx, axis=1)
    array.coords = stacked[:2].astype(sparse_array.coords.dtype)
    array.data = stacked[2].astype(sparse_array.data.dtype)

    # Do the sum which has been optimised for 2d arrays
    final_array = _2d_sparse_sum_axis_1(
        sparse_shape=array.shape,
        sparse_coords=array.coords,
        sparse_data=array.data,
    )
    return final_array.reshape(final_shape)
