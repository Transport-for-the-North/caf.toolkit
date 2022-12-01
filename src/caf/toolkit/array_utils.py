# -*- coding: utf-8 -*-
"""Collection of utility functions for numpy and sparse arrays """
# Built-Ins
import logging

from typing import Iterable
from typing import Collection

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
@nb.njit(parallel=True)
def _get_axis_lengths(shape: np.ndarray) -> np.ndarray:
    """Get the length of each axis when an array is flattened.

    Parameters
    ----------
    shape:
        The shape of the array that would be flattened.

    Returns
    -------
    axis_lengths:
        An array of length

    See Also
    --------
    `flat_array_idx()`
    `_flat_array_idx_from_lengths()`
    """
    # Get the length of each axis
    axis_lengths = np.zeros(len(shape), dtype=np.int64)
    # Disabling pylint warning, see https://github.com/PyCQA/pylint/issues/2910
    for i in nb.prange(len(shape)):  # pylint: disable=not-an-iterable
        length = 1
        for item in shape[i + 1 :]:
            length *= item
        axis_lengths[i] = length
    return axis_lengths


@nb.njit
def _flat_array_idx_from_lengths(
    index: np.ndarray,
    axis_lengths: np.ndarray,
) -> int:
    """Translate an N-dimensional index into a flat array index.

    Parameters
    ----------
    index:
        The N-dimensional index in an array of `shape` to be translated.

    axis_lengths:
        The length of each axis, in order. See `_get_axis_lengths()`
    Returns
    -------
    axis_lengths:
        An array of length

    See Also
    --------
    `flat_array_idx()`
    `_get_axis_lengths()`
    """
    # Get the index
    idx = 0
    for axis_index, axis_len in zip(index, axis_lengths):
        idx += axis_index * axis_len
    return idx


@nb.njit(fastmath=True, parallel=True)
def _sparse_sum_data(
    sparse_data: np.ndarray,
    sparse_coords: np.ndarray,
    sparse_shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # OPTIMIZE: Could we figure out how to parallelize by knowing which
    #  coords will need to access the same idx?
    # Get the size and shape of final array

    # Init
    axis_lengths = _get_axis_lengths(sparse_shape)
    idx_map = {_flat_array_idx_from_lengths(x, axis_lengths): x for x in sparse_coords}

    # Sum the data where the axis are the same
    res: dict[int, int] = dict()
    for val, coords_idx in zip(sparse_data, sparse_coords):
        idx = _flat_array_idx_from_lengths(
            index=coords_idx,
            axis_lengths=axis_lengths,
        )
        if idx in res:
            res[idx] += val
        else:
            res[idx] = val

    # Map the idx back to coords
    keys = list(res.keys())
    key_len = len(keys)
    res_coords = np.zeros((sparse_coords.shape[1], key_len), dtype=sparse_coords.dtype)
    # Disabling pylint warning, see https://github.com/PyCQA/pylint/issues/2910
    for i in nb.prange(key_len):  # pylint: disable=not-an-iterable
        res_coords[:, i] = idx_map[keys[i]]

    return res_coords, np.array(list(res.values()))


@nb.njit(parallel=True)
def _sparse_sum(
    sparse_shape: tuple[int, ...],
    sparse_coords: np.ndarray,
    sparse_data: np.ndarray,
    keep_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Internal function of sparse_sum(). Used with numba for speed"""
    # Set up the loop
    res_coords = np.zeros((len(sparse_data), len(keep_axis)), dtype=np.int32)

    # Disabling pylint warning, see https://github.com/PyCQA/pylint/issues/2910
    for i in nb.prange(len(sparse_data)):  # pylint: disable=not-an-iterable
        res_coords[i, :] = sparse_coords[:, i].take(keep_axis)

    return _sparse_sum_data(
        sparse_data=sparse_data,
        sparse_coords=res_coords,
        sparse_shape=np.take(np.array(sparse_shape), keep_axis),
    )


@nb.njit(fastmath=True, nogil=True, cache=True)
def _get_unique_idxs(groups: np.ndarray) -> np.ndarray:
    """Get the index positions of the start of each group

    Parameters
    ----------
    groups:
        The groups that each element in an array of the same shape belongs to.
        This should be a 1d array.

    Returns
    -------
    unique_idxs:
        The index of the first element where each group is found.
    """
    inv_idx = [0]
    last_group = groups[0]
    for i in range(1, len(groups)):
        if groups[i] != last_group:
            inv_idx.append(i)
            last_group = groups[i]

    return np.array(inv_idx, dtype=groups.dtype)


def _2d_sparse_sum_axis_1(
    sparse_shape: tuple[int, ...],
    sparse_coords: np.ndarray,
    sparse_data: np.ndarray,
) -> np.ndarray:
    """Internal function of sparse_sum(). Used with numba for speed"""
    unique_idxs = _get_unique_idxs(sparse_coords[0])
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

    # OPTIMISE???
    # res = np.zeros(sparse_shape[0])
    # for i in nb.prange(len(sparse_data)):
    #     idx = sparse_coords[0, i]
    #     res[idx] += sparse_data[i]
    # return res


# ## Public functions ## #
def flat_array_idx(
    shape: Collection[int],
    index: Collection[int],
) -> int:
    """Translate an N-dimensional index into a flat array index of same size

    Parameters
    ----------
    shape:
        The shape of the array that would be flattened.

    index:
        The N-dimensional index in an array of `shape` to be translated.

    Returns
    -------
    flat_index:
        The equivalent position of `index` in a flattened array.
        So that `array.flatten()[flat_index]`
        Is equivalent to `array[index]`
        Where array is of shape `shape`
    """
    return _flat_array_idx_from_lengths(
        index=np.array(index),
        axis_lengths=_get_axis_lengths(np.array(shape)),
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
