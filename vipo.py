# SPDX-License-Identifier: Apache-2.0
"""Voxel-interleaved point ordering for nested point-count budgets."""

import math

import numpy as np


def voxel_interleaved_order(xyz, cell_width=2.0):
    """Return row indices in rank-major, lexicographic voxel order.

    The coordinates are assumed to be in a local metric frame. Rows inside
    each voxel retain their input order. Coordinate precision is preserved
    during voxel assignment, so supply the intended dtype before calling.
    """
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fiu":
        raise ValueError("xyz must be an N by 3 array of real coordinates")
    if not math.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("cell_width must be positive and finite")
    if not len(points):
        return np.empty(0, dtype=np.int64)
    cell = np.floor(points / cell_width)
    if not np.all(np.isfinite(cell) & (cell >= -(2.0**63)) & (cell < 2.0**63)):
        raise ValueError("voxel indices must fit in int64")
    _, group = np.unique(cell.astype(np.int64), axis=0, return_inverse=True)
    group = group.reshape(-1)
    rows = np.arange(len(points), dtype=np.int64)
    grouped = np.lexsort((rows, group))
    in_cell_order = group[grouped]
    starts = np.r_[0, np.flatnonzero(in_cell_order[1:] != in_cell_order[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(points)])
    rank = rows - np.repeat(starts, counts)
    return grouped[np.lexsort((in_cell_order, rank))]
