# VIPO

**Voxel-Interleaved Point Ordering for Budgeted Neighborhood Queries**

VIPO builds a deterministic permutation of a point set. It assigns points to
fixed-width voxels, preserves input order inside each voxel, and visits the
occupied voxels in lexicographic order once per pass. A shorter prefix is
contained in each longer prefix of the same permutation.

Requires Python 3 and NumPy.

```python
from vipo import voxel_interleaved_order

order = voxel_interleaved_order(xyz, cell_width=2.0)
retained_row_ids = order[:budget]
retained_xyz = xyz[retained_row_ids]
```

Here `xyz` is your N-by-3 point array in a local metric coordinate frame and
`budget` is the desired point count. Build the permutation once and reuse its
prefixes as inputs to your chosen neighborhood-search engine.

License: Apache-2.0.
