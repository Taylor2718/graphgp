"""JAX refinement for real vector fields with arbitrary component covariance.

This module mirrors :mod:`graphgp.refine`: the same ``Graph`` describes spatial
dependencies, while noise and values have shape ``(N, m)``. The only covariance
interface is a callable ``covariance(x, y) -> (m, m)`` for two spatial points.
The callable must define a positive-definite block covariance with
``C(x, y) = C(y, x).T``. It may be anisotropic, nonstationary, and nonseparable.

Functions are ordinary JAX programs, including their linear transpose and
covariance JVP/VJP. Capture the callable in a closure (or a ``jax.tree_util.Partial``)
when JIT compiling. No CUDA extension or radial discretization is required.
Log determinants refer to the noise-to-field map, i.e. half the log determinant
of the implied field covariance. ``jitter`` adds only to conditional variances;
stabilize the dense seed through the supplied covariance if necessary.
"""

from collections.abc import Callable
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.scipy.linalg import solve_triangular

from .graph import Graph

Covariance = Callable[[Array, Array], Array]


def _check_values(points, values, name):
    if points.ndim != 2:
        raise ValueError("points must have shape (N, d).")
    if values.ndim != 2 or values.shape[0] != len(points) or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N, m) with m >= 1.")


def _check_structure(points, neighbors, offsets):
    if points.ndim != 2 or neighbors.ndim != 2:
        raise ValueError("points and neighbors must be two-dimensional arrays.")
    if not jnp.issubdtype(neighbors.dtype, jnp.integer):
        raise ValueError("neighbors must contain integer indices.")
    n0 = len(points) - len(neighbors)
    offsets = np.asarray(offsets)
    if (
        n0 < 0
        or offsets.ndim != 1
        or len(offsets) == 0
        or not np.issubdtype(offsets.dtype, np.integer)
        or offsets[0] != n0
        or offsets[-1] != len(points)
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError("offsets must run monotonically from N - len(neighbors) to N.")
    return n0


def compute_cov_matrix(covariance: Covariance, points_a: Array, points_b: Array) -> Array:
    """Assemble point-major covariance blocks into ``(Na*m, Nb*m)``."""
    if not callable(covariance):
        raise TypeError("covariance must be callable: (x, y) -> (m, m).")
    if points_a.ndim != 2 or points_b.ndim != 2 or points_a.shape[1] != points_b.shape[1]:
        raise ValueError("points_a and points_b must have shapes (Na, d) and (Nb, d).")
    blocks = jax.vmap(lambda x: jax.vmap(lambda y: covariance(x, y))(points_b))(points_a)
    if blocks.ndim != 4 or blocks.shape[-2] != blocks.shape[-1] or blocks.shape[-1] < 1:
        raise ValueError("covariance(x, y) must return a square (m, m) matrix with m >= 1.")
    m = blocks.shape[-1]
    return blocks.transpose(0, 2, 1, 3).reshape(len(points_a) * m, len(points_b) * m)


def _dense_factor(points, covariance, m=None):
    matrix = compute_cov_matrix(covariance, points, points)
    if m is not None and matrix.shape != (len(points) * m, len(points) * m):
        raise ValueError("covariance component count must match the values/noise component count.")
    return jnp.linalg.cholesky(matrix)


def generate_dense(points: Array, covariance: Covariance, xi: Array) -> Array:
    """Generate a dense vector GP; points and noise are in matching order."""
    _check_values(points, xi, "xi")
    factor = _dense_factor(points, covariance, xi.shape[1])
    return (factor @ xi.reshape(-1)).reshape(xi.shape)


def generate_dense_inv(points: Array, covariance: Covariance, values: Array) -> Array:
    """Invert :func:`generate_dense`."""
    _check_values(points, values, "values")
    factor = _dense_factor(points, covariance, values.shape[1])
    return solve_triangular(factor, values.reshape(-1), lower=True).reshape(values.shape)


def generate_dense_logdet(points: Array, covariance: Covariance) -> Array:
    """Return the log determinant of the dense noise-to-field map."""
    return jnp.log(jnp.diag(_dense_factor(points, covariance))).sum()


def _local_factors(covariance, joint_points, *, k, m, jitter):
    matrix = compute_cov_matrix(covariance, joint_points, joint_points)
    if matrix.shape != ((k + 1) * m, (k + 1) * m):
        raise ValueError("covariance component count must match the values/noise component count.")
    km = k * m
    if k:
        a = jnp.linalg.solve(matrix[:km, :km], matrix[:km, km:])
        mean_operator = a.T
        conditional = matrix[km:, km:] - matrix[km:, :km] @ a
    else:
        mean_operator = jnp.empty((m, 0), dtype=matrix.dtype)
        conditional = matrix
    conditional = conditional + jnp.asarray(jitter, dtype=matrix.dtype) * jnp.eye(m, dtype=matrix.dtype)
    return mean_operator, jnp.linalg.cholesky(conditional)


def _all_factors(points, neighbors, covariance, m, jitter):
    n0 = len(points) - len(neighbors)
    joint_points = jnp.concatenate((points[neighbors], points[n0:, None]), axis=1)
    return jax.vmap(lambda p: _local_factors(covariance, p, k=neighbors.shape[1], m=m, jitter=jitter))(joint_points)


def refine(
    points: Array,
    neighbors: Array,
    offsets: tuple[int, ...],
    covariance: Covariance,
    initial_values: Array,
    xi: Array,
    *,
    fast_jit: bool = True,
    jitter: float = 0.0,
) -> Array:
    """Conditionally refine seed values using the existing spatial graph.

    ``initial_values`` is ``(offsets[0], m)`` and ``xi`` is ``(N-offsets[0], m)``.
    ``offsets`` must be static under JIT. The fast path precomputes conditional
    matrices and scans padded depth batches; the other path unrolls the batches.
    """
    n0 = _check_structure(points, neighbors, offsets)
    _check_values(points[:n0], initial_values, "initial_values")
    _check_values(points[n0:], xi, "xi")
    m = initial_values.shape[1]
    if xi.shape[1] != m:
        raise ValueError("initial_values and xi must have the same component count.")
    if len(neighbors) == 0:
        return initial_values
    k = neighbors.shape[1]
    if not fast_jit:
        values = initial_values
        for start, end in pairwise(offsets):
            if start == end:
                continue
            ns = neighbors[start - n0 : end - n0]
            joint = jnp.concatenate((points[ns], points[start:end, None]), axis=1)
            mean, factor = jax.vmap(lambda p: _local_factors(covariance, p, k=k, m=m, jitter=jitter))(joint)
            coarse = values[ns].reshape(end - start, k * m)
            next_values = jnp.einsum("bim,bm->bi", mean, coarse) + jnp.einsum(
                "bij,bj->bi", factor, xi[start - n0 : end - n0]
            )
            values = jnp.concatenate((values, next_values), axis=0)
        return values

    mean, factor = _all_factors(points, neighbors, covariance, m, jitter)
    sizes = np.diff(np.asarray(offsets))
    width = int(sizes.max())
    dtype = jnp.result_type(initial_values, xi, mean, factor)
    # Store each unprocessed point's independent conditional noise in the carry;
    # replace it by its final value when its depth is visited. This also avoids
    # captured linear inputs in scan, which break direct linear_transpose on
    # JAX 0.9.1, without introducing a custom derivative or extra batch storage.
    noise = jnp.einsum("bij,bj->bi", factor, xi)
    values = jnp.pad(jnp.concatenate((initial_values.astype(dtype), noise)), ((0, width), (0, 0)))
    ns = jnp.pad(neighbors, ((0, width), (0, 0)))
    mean = jnp.pad(mean, ((0, width), (0, 0), (0, 0)))
    starts = jnp.asarray(offsets[:-1])

    def step(values, start_size):
        start, size = start_size
        offset = start - n0
        batch_neighbors = lax.dynamic_slice(ns, (offset, 0), (width, k))
        coarse = values[batch_neighbors].reshape(width, k * m)
        operator = lax.dynamic_slice(mean, (offset, 0, 0), (width, m, k * m))
        previous = lax.dynamic_slice(values, (start, 0), (width, m))
        generated = jnp.einsum("bim,bm->bi", operator, coarse) + previous
        generated = jnp.where((jnp.arange(width) < size)[:, None], generated, previous)
        return lax.dynamic_update_slice(values, generated, (start, 0)), None

    values, _ = lax.scan(step, values, (starts, jnp.asarray(sizes)))
    return values[: len(points)]


def refine_inv(
    points: Array,
    neighbors: Array,
    offsets: tuple[int, ...],
    covariance: Covariance,
    values: Array,
    *,
    fast_jit: bool = True,
    jitter: float = 0.0,
) -> tuple[Array, Array]:
    """Invert refinement, returning seed values and refinement noise.

    ``fast_jit`` is accepted for API symmetry; all conditionals can be inverted
    simultaneously because the complete field is available.
    """
    n0 = _check_structure(points, neighbors, offsets)
    _check_values(points, values, "values")
    if len(neighbors) == 0:
        return values, values[:0]
    m = values.shape[1]
    mean, factor = _all_factors(points, neighbors, covariance, m, jitter)
    coarse = values[neighbors].reshape(len(neighbors), neighbors.shape[1] * m)
    residual = values[n0:] - jnp.einsum("bim,bm->bi", mean, coarse)
    xi = jax.vmap(lambda f, r: solve_triangular(f, r, lower=True))(factor, residual)
    return values[:n0], xi


def refine_logdet(
    points: Array,
    neighbors: Array,
    offsets: tuple[int, ...],
    covariance: Covariance,
    *,
    jitter: float = 0.0,
) -> Array:
    """Log determinant of the conditional noise-to-field map."""
    _check_structure(points, neighbors, offsets)
    if len(neighbors) == 0:
        return jnp.zeros((), dtype=points.dtype)
    c00 = jnp.asarray(covariance(points[0], points[0]))
    if c00.ndim != 2 or c00.shape[0] != c00.shape[1] or c00.shape[0] < 1:
        raise ValueError("covariance(x, y) must return a square (m, m) matrix with m >= 1.")
    _, factor = _all_factors(points, neighbors, covariance, c00.shape[0], jitter)
    return jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)).sum()


def _restore_order(values, graph):
    if graph.indices is not None:
        return jnp.empty_like(values).at[graph.indices].set(values, unique_indices=True)
    return values


def generate(
    graph: Graph,
    covariance: Covariance,
    xi: Array,
    *,
    fast_jit: bool = True,
    jitter: float = 0.0,
) -> Array:
    """Generate ``(N, m)`` values in the original order of the graph's points."""
    _check_values(graph.points, xi, "xi")
    n0 = _check_structure(graph.points, graph.neighbors, graph.offsets)
    if graph.indices is not None:
        xi = xi[graph.indices]
    initial = generate_dense(graph.points[:n0], covariance, xi[:n0])
    values = refine(
        graph.points, graph.neighbors, graph.offsets, covariance, initial, xi[n0:], fast_jit=fast_jit, jitter=jitter
    )
    return _restore_order(values, graph)


def generate_inv(
    graph: Graph,
    covariance: Covariance,
    values: Array,
    *,
    fast_jit: bool = True,
    jitter: float = 0.0,
) -> Array:
    """Invert :func:`generate`, retaining the original point and component order."""
    _check_values(graph.points, values, "values")
    n0 = _check_structure(graph.points, graph.neighbors, graph.offsets)
    if graph.indices is not None:
        values = values[graph.indices]
    initial, xi = refine_inv(
        graph.points, graph.neighbors, graph.offsets, covariance, values, fast_jit=fast_jit, jitter=jitter
    )
    xi0 = generate_dense_inv(graph.points[:n0], covariance, initial)
    return _restore_order(jnp.concatenate((xi0, xi)), graph)


def generate_logdet(graph: Graph, covariance: Covariance, *, jitter: float = 0.0) -> Array:
    """Log absolute determinant of :func:`generate`, including its dense seed."""
    n0 = _check_structure(graph.points, graph.neighbors, graph.offsets)
    return generate_dense_logdet(graph.points[:n0], covariance) + refine_logdet(
        graph.points, graph.neighbors, graph.offsets, covariance, jitter=jitter
    )
