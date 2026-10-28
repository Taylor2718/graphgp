"""Independent block-Gaussian and autodiff checks for the generic JAX API."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphgp import refine_vector as rv
from graphgp.graph import Graph

jax.config.update("jax_enable_x64", True)


def _graph(*, dense=False, reordered=False, dtype=jnp.float64):
    points = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.1],
            [0.0, 0.5, 0.0],
            [0.1, 0.2, 0.5],
            [0.7, 0.5, 0.3],
            [0.6, 0.0, 0.6],
            [0.3, 0.4, 0.8],
        ],
        dtype=dtype,
    )
    if dense:
        points = points[:4]
        neighbors, offsets = [[0, 1, 2]], (3, 4)
    else:
        neighbors, offsets = [[0, 1], [1, 2], [0, 2], [3, 4]], (3, 6, 7)
    indices = jnp.arange(len(points) - 1, -1, -1) if reordered else None
    return Graph(points, jnp.asarray(neighbors, dtype=jnp.int32), offsets, indices)


def _covariance(m, theta=None):
    """Nonstationary and nonseparable SPD kernel from independent latent GPs."""
    if theta is None:
        theta = jnp.zeros(2)

    def cov(x, y):
        channels = jnp.arange(m, dtype=x.dtype)
        lengths = jnp.exp(theta[0]) * (0.35 + channels * 0.09)

        def mixing(p):
            return jnp.eye(m, dtype=x.dtype) + 0.13 * jnp.exp(theta[1]) * jnp.outer(
                jnp.sin(p[0] + 0.4 * channels), jnp.cos(p[2] + 0.3 * channels)
            )

        spatial = jnp.exp(-jnp.sum((x - y) ** 2) / (2 * lengths**2))
        return (mixing(x) * spatial) @ mixing(y).T + 0.03 * jnp.eye(m, dtype=x.dtype) * jnp.all(x == y)

    return cov


def _noise(graph, m):
    return jnp.sin(jnp.arange(len(graph.points) * m, dtype=graph.points.dtype)).reshape(-1, m)


@pytest.mark.parametrize("m", [1, 3, 6])
@pytest.mark.parametrize("fast_jit", [False, True])
def test_complete_conditionals_equal_dense_block_cholesky(m, fast_jit):
    graph = _graph(dense=True)
    cov = _covariance(m)
    xi = _noise(graph, m)
    # Independent assembly: blocks in the same point-major order as xi.ravel().
    matrix = np.block([[np.asarray(cov(x, y)) for y in graph.points] for x in graph.points])
    expected = (np.linalg.cholesky(matrix) @ np.asarray(xi).ravel()).reshape(xi.shape)
    actual = jax.jit(lambda x: rv.generate(graph, cov, x, fast_jit=fast_jit))(xi)
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    np.testing.assert_allclose(rv.compute_cov_matrix(cov, graph.points, graph.points), matrix)
    np.testing.assert_allclose(rv.generate_dense(graph.points, cov, xi), expected, atol=2e-13)
    np.testing.assert_allclose(rv.generate_dense_inv(graph.points, cov, actual), xi, atol=2e-13)


@pytest.mark.parametrize("m", [1, 3, 6])
@pytest.mark.parametrize("reordered", [False, True])
def test_irregular_batches_inverse_logdet_and_transpose(m, reordered):
    graph = _graph(reordered=reordered)
    cov = _covariance(m)
    xi = _noise(graph, m)
    forward = jax.jit(lambda x: rv.generate(graph, cov, x, jitter=1e-6))
    values = forward(xi)
    slow = jax.jit(lambda x: rv.generate(graph, cov, x, fast_jit=False, jitter=1e-6))(xi)
    np.testing.assert_allclose(values, slow, atol=3e-13)
    inverse = jax.jit(lambda v: rv.generate_inv(graph, cov, v, jitter=1e-6))(values)
    np.testing.assert_allclose(inverse, xi, atol=3e-13)

    cotangent = jnp.cos(xi)
    transpose = jax.linear_transpose(forward, xi)(cotangent)[0]
    np.testing.assert_allclose(jnp.vdot(values, cotangent), jnp.vdot(xi, transpose), atol=3e-13)

    jacobian = jax.jacfwd(forward)(jnp.zeros_like(xi)).reshape(xi.size, xi.size)
    logdet = jax.jit(lambda: rv.generate_logdet(graph, cov, jitter=1e-6))()
    np.testing.assert_allclose(logdet, np.linalg.slogdet(np.asarray(jacobian))[1], atol=4e-13)


@pytest.mark.parametrize("fast_jit", [False, True])
def test_covariance_jvp_vjp_and_finite_difference(fast_jit):
    graph = _graph(reordered=True)
    xi = _noise(graph, 6)
    theta = jnp.asarray([0.04, -0.12])
    direction = jnp.asarray([0.31, -0.72])
    forward = jax.jit(lambda t: rv.generate(graph, _covariance(6, t), xi, fast_jit=fast_jit))
    values, tangent = jax.jvp(forward, (theta,), (direction,))
    eps = 1e-5
    finite_difference = (forward(theta + eps * direction) - forward(theta - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-7, atol=1e-9)
    cotangent = jnp.cos(values)
    _, pullback = jax.vjp(forward, theta)
    gradient = pullback(cotangent)[0]
    np.testing.assert_allclose(jnp.vdot(cotangent, tangent), jnp.vdot(gradient, direction), atol=2e-12)

    # The inverse and determinant also differentiate through the same covariance.
    for objective in (
        lambda t: rv.generate_logdet(graph, _covariance(6, t)),
        lambda t: jnp.sum(rv.generate_inv(graph, _covariance(6, t), values) ** 2),
    ):
        objective = jax.jit(objective)
        actual = jnp.vdot(jax.grad(objective)(theta), direction)
        expected = (objective(theta + eps * direction) - objective(theta - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-9)


def test_dense_only_empty_and_zero_neighbors():
    points = _graph().points[:3]
    cov = _covariance(3)
    for n0, offsets in ((3, (3,)), (1, (1, 3)), (0, (0, 2, 3))):
        graph = Graph(points, jnp.empty((3 - n0, 0), dtype=jnp.int32), offsets)
        xi = _noise(graph, 3)
        for fast in (False, True):
            values = jax.jit(partial(rv.generate, graph, cov, fast_jit=fast))(xi)
            np.testing.assert_allclose(rv.generate_inv(graph, cov, values), xi, atol=2e-13)
            assert np.isfinite(rv.generate_logdet(graph, cov))
    empty = Graph(points[:0], jnp.empty((0, 0), dtype=jnp.int32), (0,))
    xi = jnp.empty((0, 3))
    assert rv.generate(empty, cov, xi).shape == xi.shape
    assert rv.generate_inv(empty, cov, xi).shape == xi.shape
    assert rv.generate_logdet(empty, cov) == 0


def test_float32_and_mixed_input_dtype():
    graph = _graph(dtype=jnp.float32)
    cov = _covariance(3, jnp.zeros(2, dtype=jnp.float32))
    xi = _noise(graph, 3)
    fast = rv.generate(graph, cov, xi)
    assert fast.dtype == jnp.float32
    np.testing.assert_allclose(fast, rv.generate(graph, cov, xi, fast_jit=False), atol=2e-6)

    # Covariance in float64 must not be silently narrowed to the noise dtype.
    def cov64(x, y):
        return cov(x, y).astype(jnp.float64)

    assert rv.generate(graph, cov64, xi).dtype == jnp.float64


def test_invalid_shapes_are_rejected():
    graph = _graph()
    xi = _noise(graph, 3)
    with pytest.raises(ValueError, match="shape"):
        rv.generate(graph, _covariance(3), xi[:, 0])
    with pytest.raises(ValueError, match="component count"):
        rv.generate(graph, _covariance(6), xi)
    with pytest.raises(ValueError, match="square"):
        rv.generate(graph, lambda x, y: jnp.ones((3, 2)), xi)
    with pytest.raises(TypeError, match="callable"):
        rv.generate(graph, (jnp.arange(3), jnp.ones(3)), xi)
    with pytest.raises(ValueError, match="same component count"):
        rv.refine(graph.points, graph.neighbors, graph.offsets, _covariance(3), xi[:3], jnp.ones((4, 6)))
    with pytest.raises(ValueError, match="offsets"):
        rv.refine(graph.points, graph.neighbors, (3, 6), _covariance(3), xi[:3], xi[3:])
