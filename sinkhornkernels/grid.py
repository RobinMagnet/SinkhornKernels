"""Gaussian smoothing on voxel grids as a matrix-free kernel.

On a voxel grid the Gaussian kernel is applied using fast separable convolution (``scipy.ndimage.gaussian_filter``).
:class:`GridGaussian` wraps this as a ``LinearOperator`` acting on the *active* voxels only of an occupancy mask.

Note that ``gaussian_filter`` uses a *normalized* (sum-one) convolution
kernel; the multiplicative constant with respect to the raw
$\\exp(-d^2/2\\sigma^2)$ kernel is absorbed by the Sinkhorn
normalization.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator


class GridGaussian(LinearOperator):
    """Separable Gaussian convolution restricted to the active voxels of a mask.

    This class only stores the ``N`` active (True) voxels.
    The matvec is: scatter to the grid, ``gaussian_filter`` with zero padding,
    gather back. This is a symmetric operation.
    Parameters
    ----------
    mask : (n1, ..., nd) bool ndarray
        Voxel occupancy (e.g. from :func:`voxelize_mesh` or :func:`boundary_mask`).
    sigma : float
        Kernel bandwidth in *voxel* units (multiply by the grid spacing to
        get world units, e.g. for
        :func:`sinkhornkernels.spectral.laplacian_eigenvalues`).
    truncate : float
        Radius of the truncated convolution window, in units of ``sigma``
        (``scipy`` default 4.0).

    Attributes
    ----------
    active_indices : (N, d) int ndarray
        Grid indices of the active voxels, in the (C-order) ordering used by
        the operator.
    """

    def __init__(self, mask, sigma, truncate=4.0):
        self.mask = np.asarray(mask, dtype=bool)  # (n1, ..., nd)
        self.sigma = float(sigma)
        self.truncate = float(truncate)
        self._flat_idx = np.flatnonzero(self.mask)  # (N,)
        self.active_indices = np.argwhere(self.mask)  # (N, d)
        N = self._flat_idx.size
        super().__init__(dtype=np.dtype(np.float64), shape=(N, N))

    def to_grid(self, x):
        """Embed an active-voxel vector into the full grid (zeros elsewhere).

        Parameters
        ----------
        x : (N,) ndarray
            Value per active voxel

        Returns
        -------
        g : (n1, ..., nd) ndarray
            Full grid array with zeros outside the active voxels.
        """
        g = np.zeros(self.mask.size, dtype=np.result_type(np.asarray(x).dtype, self.dtype))
        g[self._flat_idx] = np.asarray(x).reshape(-1)
        return g.reshape(self.mask.shape)

    def from_grid(self, g):
        """Extract the active-voxel values from a full grid array.

        Parameters
        ----------
        g : (n1, ..., nd) ndarray
            Full grid array.

        Returns
        -------
        x : (N,) ndarray
            Value per active voxel.
        """
        return np.asarray(g).reshape(-1)[self._flat_idx]

    def _matvec(self, x):
        # Rebuild the full grid, apply the Gaussian filter, and extract the active voxels.
        g = self.to_grid(np.asarray(x).reshape(-1))
        g = ndimage.gaussian_filter(g, sigma=self.sigma, mode="constant", truncate=self.truncate)
        return self.from_grid(g)

    def _rmatvec(self, x):  # symmetric
        return self._matvec(x)


def kde_masses(mask, sigma, truncate=4.0):
    """Voxel masses obtained with KDE $m = 1 / (G_\\sigma \\star 1)$.

    Parameters
    ----------
    mask : (n1, ..., nd) bool ndarray
    sigma : float
        Bandwidth in voxel units. Pass the **same** ``sigma`` as the
        :class:`GridGaussian` kernel this mass is used with.
    truncate : float
        Same convention as :class:`GridGaussian`.

    Returns
    -------
    masses : (N,) ndarray
        One value per active voxel (C-order, matching :class:`GridGaussian`).
    """
    mask = np.asarray(mask, dtype=bool)
    density = ndimage.gaussian_filter(
        mask.astype(np.float64), sigma=float(sigma), mode="constant", truncate=truncate
    )  # (n1, ..., nd)
    return 1.0 / density[mask]  # (N,)


def boundary_mask(mask):
    """One-voxel-thick boundary shell of an occupancy mask.

    A voxel is on the boundary when it is occupied but at least one of its
    ``2 * d`` face neighbors is empty. Voxels on the array border count as
    boundary (out-of-bounds neighbors are treated as empty).

    Equivalently, the shell is the mask minus its morphological interior.
    """
    mask = np.asarray(mask, dtype=bool)
    # connectivity-1 structuring element: the voxel plus its 2*d face neighbors
    face_neighbors = ndimage.generate_binary_structure(mask.ndim, 1)
    # interior voxels survive the erosion (whole face-neighborhood occupied);
    # border_value=0 treats out-of-bounds neighbors as empty, so voxels on the
    # array border are never interior.
    interior = ndimage.binary_erosion(mask, structure=face_neighbors, border_value=0)
    return mask & ~interior


def grid_coordinates(mask, origin=0.0, spacing=1.0):
    """World coordinates of the active voxels (centers), C-order.

    ``origin`` is the world position of the center of voxel ``(0, ..., 0)``.
    """
    ijk = np.argwhere(np.asarray(mask, dtype=bool))  # (N, d) integer voxel indices
    return ijk * np.asarray(spacing) + np.asarray(origin)  # (N, d)


def voxelize_points(points, spacing, origin=None, shape=None, pad=1):
    """Occupancy mask from a point cloud by regular binning.

    Parameters
    ----------
    points : (N, d) ndarray
    spacing : float
        Voxel edge length.
    origin : (d,) ndarray, optional
        Center of voxel ``(0, ..., 0)``. Defaults to
        ``points.min(0) - pad * spacing``.
    shape : tuple of int, optional
        Grid shape; computed from the points and ``pad`` when omitted.
    pad : int
        Number of empty voxel layers kept around the cloud.

    Returns
    -------
    mask : bool ndarray
    origin : (d,) ndarray
    """
    points = np.asarray(points)  # (N, d)
    spacing = float(spacing)
    if origin is None:
        origin = points.min(axis=0) - pad * spacing
    origin = np.asarray(origin, dtype=np.float64)  # (d,)

    # integer grid coordinates of the points, clipped to the grid shape
    idx = np.rint((points - origin) / spacing).astype(np.int64)  # (N, d)
    if shape is None:
        shape = tuple(idx.max(axis=0) + 1 + pad)
    idx = np.clip(idx, 0, np.asarray(shape) - 1)  # (N, d)

    mask = np.zeros(shape, dtype=bool)  # (n1, ..., nd)
    mask[tuple(idx.T)] = True
    return mask, origin


def voxelize_mesh(vertices, faces, density):
    """Solid occupancy mask of a closed triangle mesh.

    Requires ``pyvista`` (installed with the ``[examples]`` extra).

    Parameters
    ----------
    vertices : (N, 3) ndarray
    faces : (m, 3) int ndarray
    density : float
        Voxel edge length.

    Returns
    -------
    mask : (n1, n2, n3) bool ndarray
    origin : (3,) ndarray
        World position of the center of voxel ``(0, 0, 0)``.
    spacing : float
        Equal to ``density``.
    """
    try:
        import pyvista as pv
    except ImportError as e:  # pragma: no cover - exercised only without pyvista
        raise ImportError(
            "voxelize_mesh requires pyvista; install it with `pip install sinkhornkernels[examples]` "
            "or voxelize a sampled point cloud with voxelize_points instead."
        ) from e

    faces = np.asarray(faces)  # (m, 3)
    # pyvista faces format: each face prefixed by its vertex count (3)
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=faces.dtype), faces]
    ).ravel()  # (4 * m,)
    surface = pv.PolyData(np.asarray(vertices), faces_pv)
    voxels = pv.voxelize(surface, density=density, check_surface=False)
    centers = np.asarray(voxels.cell_centers().points, dtype=np.float64)  # (n_vox, 3)

    origin = centers.min(axis=0)  # (3,)
    idx = np.rint((centers - origin) / density).astype(np.int64)  # (n_vox, 3)
    shape = tuple(idx.max(axis=0) + 1)
    idx = np.clip(idx, 0, np.asarray(shape) - 1)  # (n_vox, 3)

    mask = np.zeros(shape, dtype=bool)  # (n1, n2, n3)
    mask[tuple(idx.T)] = True
    return mask, origin, float(density)
