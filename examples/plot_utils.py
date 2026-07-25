"""Backend-aware plotting helpers for the example notebooks.

Plots support two backends:

- ``"matplotlib"`` : 3D scatter plots (no extra dependencies).
- ``"pyvista"``    : interactive 3D rendering (voxels as cubes, GMM as
  ellipsoids).

Choose the default with :func:`set_backend`, or override per call via the
``backend=`` argument of :func:`plot_panels`::

    import plot_utils as plu
    plu.set_backend("pyvista")          # or "matplotlib"
    plu.plot_panels([
        plu.Mesh(vertices, faces, evec_mesh, title="mesh"),
        plu.Points(points, evec_pc, title="point cloud"),
    ])
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_VALID_BACKENDS = ("matplotlib", "pyvista")
_BACKEND = "pyvista"
_pv_initialized = False


def set_backend(name):
    """Set the default rendering backend (``"matplotlib"`` or ``"pyvista"``)."""
    global _BACKEND
    if name not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {name!r}")
    _BACKEND = name


def get_backend():
    """Return the current default rendering backend."""
    return _BACKEND


def _pv():
    """Lazily import pyvista and set the Jupyter backend once.

    pyvista is an optional dependency (the ``matplotlib`` backend does not need
    it); it ships with the ``examples`` extra.
    """
    global _pv_initialized
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            'the "pyvista" backend requires pyvista; install the examples extra '
            'with `pip install -e ".[examples]"`, or use '
            'plot_utils.set_backend("matplotlib").'
        ) from exc

    if not _pv_initialized:
        pv.set_jupyter_backend("trame")
        _pv_initialized = True
    return pv


# ---------------------------------------------------------------------------
# Mesh / array helpers (pyvista)
# ---------------------------------------------------------------------------


class ToyMesh:
    """
    A toy mesh class with minimal attributes to be compatible with the plotting functions.
    """

    def __init__(self, vertices, faces=None):
        self.vertices = vertices
        self.faces = faces

    @property
    def faces_extrinsic(self):
        """Return the faces of the mesh in extrinsic coordinates."""
        return self.faces

    @property
    def n_faces(self):
        """Return the number of faces in the mesh."""
        return 0 if self.faces is None else self.faces.shape[0]

    @property
    def n_vertices(self):
        """Return the number of vertices in the mesh."""
        return self.vertices.shape[0]


def normalize(f, vmin=0, vmax=1):
    """
    Normalize a function or a set of functions between vmin and vmax

    Parameters
    ----------------------------
    f : (n,) or (n,p) - one or multiple functions
    vmin : minimum value for the normalized function(s)
    vmax : maximum value for the normalized function(s)

    Output
    ---------------------------
    f_normalized : (n,) or (n,p) - normalized function(s)
    """
    if f.ndim == 1:
        f_norm = f - np.min(f)
        f_norm = vmin + (vmax - vmin) * f_norm / np.max(f_norm)

    else:
        f_norm = f - np.min(f, axis=0, keepdims=True)
        f_norm = vmin + (vmax - vmin) * f_norm / np.max(f_norm, axis=0, keepdims=True)

    return f_norm


def load_texture(texture):
    pv = _pv()
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")

    if texture is None:
        texture = "texture_1.jpg"

    if os.path.isfile(texture):
        texture_path = texture

    elif os.path.isfile(os.path.join(data_dir, texture)):
        texture_path = os.path.join(data_dir, texture)

    else:
        raise ValueError(f"Texture file {texture} not found")

    return pv.read_texture(texture_path)


def triangles_to_cells(faces):
    """
    Convert list of faces to cells

    Parameters
    ------------------------------
    faces : np.ndarray or list
        (n,3) - list of faces

    Output
    ------------------------------
    cells : np.ndarray
        (n,4) - list of cells
    """
    if faces is None:
        return None
    cells = np.zeros((len(faces), 4), dtype=int)
    cells[:, 0] = 3
    cells[:, 1:] = faces

    return cells


def toPV(mesh=None, vertices=None, faces=None, cmap=None, vfield=None, uv=None):
    """
    Convert a mesh (or raw vertices/faces) to a pyvista.PolyData object

    Parameters
    ------------------------------
    mesh       : object with ``vertices`` / ``faces_extrinsic`` attributes
        mesh object to convert
    cmap       : np.ndarray or list
        (m|n,) or (m|n, 3) - scalar or RGB values for each face or vertex
    vfield     : np.ndarray or list
        (m,3) or (n,3) - vector field for each face or vertex

    Output
    ------------------------------
    pv_mesh : pyvista.PolyData
        mesh object in pyvista format
    """
    pv = _pv()
    if mesh is not None:
        mesh_pv = pv.PolyData(mesh.vertices, triangles_to_cells(mesh.faces_extrinsic))
    elif vertices is not None:
        if faces is None:
            mesh_pv = pv.PolyData(vertices)
        else:
            mesh_pv = pv.PolyData(vertices, triangles_to_cells(faces))
    else:
        raise ValueError("Either mesh or vertices must be provided")

    if cmap is not None:
        if cmap.shape[0] == mesh.n_vertices:
            mesh_pv.point_data["cmap"] = cmap  # (n,) or (n,3)
        else:
            assert cmap.shape[0] == mesh.n_faces
            mesh_pv.cell_data["cmap"] = cmap  # (m,)  or (m,3)

    if vfield is not None:
        if vfield.shape[0] == mesh.n_vertices:
            mesh_pv.point_data.set_vectors(vfield, name="vfield")
        else:
            assert vfield.shape[0] == mesh.n_faces
            mesh_pv.cell_data.set_vectors(vfield, name="vfield")

    if uv is not None:
        mesh_pv.active_texture_coordinates = uv

    return mesh_pv


def plot(
    mesh,
    cmap=None,
    wireframe=False,
    line_width=None,
    texture=None,
    interpolate_before_map=True,
    point_size=None,
    uv=None,
    show_colorbar=False,
    smooth=False,
    opacity=1,
    colormap="viridis",
    clim=None,
    pl=None,
    return_plot=False,
):
    """
    Plot a mesh or point cloud with pyvista

    Parameters
    ------------------------------
    mesh       : mesh object to plot (``ToyMesh`` or similar)
    point_size : float - size of the points
    cmap       : np.ndarray or list - (n,) or (n,3) - scalar or RGB values for each face or vertex
    wireframe  : bool - whether to show the mesh as a wireframe
    line_width : float - width of the wireframe
    show_colorbar : bool - whether to show the colorbar
    smooth     : bool - whether to use smooth shading
    colormap   : str - colormap to use
    pl         : pyvista.Plotter - existing plotter to draw into
    return_plot : bool - whether to return the plotter

    Output
    ------------------------------
    pyvista.Plotter (when ``pl`` is provided or ``return_plot`` is True)
    """
    pv = _pv()
    is_pointcloud = mesh.n_faces == 0
    if uv is not None:
        if cmap is not None:
            print("WARNING: UV and cmap are both activated. Using UV only")
        cmap = None
    mesh_pv = toPV(mesh, cmap=cmap, uv=uv)

    if uv is not None:
        texture = load_texture(texture)

    scalars = None
    if cmap is not None:
        cmap = np.asarray(cmap)
        scalars = "cmap"
        if cmap.ndim == 1:
            is_rgb_cmap = False
        elif cmap.ndim == 2:
            is_rgb_cmap = True
        else:
            raise ValueError("cmap must be either (n,) or (n,3)")
    else:
        is_rgb_cmap = False

    show_plot = False
    if pl is None:
        show_plot = True if not return_plot else False
        pl = pv.Plotter()
    if is_pointcloud:
        pl.add_points(
            mesh_pv,
            scalars=scalars,
            clim=clim,
            cmap=colormap,
            rgb=is_rgb_cmap,
            render_points_as_spheres=True,
            style="points",
            point_size=point_size,
            show_scalar_bar=show_colorbar,
        )
    else:
        pl.add_mesh(
            mesh_pv,
            scalars=scalars,
            clim=clim,
            cmap=colormap,
            rgb=is_rgb_cmap,
            color="white",
            interpolate_before_map=interpolate_before_map,
            point_size=None,
            show_edges=wireframe,
            line_width=line_width,
            smooth_shading=smooth,
            texture=texture,
            show_scalar_bar=show_colorbar,
            opacity=opacity,
        )

    if show_plot:
        pl.show()
    else:
        return pl


# ---------------------------------------------------------------------------
# Panels: one description, two backends
# ---------------------------------------------------------------------------


class _Panel:
    """Base class: a titled scalar field on some geometry.

    ``points`` returns the representative point set matplotlib scatters;
    ``render_pv`` draws the (richer) pyvista geometry into a subplot.
    """

    def __init__(self, scalar=None, size=3.0, title=None):
        self.scalar = None if scalar is None else np.asarray(scalar)
        self.size = size
        self.title = title

    @property
    def points(self):
        raise NotImplementedError

    def render_pv(self, pl, cmap, clim):
        raise NotImplementedError


class Points(_Panel):
    """A point cloud coloured by a scalar field."""

    def __init__(self, pts, scalar=None, size=3.0, pv_point_size=8.0, title=None):
        super().__init__(scalar, size, title)
        self._pts = np.asarray(pts)
        self.pv_point_size = pv_point_size

    @property
    def points(self):
        return self._pts

    def render_pv(self, pl, cmap, clim):
        plot(
            ToyMesh(self._pts),
            cmap=self.scalar,
            colormap=cmap,
            clim=clim,
            point_size=self.pv_point_size,
            pl=pl,
        )


class Mesh(_Panel):
    """A triangle mesh coloured by a per-vertex scalar field."""

    def __init__(self, vertices, faces, scalar=None, size=1.0, title=None):
        super().__init__(scalar, size, title)
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)

    @property
    def points(self):
        return self.vertices

    def render_pv(self, pl, cmap, clim):
        plot(
            ToyMesh(self.vertices, self.faces),
            cmap=self.scalar,
            colormap=cmap,
            clim=clim,
            smooth=True,
            pl=pl,
        )


class Voxels(_Panel):
    """A voxel occupancy set drawn as cubes (pyvista) / points (matplotlib)."""

    def __init__(self, centers, spacing, scalar=None, size=6.0, title=None):
        super().__init__(scalar, size, title)
        self.centers = np.asarray(centers)
        self.spacing = spacing

    @property
    def points(self):
        return self.centers

    def render_pv(self, pl, cmap, clim):
        pv = _pv()
        grid = pv.PolyData(self.centers)
        if self.scalar is not None:
            grid["scalars"] = self.scalar
        factor = float(np.min(self.spacing))
        cubes = grid.glyph(geom=pv.Cube(), scale=False, factor=factor, orient=False)
        pl.add_mesh(
            cubes,
            scalars="scalars" if self.scalar is not None else None,
            cmap=cmap,
            clim=clim,
            show_edges=True,
            line_width=1,
            show_scalar_bar=False,
        )


class Gaussians(_Panel):
    """A Gaussian mixture drawn as ellipsoids (pyvista) / means (matplotlib)."""

    def __init__(
        self, means, covariances, scalar=None, weights=None, size=25.0, title=None, res=None
    ):
        super().__init__(scalar, size, title)
        self.means = np.asarray(means)
        self.covariances = np.asarray(covariances)
        self.weights = None if weights is None else np.asarray(weights)
        self.res = res

    @property
    def points(self):
        return self.means

    def render_pv(self, pl, cmap, clim):
        pv = _pv()
        n = self.means.shape[0]
        eig = np.linalg.eigh(self.covariances)
        # order axes largest-first and fix reflections so orientations are proper
        evecs_rot = eig.eigenvectors[..., ::-1].copy()
        evecs_rot[np.linalg.det(evecs_rot) < 0, :, 0] *= -1
        axes = np.sqrt(3 * eig.eigenvalues)  # (n, 3), covers ~1 std at 3-sigma scale
        res = self.res if self.res is not None else (90 if n <= 200 else 30)
        scalar = self.scalar if self.scalar is not None else np.zeros(n)

        ellipsoids = []
        for i in range(n):
            ell = pv.ParametricEllipsoid(*axes[i, ::-1], u_res=res, v_res=res)
            ell.points = ell.points @ evecs_rot[i].T + self.means[i]
            ell["scalars"] = scalar[i] * np.ones(ell.n_points)
            ellipsoids.append(ell)

        merged = pv.merge(ellipsoids)
        merged.flip_faces(inplace=True)  # pyvista ellipsoid normals point inward
        pl.add_mesh(
            merged,
            scalars="scalars",
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
        )


def plot_panels(panels, backend=None, cmap="coolwarm", clim=None, figsize=None):
    """Draw a row of scalar-coloured geometry panels in the chosen backend.

    Parameters
    ------------------------------
    panels  : list of ``Points`` / ``Mesh`` / ``Voxels`` / ``Gaussians``
    backend : "matplotlib" | "pyvista" | None (use the module default)
    cmap    : colormap name applied to every panel
    clim    : (vmin, vmax) shared colour limits, or None to autoscale per panel
    figsize : matplotlib figure size (matplotlib backend only)

    Output
    ------------------------------
    matplotlib Figure or pyvista Plotter
    """
    backend = backend or _BACKEND
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
    if backend == "matplotlib":
        return _plot_panels_mpl(panels, cmap, clim, figsize)
    return _plot_panels_pv(panels, cmap, clim)


def _plot_panels_mpl(panels, cmap, clim, figsize):
    import matplotlib.pyplot as plt

    n = len(panels)
    if figsize is None:
        figsize = (3.6 * n, 4)
    vmin, vmax = (None, None) if clim is None else clim

    fig, axes = plt.subplots(1, n, figsize=figsize, subplot_kw={"projection": "3d"})
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        pts = np.asarray(panel.points)
        ax.scatter(*pts.T, c=panel.scalar, s=panel.size, cmap=cmap, vmin=vmin, vmax=vmax)
        if panel.title:
            ax.set_title(panel.title)
        ax.set_axis_off()
        ax.view_init(elev=100, azim=-90)
    fig.tight_layout()
    return fig


def _plot_panels_pv(panels, cmap, clim):
    pv = _pv()
    n = len(panels)
    pl = pv.Plotter(shape=(1, n))
    for i, panel in enumerate(panels):
        pl.subplot(0, i)
        panel.render_pv(pl, cmap, clim)
        if panel.title:
            pl.add_title(panel.title, font_size=6)
    if n > 1:
        pl.link_views()
    pl.show()
    return pl
