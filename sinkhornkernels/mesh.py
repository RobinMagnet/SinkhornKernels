"""Mesh utilities: vertex masses and reference FEM Laplacians.

These provide the classical baselines the normalized kernels are compared
against in the paper: the cotangent Laplacian on triangle meshes, its
tetrahedral (dihedral-cotangent) counterpart on volume meshes, and the
consistent / lumped P1 mass matrices. All functions take plain
``(vertices, faces)`` index arrays; everything is vectorized numpy/scipy.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh


def load_obj(path):
    """Minimal Wavefront OBJ loader (triangle meshes, no materials).

    Returns
    -------
    vertices : (N, 3) float ndarray
    faces : (m, 3) int ndarray (0-indexed)
    """
    vertices, faces = [], []
    with open(path) as fp:
        for line in fp:
            if line.startswith("v "):
                vertices.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def sample_surface(vertices, faces, n_samples, rng=None):
    """Uniform-area random sampling of points on a triangle mesh surface.

    Returns
    -------
    points : (n_samples, 3) ndarray
    """
    rng = np.random.default_rng(rng)
    # Sample triangle wrt their area
    areas = face_areas(vertices, faces)  # (m,)
    chosen = rng.choice(len(faces), size=n_samples, p=areas / areas.sum())  # (n_samples,)

    # Sample random barycentric coordinates in each triangle
    u, v = rng.random((2, n_samples))  # each (n_samples,) barycentric coords
    flip = u + v > 1  # (n_samples,) reflect points that fell outside the triangle
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    tri = np.asarray(vertices)[np.asarray(faces)[chosen]]  # (n_samples, 3, 3)
    return (1 - u - v)[:, None] * tri[:, 0] + u[:, None] * tri[:, 1] + v[:, None] * tri[:, 2]


def face_areas(vertices, faces):
    """Areas of the triangles of a mesh. ``faces``: (m, 3) int array."""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    v1, v2, v3 = (vertices[faces[:, i]] for i in range(3))  # each (m, 3)
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)


def vertex_areas(vertices, faces, faces_areas=None):
    """Barycentric lumped vertex areas (one third of each incident triangle)."""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if faces_areas is None:
        faces_areas = face_areas(vertices, faces)  # (m,)
    areas = np.zeros(vertices.shape[0])  # (N,)
    np.add.at(areas, faces.ravel(), np.repeat(faces_areas / 3.0, 3))
    return areas


def tet_volumes(vertices, tets):
    """Volumes of the tetrahedra of a mesh. ``tets``: (m, 4) int array."""
    vertices, tets = np.asarray(vertices), np.asarray(tets)
    v1, v2, v3, v4 = (vertices[tets[:, i]] for i in range(4))  # each (m, 3)
    return np.abs(np.einsum("md,md->m", np.cross(v2 - v1, v3 - v1), v4 - v1)) / 6.0  # (m,)


def vertex_volumes(vertices, tets, tets_volumes=None):
    """Barycentric lumped vertex volumes (one fourth of each incident tet)."""
    vertices, tets = np.asarray(vertices), np.asarray(tets)
    if tets_volumes is None:
        tets_volumes = tet_volumes(vertices, tets)  # (m,)
    volumes = np.zeros(vertices.shape[0])  # (N,)
    np.add.at(volumes, tets.ravel(), np.repeat(tets_volumes / 4.0, 4))
    return volumes


def _cotangent(u, v):
    """Row-wise cot of the angle between u and v; 0 for degenerate pairs.

    ``u``, ``v`` are (m, 3) arrays; returns (m,).
    """
    cos = np.einsum("md,md->m", u, v)  # (m,)
    sin = np.linalg.norm(np.cross(u, v), axis=1)  # (m,)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sin > 0, cos / sin, 0.0)


def cotangent_laplacian(vertices, faces):
    """Cotangent stiffness matrix of a triangle mesh (PSD, $L 1 = 0$).

    Each triangle contributes $\\cot(\\theta)/2$ to the edge opposite
    the angle $\\theta$. Degenerate triangles contribute 0.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    N = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    # Edge lengths indexed by opposite vertex
    u1 = v3 - v2
    u2 = v1 - v3
    u3 = v2 - v1

    L1 = np.linalg.norm(u1, axis=1)  # (m,)
    L2 = np.linalg.norm(u2, axis=1)  # (m,)
    L3 = np.linalg.norm(u3, axis=1)  # (m,)

    # Compute cosine of angles
    A1 = np.einsum("ij,ij->i", -u2, u3) / (L2 * L3)  # (m,)
    A2 = np.einsum("ij,ij->i", u1, -u3) / (L1 * L3)  # (m,)
    A3 = np.einsum("ij,ij->i", -u1, u2) / (L1 * L2)  # (m,)

    # Use cot(arccos(x)) = x/sqrt(1-x^2)
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    S = np.concatenate([A3, A1, A2])
    S = 0.5 * S / np.sqrt(1 - S**2)

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S, -S, S, S])

    W = sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
    return W


def _dihedral_cot(a, b, c, d):
    """Cot of the dihedral angle along edge (a, b) between faces (a,b,c), (a,b,d).

    All arguments are (m, 3) arrays of points; returns (m,).
    """
    n1 = np.cross(b - a, c - a)  # (m, 3) face normal
    n2 = np.cross(b - a, d - a)  # (m, 3) face normal
    cos = np.einsum("md,md->m", n1, n2)  # (m,)
    sin = np.linalg.norm(np.cross(n1, n2), axis=1)  # (m,)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sin > 0, cos / sin, 0.0)


def tet_laplacian(vertices, tets):
    """Tetrahedral cotangent Laplacian (PSD, $L 1 = 0$).

    In each tet, the edge (a, b) is weighted by
    $\\ell_{cd} \\cot(\\theta_{cd}) / 6$ where $(c, d)$ is the
    opposite edge and $\\theta_{cd}$ its dihedral angle.
    """
    vertices, tets = np.asarray(vertices), np.asarray(tets)
    N = vertices.shape[0]
    vi, vj, vk, vl = (vertices[tets[:, i]] for i in range(4))  # each (m, 3) tet corners
    ti, tj, tk, tl = (tets[:, i] for i in range(4))  # each (m,) corner vertex indices

    def w(a_pts, b_pts, c_pts, d_pts):
        # weight of edge (c, d): length of (a, b) times its dihedral cot -> (m,)
        return _dihedral_cot(a_pts, b_pts, c_pts, d_pts) * np.linalg.norm(a_pts - b_pts, axis=1)

    # 6 edges per tet, each contributing a (m,) weight
    edges = [
        (ti, tj, w(vk, vl, vi, vj)),
        (ti, tk, w(vj, vl, vi, vk)),
        (ti, tl, w(vj, vk, vi, vl)),
        (tj, tk, w(vi, vl, vj, vk)),
        (tj, tl, w(vi, vk, vj, vl)),
        (tk, tl, w(vi, vj, vk, vl)),
    ]
    I = np.concatenate([e[0] for e in edges])  # (6m,)
    J = np.concatenate([e[1] for e in edges])  # (6m,)
    V = np.concatenate([e[2] for e in edges])  # (6m,)

    W = sparse.coo_matrix(  # (N, N) symmetric edge-weight matrix
        (np.concatenate([V, V]), (np.concatenate([I, J]), np.concatenate([J, I]))),
        shape=(N, N),
    ).tocsr()
    return ((sparse.diags(np.asarray(W.sum(axis=1)).ravel()) - W) / 6.0).tocsr()  # (N, N)


def fem_mass_matrix(vertices, faces, faces_areas=None):
    """Consistent P1 mass matrix of a triangle mesh (A/6 diagonal, A/12 off-diagonal)."""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    N = vertices.shape[0]
    if faces_areas is None:
        faces_areas = face_areas(vertices, faces)  # (m,)

    # per-triangle edge endpoints and areas, all (3m,)
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    S = np.tile(faces_areas, 3)

    In = np.concatenate([I, J, I])  # (9m,)
    Jn = np.concatenate([J, I, I])  # (9m,)
    Sn = np.concatenate([S, S, 2 * S]) / 12.0  # (9m,) off/off/diagonal entries
    return sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsr()  # (N, N)


def fem_mass_matrix_tet(vertices, tets, tets_volumes=None):
    """Consistent P1 mass matrix of a tet mesh (V/10 diagonal, V/20 off-diagonal)."""
    vertices, tets = np.asarray(vertices), np.asarray(tets)
    N = vertices.shape[0]
    if tets_volumes is None:
        tets_volumes = tet_volumes(vertices, tets)  # (m,)

    ti, tj, tk, tl = (tets[:, i] for i in range(4))  # each (m,) corner indices
    pairs = [
        (ti, tj),
        (ti, tk),
        (ti, tl),
        (tj, tk),
        (tj, tl),
        (tk, tl),
    ]  # 6 off-diagonal edges
    # 4 diagonal (V/10) + 12 off-diagonal (V/20) entries per tet, all (16m,)
    In = np.concatenate([ti, tj, tk, tl] + [p[0] for p in pairs] + [p[1] for p in pairs])
    Jn = np.concatenate([ti, tj, tk, tl] + [p[1] for p in pairs] + [p[0] for p in pairs])
    Sn = np.concatenate([np.tile(tets_volumes / 10.0, 4), np.tile(tets_volumes / 20.0, 12)])
    return sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsr()  # (N, N)


def fem_spectrum(L, M, k=64, shift=-0.01):
    """Smallest generalized eigenpairs $L \\phi = \\lambda M \\phi$.

    Shift-invert ``eigsh`` around ``shift`` (slightly negative, so the
    constant eigenvector at $\\lambda = 0$ is retrieved robustly).

    Returns
    -------
    evals : (k,) ndarray, sorted increasingly (first one ~0).
    evecs : (N, k) ndarray, M-orthonormal.
    """
    evals, evecs = eigsh(L.tocsc(), k=k, M=M.tocsc(), sigma=shift)  # (k,), (N, k)
    order = np.argsort(evals)  # (k,) sort ascending
    return evals[order], evecs[:, order]
