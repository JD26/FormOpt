import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, Normalize

import pathlib as Path

# import matplotlib
# matplotlib.rcParams.update({
# 	"text.usetex": False,
# 	"font.family": "serif",
# 	"font.serif": ["STIX"],
# 	"font.size": 14
# })

import numpy as np
import pyvista as pvt
import h5py

from dolfinx.plot import vtk_mesh
from dolfinx.fem import functionspace, Function
from dolfinx.io import XDMFFile

from mpi4py import MPI

rank = MPI.COMM_WORLD.rank


def select(test_path, niter, limits):

    points, cells, phi = None, None, None

    with h5py.File(test_path / "phi_functions.h5", "r") as f:
        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]
        phi_group = f["/Function/phi"]
        phi = phi_group[str(0)][:, 0]
        u0_group = f["/Function/u" + str(niter)]
        u0 = u0_group[str(0)][:, [0, 1]]
        points = points + u0

    x_coords, y_coords = points[:, 0], points[:, 1]
    triang = mtri.Triangulation(x_coords, y_coords, cells)
    fig, ax = plt.subplots()
    ax.tricontourf(
        triang,
        phi,
        levels=[min(phi), 0.0, max(phi)],
        colors=["black", (1, 1, 1)],
    )

    ax.tricontour(triang, phi, levels=[0], colors="k", linewidths=1)
    ax.scatter(x_coords, y_coords, s=5, color="blue")
    ax.set_aspect("equal")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    (selected_point,) = ax.plot([], [], "ro", markersize=8)

    def onclick(event):
        if event.inaxes != ax:
            return

        # Distancia al punto clickeado
        dist = (x_coords - event.xdata) ** 2 + (y_coords - event.ydata) ** 2
        idx = np.argmin(dist)

        # Resaltar nodo
        selected_point.set_data([x_coords[idx]], [y_coords[idx]])
        fig.canvas.draw_idle()

        print(f": {idx}")
        # print(f"Coordenadas: ({x_coords[idx]}, {y_coords[idx]})")
        # print("-" * 40)

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()


def plot_lv_with_control_pts(
    test_path, final_iter, limits, idxs, title=None, figsize=None
):

    points, cells, phi = None, None, None

    with h5py.File(test_path / "phi_functions.h5", "r") as f:
        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]
        phi_group = f["/Function/phi"]
        phi = phi_group[str(0)][:, 0]

    x_coords, y_coords = points[:, 0], points[:, 1]
    triang = mtri.Triangulation(x_coords, y_coords, cells)
    fig, ax = plt.subplots(figsize=figsize)
    ax.tricontourf(
        triang,
        phi,
        levels=[min(phi), 0.0, max(phi)],
        colors=["black", (1, 1, 1)],
    )

    ax.tricontour(triang, phi, levels=[0], colors="k", linewidths=1)
    ax.scatter(x_coords[idxs], y_coords[idxs], s=5, color="yellow")
    for i, idx in enumerate(idxs):
        ax.annotate(
            str(i),  # texto que quieres mostrar
            (x_coords[idx], y_coords[idx]),  # posición del punto
            textcoords="offset points",  # desplaza el texto respecto al punto
            xytext=(4, 4),  # desplazamiento en puntos (x,y)
            ha="left",  # alineación horizontal
            va="bottom",  # alineación vertical
            fontsize=10,
            bbox=dict(boxstyle="round", fc="0.8"),
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ls = []
    with h5py.File(test_path / "phi_functions.h5", "r") as f:
        for i in range(1, final_iter + 1):
            u_group = f["/Function/u" + f"{i:03}"]
            u = u_group[str(0)][:, [0, 1]]
            ls.append(np.linalg.norm(u[idxs], axis=1))

    return np.array(ls).T


def plot_deformations(
    test_path,
    name,
    limits,
    title=None,
    figsize=None,
    boundaries=None,
    lw=2,
    filename=None,
):

    points, cells, phi = None, None, None

    with h5py.File(test_path / "deformations.h5", "r") as f:
        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]
        phi_group = f["/Function/phi"]
        phi = phi_group[str(0)][:, 0]
        u0_group = f["/Function/" + name]
        u0 = u0_group[str(0)][:, [0, 1]]
        points = points + u0

    x_coords, y_coords = points[:, 0], points[:, 1]
    triang = mtri.Triangulation(x_coords, y_coords, cells)
    fig, ax = plt.subplots(figsize=figsize)
    ax.tricontourf(
        triang,
        phi,
        levels=[min(phi), 1e-4, max(phi)],
        colors=["gray", (1, 1, 1)],
    )

    ax.tricontour(triang, phi, levels=[0], colors="k", linewidths=1)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if boundaries:
        for xy, cl in boundaries:
            ax.plot(xy[:, 0], xy[:, 1], color=cl, linewidth=lw)

    if filename:
        plt.savefig(filename, dpi=300, pad_inches=0, bbox_inches="tight")


def multiply_lv_by_u(test_path, niter):
    with h5py.File(test_path / "results.h5", "r+") as f:
        phi = f["/Function/phi"][str(niter)][:, 0]
        u0_dset = f["/Function/u0"][str(niter)]
        u0 = u0_dset[:, 0]
        # multiply
        u0_new = phi * u0
        # write back
        u0_dset[:, 0] = u0_new


def plot_u0(
    test_path,
    niter,
    limits,
    figsize=None,
    boundaries=None,
    lw=2,
    filename=None,
    title=None,
):

    points, cells, u0 = None, None, None

    with h5py.File(test_path / "results.h5", "r") as f:
        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]
        u0_group = f["/Function/u0"]
        u0 = u0_group[str(niter)][:, 0]

    x_coords, y_coords = points[:, 0], points[:, 1]
    triang = mtri.Triangulation(x_coords, y_coords, cells)
    fig, ax = plt.subplots(figsize=figsize)

    # ax.tricontourf(triang, u0, levels=np.linspace(u0.min(), u0.max(), 20))

    cs = ax.tricontour(
        triang, u0, levels=np.linspace(u0.min(), u0.max(), 15), colors="black"
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    ax.set_aspect("equal")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        plt.title(title)

    if filename:
        plt.savefig(filename, dpi=300, pad_inches=0, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_lv(
    test_path,
    niter,
    limits,
    displacement=True,
    figsize=None,
    boundaries=None,
    lw=2,
    filename=None,
    title=None,
):

    points, cells, phi = None, None, None

    with h5py.File(test_path / "results.h5", "r") as f:
        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]
        phi_group = f["/Function/phi"]
        phi = phi_group[str(niter)][:, 0]
        if displacement:
            u0_group = f["/Function/u0"]
            u0 = u0_group[str(niter)][:, [0, 1]]
            points = points + u0

    x_coords, y_coords = points[:, 0], points[:, 1]
    triang = mtri.Triangulation(x_coords, y_coords, cells)
    fig, ax = plt.subplots(figsize=figsize)
    ax.tricontourf(
        triang,
        phi,
        levels=[min(phi), 1e-4, max(phi)],
        colors=["gray", (1, 1, 1)],
    )

    ax.tricontour(triang, phi, levels=[0], colors="black", linewidths=1)

    ax.set_aspect("equal")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        plt.title(title)

    if boundaries:
        for xy, cl in boundaries:
            ax.plot(xy[:, 0], xy[:, 1], color=cl, linewidth=lw)

    if filename:
        plt.savefig(filename, dpi=300, pad_inches=0, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_vm(
    test_path,
    niter,
    limits,
    vmax,
    displacement=True,
    figsize=None,
    boundaries=None,
    lw=2,
    filename=None,
    title="Von Mises stress",
):
    """
    Von Misses stress
    """

    points, cells, vm = None, None, None

    with h5py.File(test_path / "results.h5", "r") as f:
        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]
        phi_group = f["/Function/phi"]
        phi = phi_group[str(niter)][:, 0]
        vm_group = f["/Function/VonMises"]
        vm = vm_group[str(niter)][:, 0]
        if displacement:
            u0_group = f["/Function/u0"]
            u0 = u0_group[str(niter)][:, [0, 1]]
            points = points + u0

    print("> Von Mises stress - maximum value = ", np.max(vm))
    x_coords, y_coords = points[:, 0], points[:, 1]
    triang = mtri.Triangulation(x_coords, y_coords, cells)
    fig, ax = plt.subplots(figsize=figsize)
    if title != "":
        ax.set_title(title)
    cf = ax.tricontourf(
        triang,
        vm,
        cmap="turbo",
        levels=np.linspace(0, vmax, 14),
        vmin=0,
        vmax=vmax,
    )

    ax.tricontour(triang, phi, levels=[0], colors="k", linewidths=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(cf, cax=cax)

    ax.set_aspect("equal")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if boundaries:
        for xy, cl in boundaries:
            ax.plot(xy[:, 0], xy[:, 1], color=cl, linewidth=lw)

    if filename:
        plt.savefig(filename, dpi=300, pad_inches=0, bbox_inches="tight")


def plot_dp(
    test_path,
    niter,
    limits,
    vmax,
    displacement=True,
    figsize=None,
    boundaries=None,
    lw=2,
    filename=None,
    title="Displacement (in norm)",
):
    """
    Displacement
    """

    points, cells, u0 = None, None, None

    with h5py.File(test_path / "results.h5", "r") as f:
        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]
        phi_group = f["/Function/phi"]
        phi = phi_group[str(niter)][:, 0]
        dp_group = f["/Function/Displacement"]
        dp = dp_group[str(niter)][:, 0]
        if displacement:
            u0_group = f["/Function/u0"]
            u0 = u0_group[str(niter)][:, [0, 1]]
            points = points + u0

    print("> Displacement (in norm) - maximum value = ", np.max(dp))
    x_coords, y_coords = points[:, 0], points[:, 1]
    triang = mtri.Triangulation(x_coords, y_coords, cells)
    fig, ax = plt.subplots(figsize=figsize)
    if title != "":
        ax.set_title(title)
    cf = ax.tricontourf(
        triang,
        dp,
        cmap="turbo",
        levels=np.linspace(0, vmax, 14),
        vmin=0,
        vmax=vmax,
    )

    ax.tricontour(triang, phi, levels=[0], colors="k", linewidths=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(cf, cax=cax)

    ax.set_aspect("equal")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if boundaries:
        for xy, cl in boundaries:
            ax.plot(xy[:, 0], xy[:, 1], color=cl, linewidth=lw)

    if filename:
        plt.savefig(filename, dpi=300, pad_inches=0, bbox_inches="tight")


def plot_bars(num_procs, times, filename, ylimits=None):
    #  "DarkGray" or "CornflowerBlue"
    color_bar = "CornflowerBlue"

    fig, ax = plt.subplots()

    ax.bar(num_procs, times, color=color_bar, edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Number of processes")
    ax.set_ylabel("Execution time (seconds)")
    # ax.set_title("Execution time vs. number of processors", fontsize=13, pad=10)
    ax.set_xticks(num_procs)
    # Labels over each bar (optional)
    # for i, t in enumerate(times):
    #     ax.text(
    #         num_procs[i], t + 0.1, f"{t:.1f}", ha="center", va="bottom", fontsize=10
    #     )

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    if ylimits:
        ax.set_ylim(ylimits)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def plot_tutorial(h5file, size, limits=None, boundaries=None, lw=2):

    with h5py.File(h5file, "r") as f:

        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]

        phi_group = f["/Function/phi"]
        niter = len(list(phi_group.keys())) - 1

        plot_level(
            2,
            points,
            cells,
            phi_group["0"][:, 0],
            figsize=size,
            lims=limits,
            filename=None,
            boundaries=boundaries,
            lw=lw,
        )

        plot_level(
            2,
            points,
            cells,
            phi_group[str(niter)][:, 0],
            figsize=size,
            lims=limits,
            filename=None,
            boundaries=boundaries,
            lw=lw,
        )


def InitialLevel(centers, radii, ftr=1.0):
    dim = centers.shape[1]

    def func(x):

        xT = (x[:dim].T)[None, :, :]
        comps = (centers[:, None, :] - xT) ** 2
        norms = np.sqrt(np.sum(comps, axis=2))
        dists = radii[:, None] - norms

        return ftr * np.max(dists, axis=0)

    return func


def see_initial_guess(
    initial_guess, filename, boundaries=None, save_filename=None, figsize=None
):
    """
    Read a domain and plot the initial guess.
    Only for triangular meshes.
    Usage:
        sp.see_initial_guess(
            (centers, radii),
            Path(test_path)/"domain.xdmf",
            boundaries
        )
    """
    if rank == 0:
        with XDMFFile(MPI.COMM_SELF, filename, "r") as xdmf:
            domain = xdmf.read_mesh(name="mesh")
        initial_level = Function(functionspace(domain, ("CG", 1)))
        initial_level.interpolate(InitialLevel(*initial_guess))
        dim = domain.topology.dim
        cells = domain.topology.connectivity(dim, 0).array.reshape((-1, dim + 1))
        plot_level(
            dim,
            domain.geometry.x,
            cells,
            initial_level.x.array,
            boundaries,
            figsize,
            "Initial guess",
            save_filename,
        )


def see_domain_T(filename, boundaries=None, figsize=None):
    """
    Read a mesh of Triangular elements (2D) or
    Tetrahedral elements (3D), and plot it with
    matplotlib (2D) or pyvista (3D). Coordinates
    (2D) or marker functions (3D) are allowed for
    coloring parts of the boundary.

    Usage:
        Collect the boundary points
        vertices = vertices + [vertices[0]]
        dir = np.array(
            vertices[(dir_index[0]-1):(dir_index[-1]+1)]
        )
        neu = np.array(
            vertices[(neu_index[0]-1):(neu_index[-1]+1)]
        )
        boundaries = [(dir, "red"), (neu, "blue")]
        sp.see_domain_T(
            Path(test_path)/"domain.xdmf", boundaries
        )
    or
        boundaries = [
            (boundary_dirichlet, [255, 0, 0]),
            (boundary_neumann, [0, 255, 0])
        ]
        sp.see_domain_T(
            Path(test_path)/"domain.xdmf", boundaries
        )
    """
    if rank == 0:
        with XDMFFile(MPI.COMM_SELF, filename, "r") as xdmf:
            domain = xdmf.read_mesh(name="mesh")
        dim = domain.topology.dim
        cells = domain.topology.connectivity(dim, 0).array.reshape((-1, dim + 1))
        plot_domain_T(
            dim,
            domain.topology,
            domain.geometry.x,
            cells,
            boundaries,
            figsize,
            "Domain",
        )


def see_domain(filename):
    """
    Read a mesh and plot it with pyvista.
    """
    if rank == 0:
        with XDMFFile(MPI.COMM_SELF, filename, "r") as xdmf:
            domain = xdmf.read_mesh(name="mesh")
        plot_domain(domain, "Domain")


def plot_basic(test_path, boundaries=None):
    """
    This function plots:
    (1) the level set function at several iterations;
    (2) the values of the cost and Lagrangian functionals,
        the constraint errors, and the norm of the derivative.

    Note: (1) works for triangular meshes (2D case).
    """

    plot_results(
        test_path / "results.h5", lambda i: test_path / f"iter{i:05d}.png", boundaries
    )

    data = np.load(test_path / "data.npz")

    print("> Data fields:", data.files)

    plot_cost(data["cost"], filename=test_path / "cost.png")

    plot_derivative(data["nder"], filename=test_path / "derivative.png")

    if "Lg" in data:
        plot_lagrangian(data["Lg"], filename=test_path / "lagrangian.png")

    if "ctrs" in data:
        plot_constraint(
            np.linalg.norm(data["ctrs"] - 1.0, axis=1),
            filename=test_path / "constr_error.png",
        )

    print("> See", test_path)


def see_magnitude(
    domain, state, rank_dimension, boundaries=None, figsize=None, save_path=None
):
    """
    Plot the magnitude of a vector-valued function.
    """
    dim = domain.topology.dim
    plot_magnitude(
        dim,
        domain.geometry.x,
        domain.topology.connectivity(dim, 0).array.reshape((-1, dim + 1)),
        state.x.array,
        rank_dimension,
        boundaries=boundaries,
        figsize=figsize,
        save_path=save_path,
    )


def see_2D_spacial(domain, u):
    """
    Plot a scalar-valued function
    on a 2-dimensional domain.
    Only for CG-1 functions.
    """
    plot_2D_spacial(
        domain.geometry.x[:, 0],
        domain.geometry.x[:, 1],
        domain.topology.connectivity(2, 0).array.reshape((-1, 3)),
        u.x.array,
    )


def plot_domain(domain, title=None):
    plotter = pvt.Plotter()
    domain_parts = vtk_mesh(functionspace(domain, ("CG", 1)))
    grid = pvt.UnstructuredGrid(*domain_parts)
    plotter.add_mesh(grid, color="gray", show_edges=True)
    if title:
        plotter.add_title(title)
    plotter.show()


def plot_triangulation_and_level0(h5file, niter, lims, lw=0.1, filename=None):

    with h5py.File(h5file / "results.h5", "r") as f:

        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]
        phi_group = f["/Function/phi"]
        values = phi_group[str(niter)][:, 0]

        x_coords, y_coords = points[:, 0], points[:, 1]
        triang = mtri.Triangulation(x_coords, y_coords, cells)
        fig, ax = plt.subplots()
        ax.tricontour(
            triang,
            values,
            levels=[min(values), 0.0, max(values)],
            linestyles="-",
            linewidths=lw,
        )
        ax.triplot(triang, "b-", linewidth=0.1)
        ax.set_aspect("equal")

        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, pad_inches=0, bbox_inches="tight")


def plot_results_for_doc(h5file, niter, lims, lw, path_name, boundaries=None):

    with h5py.File(h5file / "results.h5", "r") as f:

        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]

        phi_group = f["/Function/phi"]

        plot_level_for_doc(
            points,
            cells,
            phi_group[str(niter)][:, 0],
            lims=lims,
            lw=lw,
            filename=path_name,
            boundaries=boundaries,
        )


def plot_results(h5file, p, boundaries=None):

    with h5py.File(h5file, "r") as f:

        points = f["/Mesh/mesh/geometry"][:]
        cells = f["/Mesh/mesh/topology"][:]

        phi_group = f["/Function/phi"]
        niter = len(list(phi_group.keys())) - 1
        plot_level(
            2, points, cells, phi_group["0"][:, 0], filename=p(0), boundaries=boundaries
        )
        for iter in range(1, niter):
            if iter % 10 == 0:
                plot_level(
                    2,
                    points,
                    cells,
                    phi_group[str(iter)][:, 0],
                    filename=p(iter),
                    boundaries=boundaries,
                )
        values = (phi_group[str(niter)][:, 0])[:]
        plot_level(
            2,
            points,
            cells,
            phi_group[str(niter)][:, 0],
            filename=p(niter),
            boundaries=boundaries,
        )

    x_coords, y_coords = points[:, 0], points[:, 1]
    plot_2D_spacial(x_coords, y_coords, cells, values)


def plot_2D_spacial(x_coords, y_coords, cells, values, figsize=None, save_path=None):

    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    z_range = values.max() - values.min()
    eps = min(5.0 * x_range / 100.0, 5.0 * y_range / 100.0, 5.0 * z_range / 100.0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_trisurf(x_coords, y_coords, values, triangles=cells, cmap="bone")
    surf.set_facecolor((0, 0, 1, 0.2))

    triang = mtri.Triangulation(x_coords, y_coords, cells)

    ax.tricontour(triang, values, levels=[0], colors="red", linewidths=2)

    ax.set_aspect("equal")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, format=save_path.split(".")[-1], dpi=300)
    else:
        plt.show()


def plot_level_for_doc(
    points, cells, values, lims, lw, boundaries=None, figsize=None, filename=None
):

    x_coords, y_coords = points[:, 0], points[:, 1]

    triang = mtri.Triangulation(x_coords, y_coords, cells)

    fig, ax = plt.subplots(figsize=figsize)
    ax.tricontourf(
        triang,
        values,
        levels=[min(values), 0.0, max(values)],
        colors=["black", (0.9, 0.9, 0.9)],
    )
    ax.set_aspect("equal")

    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ax.set_position([0, 0, 1, 1])

    fig.tight_layout()

    if boundaries:
        for xy, cl in boundaries:
            ax.plot(xy[:, 0], xy[:, 1], color=cl, linewidth=lw)

    if filename:
        # format = filename.suffix[1:]
        plt.savefig(filename, dpi=300, pad_inches=0, bbox_inches="tight")


def plot_level(
    dim,
    points,
    cells,
    values,
    lims=None,
    boundaries=None,
    lw=2,
    figsize=None,
    title=None,
    filename=None,
):
    if dim == 2:
        x_coords, y_coords = points[:, 0], points[:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        eps = min(5.0 * x_range / 100.0, 5.0 * y_range / 100.0)

        triang = mtri.Triangulation(x_coords, y_coords, cells)

        fig, ax = plt.subplots(figsize=figsize)
        ax.tricontourf(
            triang,
            values,
            levels=[min(values), 0.0, max(values)],
            colors=["black", (0.9, 0.9, 0.9)],
        )
        ax.set_aspect("equal")
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            # ax.set_position([0, 0, 1, 1])
        else:
            ax.set_xlim(x_coords.min() - eps, x_coords.max() + eps)
            ax.set_ylim(y_coords.min() - eps, y_coords.max() + eps)

        fig.tight_layout()

        if title:
            ax.set_title(title)

        if boundaries:
            for xy, cl in boundaries:
                ax.plot(xy[:, 0], xy[:, 1], color=cl, linewidth=lw)

        if filename:
            # format = filename.suffix[1:]
            plt.savefig(filename, dpi=300, pad_inches=0, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    if dim == 3:
        N = cells.shape[0]
        grid = pvt.UnstructuredGrid(
            np.column_stack((np.full(N, 4), cells)).flatten(),
            np.full(N, pvt.CellType.TETRA),
            points,
        )
        cell_connectivity = grid.cells.reshape(-1, 5)[:, 1:]
        neg_elements = [
            i for i, cell in enumerate(cell_connectivity) if np.all(values[cell] < 0)
        ]
        neg_cells = grid.extract_cells(neg_elements)
        if filename:
            plotter = pvt.Plotter(off_screen=True)
            plotter.add_mesh(neg_cells, color="gray", show_edges=False, opacity=0.5)
            plotter.show_axes()
            plotter.screenshot(filename)
        else:
            plotter = pvt.Plotter()
            plotter.add_mesh(neg_cells, color="gray", show_edges=False, opacity=1.0)
            plotter.show_axes()
            plotter.show()


def plot_domain_T(
    dim,
    topology,
    points,
    cells,
    boundaries=None,
    figsize=None,
    title=None,
    filename=None,
):

    if dim == 2:
        x_coords, y_coords = points[:, 0], points[:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        eps = min(5.0 * x_range / 100.0, 5.0 * y_range / 100.0)

        fig, ax = plt.subplots(figsize=figsize)

        ax.triplot(x_coords, y_coords, cells, color="lightgray", linewidth=0.5)
        if title:
            ax.set_title(title)

        if boundaries:
            for xy, cl in boundaries:
                ax.plot(xy[:, 0], xy[:, 1], color=cl)

        ax.set_aspect("equal")
        ax.set_xlim(x_coords.min() - eps, x_coords.max() + eps)
        ax.set_ylim(y_coords.min() - eps, y_coords.max() + eps)

        fig.tight_layout()

        if filename:
            plt.savefig(filename, format=filename.suffix[1:], dpi=300)
            plt.close()
        else:
            plt.show()

    if dim == 3:
        """
        # Old code for plot 3D domain
        N = cells.shape[0]
        grid = pvt.UnstructuredGrid(
            np.column_stack((np.full(N, 4), cells)).flatten(),
            np.full(N, pvt.CellType.TETRA),
            points
        )
        #grid.cell_data["Boundary Tags"] = boundaries
        plotter = pvt.Plotter()
        plotter.add_mesh(grid, show_edges = True, opacity = 0.5)
        plotter.show_axes()
        """

        topology.create_connectivity(2, 3)
        face_cells = topology.connectivity(2, 3)
        topology.create_connectivity(2, 0)
        face_nodes = topology.connectivity(2, 0)

        # Extraer las caras que están en la superficie
        nodes_in_surface = [
            face_nodes.links(face_id)
            for face_id in range(face_nodes.num_nodes)
            if len(face_cells.links(face_id)) == 1
        ]

        surface_faces = np.array(
            [[3, *nodes] for nodes in nodes_in_surface], dtype=np.int32
        ).flatten()

        N = len(boundaries)
        booundary_without_condition = [128, 128, 128]
        points_T = points.T
        conditions = np.array(
            [np.all(f(points_T)[nodes_in_surface], axis=1) for f, _ in boundaries]
        )
        scalar_values = np.argmax(conditions, axis=0)
        scalar_values[np.all(conditions == 0, axis=0)] = N

        color_array = np.array(
            [color for _, color in boundaries] + [booundary_without_condition],
            dtype=np.uint8,
        )
        rgb_colors = np.take(color_array, scalar_values, axis=0)

        surface_mesh = pvt.PolyData(points, surface_faces)
        surface_mesh["colors"] = rgb_colors

        plotter = pvt.Plotter()
        plotter.add_mesh(surface_mesh, scalars="colors", show_edges=True, rgb=True)

        if filename:
            plotter.screenshot(filename)
        else:
            plotter.show()


def plot_lagrangian(lag_values, figsize=None, filename=None):
    """
    Plots the cost function values for each iteration.

    Parameters:
        lag_values (list or numpy array): Array of positive real values representing the cost function J.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 300.
        save_path (str, optional): If provided, saves the plot to the specified path. Default is None.
    """
    iterations = np.arange(0, len(lag_values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, lag_values, color="black", linewidth=2)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Lagrangian")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    if filename:
        plt.savefig(filename, format=filename.suffix[1:], dpi=300)
        plt.close()
    else:
        plt.show()


def plot_derivative(der_norm, figsize=None, filename=None):
    iterations = np.arange(0, len(der_norm))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, der_norm, color="black", linewidth=2)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Derivative norm")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    if filename:
        plt.savefig(filename, format=filename.suffix[1:], dpi=300)
        plt.close()
    else:
        plt.show()


def plot_cost(cost_values, figsize=None, filename=None):
    """
    Plots the cost function values for each iteration.

    Parameters:
        cost_values (list or numpy array): Array of positive real values representing the cost function J.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 300.
        save_path (str, optional): If provided, saves the plot to the specified path. Default is None.
    """
    iterations = np.arange(0, len(cost_values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, cost_values, color="black", linewidth=2)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    if filename:
        plt.savefig(filename, format=filename.suffix[1:], dpi=300)
        plt.close()
    else:
        plt.show()


def plot_volume(volume_values, figsize=None, filename=None):
    iterations = np.arange(0, len(volume_values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, volume_values, color="black", linewidth=2)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Volume")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    if filename:
        plt.savefig(filename, format=filename.suffix[1:], dpi=300)
        plt.close()
    else:
        plt.show()


def plot2(values, title="values", figsize=None, filename=None, hline=None):
    iterations = np.arange(0, len(values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, values, color="black", linewidth=2)

    ax.set_xlabel("Iterations")
    if title != "":
        ax.set_ylabel(title)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    if hline:
        ax.plot(
            iterations,
            np.full_like(iterations, hline),
            color="black",
            linestyle="--",
            linewidth=2,
        )

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_lag2(
    values, label, extra_values, extra_label, ylimits=None, figsize=None, filename=None
):
    """
    Plots the cost function values for each iteration and optionally another array.

    Parameters:
        values (list or numpy array): Array of positive real values representing the cost function J.
        extra_values (list or numpy array, optional): Another array to plot for comparison (e.g., validation cost).
        extra_label (str, optional): Label for the extra plot line. Default is "Extra".
        figsize (tuple, optional): Figure size passed to matplotlib.
        filename (Path or str, optional): If provided, saves the plot to the specified path. Default is None.
    """
    iterations = np.arange(len(values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, values, color="black", linewidth=2, label=label)

    extra_iterations = np.arange(len(extra_values))
    ax.plot(
        extra_iterations,
        extra_values,
        color="black",
        linestyle="--",
        linewidth=2,
        label=extra_label,
    )

    # ax.set_xlabel("Iterations")
    # ax.set_ylabel("Lagragangian values")
    if ylimits:
        ax.set_ylim(ylimits)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_constraint(cost_values, figsize=None, filename=None):
    """
    Plots the cost function values for each iteration.

    Parameters:
        cost_values (list or numpy array): Array of positive real values representing the cost function J.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 300.
        save_path (str, optional): If provided, saves the plot to the specified path. Default is None.
    """
    iterations = np.arange(0, len(cost_values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, cost_values, color="black", linewidth=2)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Constraint error")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    if filename:
        plt.savefig(filename, format=filename.suffix[1:], dpi=300)
        plt.close()
    else:
        plt.show()


"""
def plot_trisurf(x, y, cells, uxy):
    fig = ff.create_trisurf(
        x = x, y = y, z = uxy,
        simplices = cells,
        show_colorbar = True, plot_edges = True, colormap = 'Viridis')
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()
"""


def plot_velocity_2D(
    coords, cells, velocity_values, figsize=(6, 6), fontsize=12, filename=None, dpi=300
):
    x_coords, y_coords = coords
    u, v = velocity_values
    fig, ax = plt.subplots(figsize=figsize)
    ax.triplot(x_coords, y_coords, cells, color="lightgray", linewidth=0.5)
    ax.quiver(x_coords, y_coords, u, v, units="xy", color="darkblue", alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(x_coords.min() - 0.1, x_coords.max() + 0.1)
    ax.set_ylim(y_coords.min() - 0.1, y_coords.max() + 0.1)
    ax.set_xlabel("x", fontsize=fontsize)
    ax.set_ylabel("y", fontsize=fontsize)
    ax.set_title("Velocity Field $\theta$", fontsize=fontsize + 2)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_velocities_2D(
    data,
    indices,
    xlim,
    ylim,
    orientation="vertical",
    theme="scientific",
    fontsize=12,
    save_path=None,
    dpi=300,
):
    if theme == "scientific":
        plt.style.use("seaborn-v0_8-paper")

    # Set up the figure and axes
    num_plots = len(indices)
    xlen = xlim[1] - xlim[0]
    ylen = ylim[1] - ylim[0]
    if orientation == "vertical":
        fig, axes = plt.subplots(
            num_plots, 1, figsize=(5.0, (5.5 * ylen / xlen) * num_plots), sharex=True
        )
    else:
        fig, axes = plt.subplots(
            1, num_plots, figsize=((5.5 * xlen / ylen) * num_plots, 5), sharey=True
        )

    if num_plots == 1:  # Handle the case of a single subplot
        axes = [axes]

    for ax, i in zip(axes, indices):
        dti = data[i]
        x_coords, y_coords = dti["coords"]
        cells = dti["cells"]
        u, v = dti["velocity_vector"]

        ax.triplot(x_coords, y_coords, cells, color="lightgray", linewidth=0.5)
        ax.quiver(x_coords, y_coords, u, v, units="xy", color="darkblue", alpha=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xlim[0] - 0.1, xlim[1] + 0.1)
        ax.set_ylim(ylim[0] - 0.1, ylim[1] + 0.1)
        ax.set_title(f"i = {i}", fontsize=fontsize)

    plt.tight_layout()

    # Save or show the plots
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()


def plot_level_2D(
    coords, cells, cell_values, figsize=(6, 6), fontsize=12, filename=None, dpi=300
):

    x_coords, y_coords = coords[:, 0], coords[:, 1]

    # Define a binary colormap (white for 0, dark gray for 1)
    colors = [(1, 1, 1), (0.2, 0.2, 0.2)]  # RGB for white and dark gray
    cmap = ListedColormap(colors)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.tripcolor(
        x_coords,
        y_coords,
        cells,
        facecolors=[1 if v < 0 else 0 for v in cell_values],
        shading="flat",
        linewidth=0.5,
        cmap=cmap,
    )

    # Adjust plot appearance
    ax.set_aspect("equal")
    ax.set_xlim(x_coords.min() - 0.1, x_coords.max() + 0.1)
    ax.set_ylim(y_coords.min() - 0.1, y_coords.max() + 0.1)
    ax.set_xlabel("x", fontsize=fontsize)
    ax.set_ylabel("y", fontsize=fontsize)
    ax.set_title("Level Set Function", fontsize=fontsize + 2)

    # Improve layout
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    # Display the plot
    plt.show()


def plot_levels_2D(
    data, indices, xlim, ylim, theme="scientific", fontsize=12, save_path=None, dpi=300
):
    colors = [(1, 1, 1), (0.2, 0.2, 0.2)]
    cmap = ListedColormap(colors)

    if theme == "scientific":
        plt.style.use("seaborn-v0_8-paper")

    # Set up the figure and axes
    num_plots = len(indices)
    xlen = xlim[1] - xlim[0]
    ylen = ylim[1] - ylim[0]
    fig, axes = plt.subplots(
        num_plots, 1, figsize=(5.0, (5.5 * ylen / xlen) * num_plots), sharex=True
    )
    if num_plots == 1:  # Handle the case of a single subplot
        axes = [axes]

    for ax, i in zip(axes, indices):
        dti = data[i]
        x_coords, y_coords = dti["coords"]
        cells = dti["cells"]
        cell_values = dti["cell_values"]

        ax.tripcolor(
            x_coords,
            y_coords,
            cells,
            facecolors=cell_values,
            shading="flat",
            linewidth=0.5,
            cmap=cmap,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        # ax.set_ylabel(f"i = {i}", fontsize=fontsize)
        ax.set_title(f"i = {i}", fontsize=fontsize)

    plt.tight_layout()

    # Save or show the plots
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()


def plot_magnitude(
    dim,
    points,
    cells,
    values,
    rank_dimension,
    boundaries=None,
    figsize=None,
    save_path=None,
):
    print("\n> plot magnitude")
    if dim == 2:
        x_coords, y_coords = points[:, 0], points[:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        eps = min(5.0 * x_range / 100.0, 5.0 * y_range / 100.0)

        triang = mtri.Triangulation(x_coords, y_coords, cells)

        if rank_dimension == 1:
            values = abs(values)
        elif rank_dimension == 2:
            vec = values.reshape((-1, 2))
            # vec = values.copy()
            values = np.linalg.norm(vec, axis=1)

        print(f"> minimum = {min(values)}, maximum = {max(values)}")
        # norm = Normalize(vmin=0, vmax=7.5)
        fig, ax = plt.subplots(figsize=figsize)
        tpc = ax.tripcolor(
            triang,
            values,
            shading="gouraud",
            cmap="viridis",
            # norm = norm
        )
        fig.colorbar(tpc, ax=ax)

        if rank_dimension == 2:
            ax.quiver(
                x_coords,
                y_coords,
                vec[:, 0] / values,
                vec[:, 1] / values,
                units="xy",
                color="black",
                alpha=0.8,
            )

        ax.set_aspect("equal")
        ax.set_xlim(x_coords.min() - eps, x_coords.max() + eps)
        ax.set_ylim(y_coords.min() - eps, y_coords.max() + eps)

        if boundaries:
            for xs, ys, cl in boundaries:
                ax.plot(xs, ys, color=cl)

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, format=save_path.split(".")[-1], dpi=300)
            plt.close()
        else:
            plt.show()
