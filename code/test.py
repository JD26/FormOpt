import formopt as fop

# Pre-existing models
from models import (
    Logistic,
    Compliance,
    CompliancePlus,
    InverseElasticity,
    Heat,
    HeatPlus,
    Mechanism,
    Gripping,
    SVK,
    NHK,
)

import numpy as np
from pathlib import Path  # To manage where results are saved

# Necessary for all parallelism modes
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

"""
Tests
-----
test_01 : Symmetric Cantilever 2D - Data Parallelism
test_02 : Symmetric cantilever 3D - Data Parallelism
test_03 : Cantilever with two loads I - Data Parallelism
test_04 : Cantilever with two loads I - Task Parallelism
test_05 : Cantilever with two loads I - Mixed Parallelism
test_06 : Elasticity Inverse Problem - Data Parallelism
test_07 : Elasticity Inverse Problem - Task Parallelism
test_08 : Elasticity Inverse Problem - Mixed Parallelism
test_09 : Heat conduction 1 - Data Parallelism
test_10 : Heat conduction 2 - Data Parallelism
test_11 : Heat conduction with one source - Data Parallelism
test_12 : Heat conduction with two sinks (single) - Data Parallelism
test_13 : Heat conduction with two sinks (multiple) - Task Parallelism
test_14 : Logistic equation (r = 10) - Data Parallelism
test_15 : Logistic equation (r = 40) - Data Parallelism
test_16 : Logistic equation (r = 100) - Data Parallelism
test_17 : ------
test_18 : Cantilever with two loads II - Data Parallelism
test_19 : Cantilever with two loads II - Task Parallelism
test_20 : Symmetric Cantilever 2D (non-rectangular domain) - Data Parallelism
test_21 : Heat conduction with four sources (single) - Data Parallelism
test_22 : Heat conduction with four sources (multiple) - Task Parallelism
test_23 : Heat conduction with four sources (multiple) - Mixed Parallelism
test_31 : Mechanism (Antoine - in progress) 
test_32 : Execution times
test_33 : Gripping Mechanism 2D - Data Parallelism
test_34 : Elasticity Inverse Problem (two inclusions) - Data Parallelism
test_35 : Elasticity Inverse Problem (two inclusions) - Task Parallelism
test_36 : Elasticity Inverse Problem (two inclusions) - Mixed Parallelism
test_37 : Symmetric Cantilever 2D (SVK)
test_42 : Symmetric cantilever 3D - Data Parallelism (1 process)
test_43 : Symmetric cantilever 3D - Data Parallelism (2 process)
test_44 : Symmetric cantilever 3D - Data Parallelism (4 process)
test_45 : Symmetric Cantilever 2D with several Ersatz material
"""


def test_01():
    """
    Symmetric Cantilever 2D - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 01`.
    For instance, `mpirun -np 4 python test.py 01`.

    To save the output, append `> ../results/t01/out.txt`.
    To delete the images, enter `rm ../results/t01/*.png`.
    """

    test_name = "Symmetric Cantilever 2D - Data Parallelism"
    test_path = Path("../results/t01/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    vertices = np.array(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.45), (2.0, 0.55), (2.0, 1.0), (0.0, 1.0)]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [(dir_idx, dir_mkr, "dir"), (neu_idx, neu_mkr, "neu")]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)

    # Dirichlet condition
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )

    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])

    area = 1.0
    g = (0.0, -2.0)
    # Create the model
    md = Compliance(dim, domain, space, g, ds_g[0], dirichlet_bcs, area, test_path)

    @fop.region_of(domain)
    def sub_domain(x):
        # 0.42 < x[1] < 0.58
        # 1.95 < x[0]
        ineqs = [x[1] - 0.42, 0.58 - x[1], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    # Initial guess: centers and radii

    # First set of centers (for example in manuscript):
    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * 0.7, 0.0) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.25) for i in range(2)]
    centers += [(0.3 + i * 0.7, 0.5) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.75) for i in range(2)]
    centers += [(0.3 + i * 0.7, 1.0) for i in range(3)]

    # Second set of centers:
    # centers = [(0.0, 0.5), (2.0, 0.35), (2.0, 0.65)]
    # centers += [((1 + i) * 0.25, 0.0) for i in range(8)]
    # centers += [(i * 0.5, 0.25) for i in range(5)]
    # centers += [(0.25 + i * 0.5, 0.5) for i in range(4)]
    # centers += [(i * 0.5, 0.75) for i in range(5)]
    # centers += [((1 + i) * 0.25, 1.0) for i in range(8)]

    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])

    # Create the initial level set function
    md.create_initial_level(centers, radii)
    # Save as initial.xdmf
    md.save_initial_level(comm)

    md.runDP(
        ctrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(20, 0.1),
        smooth=True,
    )


def test_02():
    """
    Run: mpirun -np <nbr of processes> python test.py 02
    Recommended: python test.py 02
    """

    test_name = "Symmetric cantilever 3D (Data Parallelism)"
    test_path = Path("../results/t44/")
    dim = 3
    rank_dim = 3
    mesh_size = 60

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")

    # Create a dolfinx domain
    from dolfinx.mesh import create_box

    domain = create_box(
        comm, [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], [2 * mesh_size, mesh_size, mesh_size]
    )

    fop.all_connectivities(domain)
    fop.save_domain(comm, domain, test_path / "domain.xdmf")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)

    # marker functions
    def boundary_dirichlet(x):
        return np.isclose(x[0], 0.0)

    def boundary_neumann(x):
        in_plane = np.isclose(x[0], 2.0)
        in_square = np.maximum(np.abs(x[1] - 0.5), np.abs(x[2] - 0.5)) <= 0.1
        return in_plane & in_square

    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet_fun(
        domain, space, [boundary_dirichlet], rank_dim
    )
    # Boundary to force application
    ds_g = fop.fun_ds(domain, [boundary_neumann])
    volume, g = 1.0, [0.0, 0.0, -4.0]
    # Create the model
    md = Compliance(dim, domain, space, g, ds_g[0], dirichlet_bcs, volume, test_path)

    @fop.region_of(domain)
    def sub_domain(x):
        ineqs = [x[1] - 0.35, 0.65 - x[1], x[2] - 0.35, 0.65 - x[2], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    centers = np.array(
        [
            (2.0, 0.7, 0.3),
            (2.0, 0.7, 0.7),
            (2.0, 0.3, 0.3),
            (2.0, 0.3, 0.7),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
            (2.0, 0.5, 0.0),
            (2.0, 0.5, 1.0),
            (2.0, 0.0, 0.5),
            (2.0, 1.0, 0.5),
            (1.7, 0.0, 0.0),
            (1.7, 0.0, 1.0),
            (1.7, 1.0, 0.0),
            (1.7, 1.0, 1.0),
            (1.7, 0.5, 0.5),
            (1.7, 0, 0.5),
            (1.7, 1.0, 0.5),
            (1.7, 0.5, 0.0),
            (1.7, 0.5, 1.0),
            (1.35, 0.25, 0.25),
            (1.35, 0.75, 0.25),
            (1.35, 0.25, 0.75),
            (1.35, 0.75, 0.75),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.5, 0.5),
            (1.0, 0, 0.5),
            (1.0, 1.0, 0.5),
            (1.0, 0.5, 0.0),
            (1.0, 0.5, 1.0),
            (0.65, 0.25, 0.25),
            (0.65, 0.75, 0.25),
            (0.65, 0.25, 0.75),
            (0.65, 0.75, 0.75),
            (0.3, 0.0, 0.0),
            (0.3, 0.0, 1.0),
            (0.3, 1.0, 0.0),
            (0.3, 1.0, 1.0),
            (0.3, 0.5, 0.5),
            (0.3, 0, 0.5),
            (0.3, 1.0, 0.5),
            (0.3, 0.5, 0.0),
            (0.3, 0.5, 1.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.5, 1.0),
            (0.0, 0.0, 0.5),
            (0.0, 1.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.0, 0.25, 0.25),
            (0.0, 0.75, 0.25),
            (0.0, 0.25, 0.75),
            (0.0, 0.75, 0.75),
            (2.0, 0.5, 0.3),
            (2.0, 0.5, 0.7),
            (2.0, 0.3, 0.5),
            (2.0, 0.7, 0.5),
        ]
    )
    radii = np.repeat(0.1, centers.shape[0])

    md.create_initial_level(centers, radii, ord=np.inf)
    md.save_initial_level(comm)

    # Run data parallelism
    md.runDP(
        ctrn_tol=1e-3, dfactor=1e-1, reinit_step=4, reinit_pars=(4, 0.01), smooth=True
    )


def test_03():
    """
    Run: mpirun -np <nbr of processes> python test.py 03
    For instance: mpirun -np 2 python test.py 03
    """

    test_name = "Multiple load cases - Data Parallelism"
    test_path = Path("../results/t03/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.012

    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [1.0, 0.9], [1.0, 1.0], [0.0, 1.0]]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_top, neu_mkr_top = [4], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_top, neu_mkr_top, "neu_top"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr_bot, neu_mkr_top])
    area = 0.5
    g = [(0.0, -2.0), (0.0, -2.0)]
    # Create the model
    md = CompliancePlus(dim, domain, space, g, ds_g, dirichlet_bcs, area, test_path)

    # Initial guess: centers and radii
    centers = []
    centers += [(i * 0.2, 0.25) for i in range(5)]
    centers += [(0.1 + i * 0.2, 0.5) for i in range(5)]
    centers += [(i * 0.2, 0.75) for i in range(5)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])

    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    # Run Data Parallelism
    md.runDP(
        ctrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(16, 0.05),
        smooth=True,
    )


def test_04():
    """
    Run: mpirun -np 2 python test.py 04
    """

    # Verification
    task_nbr = 2
    if size != task_nbr:
        print(f"Number of processes must be = {task_nbr}")
        return

    comm_self = MPI.COMM_SELF

    test_name = "Cantilever with two loads I (Task Parallelism)"
    test_path = Path("../results/t04/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.012

    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [1.0, 0.9], [1.0, 1.0], [0.0, 1.0]]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_top, neu_mkr_top = [4], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_top, neu_mkr_top, "neu_top"),
    ]

    # Create gmsh domain for Task Parallelism
    output = fop.create_domain_2d_TP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=True
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr_bot, neu_mkr_top])
    area = 0.5
    g = [(0.0, -2.0), (0.0, -2.0)]
    # Create the model
    md = CompliancePlus(dim, domain, space, g, ds_g, dirichlet_bcs, area, test_path)

    # Initial guess: centers and radii
    centers = []
    centers += [(i * 0.2, 0.25) for i in range(5)]
    centers += [(0.1 + i * 0.2, 0.5) for i in range(5)]
    centers += [(i * 0.2, 0.75) for i in range(5)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])

    md.create_initial_level(centers, radii)
    if rank == 0:
        md.save_initial_level(comm_self)

    # Run Task Parallelism
    md.runTP(
        ctrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(16, 0.05),
        smooth=True,
    )


def test_05():
    """
    Run: mpirun -np <2n> python test.py 05
    For instance: mpirun -np 4 python test.py 05

    Execution times:
    - 2 processes (70 iterations):
        > Assembly time = 5.006272704000001 s
        > Resolution time = 25.917571647000003 s
    - 4 processes (70 iterations):
        > Assembly time = 7.219901673999999 s
        > Resolution time = 14.377620046999994 s
    - 6 processes (71 iterations):
        > Assembly time = 7.494962302000005 s
        > Resolution time = 10.311590593999995 s
    - 8 processes (68 iterations):
        > Assembly time = 10.915818561999998 s
        > Resolution time = 14.416016135999996 s
    - 10 processes (71 iterations):
        > Assembly time = 11.317666292000013 s
        > Resolution time = 13.314896329000021 s

    resolution_times = [
        25.917571647000003,
        14.377620046999994,
        10.311590593999995,
        14.416016135999996,
        13.314896329000021
    ]
    """

    # Verification
    nbr_groups = 2
    if size % nbr_groups != 0:
        print(f"Nbr of processes must be divisible by {nbr_groups}")
        return

    # Subcommunicators
    color = rank * nbr_groups // size
    sub_comm = comm.Split(color, rank)

    test_name = "Multiple load cases (Mix Parallelism)"
    test_path = Path("../results/t05/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.012

    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [1.0, 0.9], [1.0, 1.0], [0.0, 1.0]]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_top, neu_mkr_top = [4], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_top, neu_mkr_top, "neu_top"),
    ]

    # Create gmsh domain for Mix Parallelism
    output = fop.create_domain_2d_MP(
        sub_comm, color, vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr_bot, neu_mkr_top])
    area = 0.5
    g = [(0.0, -2.0), (0.0, -2.0)]
    # Create the model
    md = CompliancePlus(dim, domain, space, g, ds_g, dirichlet_bcs, area, test_path)

    # Initial guess: centers and radii
    centers = []
    centers += [(i * 0.2, 0.25) for i in range(5)]
    centers += [(0.1 + i * 0.2, 0.5) for i in range(5)]
    centers += [(i * 0.2, 0.75) for i in range(5)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])

    md.create_initial_level(centers, radii)
    if color == 0:
        md.save_initial_level(sub_comm)

    # Run Mix Parallelism
    md.runMP(
        sub_comm=sub_comm,
        ctrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(16, 0.05),
        smooth=True,
    )


def test_06():
    """
    Elasticity Inverse Problem - Data Parallelism

    Run `mpirun -np <number of processes> python test.py 06`.
    For instance, `mpirun -np 6 python test.py 06`.

    To save the output, append `> ../results/t06/out.txt`.
    To delete the images, enter `rm ../results/t06/*.png`.
    """

    test_name = "Elasticity Inverse Problem - Data Parallelism"
    test_path = Path("../results/t06/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps) / b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a * np.cos(t)
        y = b * np.sin(t) + (b - eps)
        return x, y

    npts = 80  # npts % 4 = 0
    part = npts // 4

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.2, npts))

    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part // 2 + 1), 2
    neu_idxA, neu_mkrA = part // 2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part // 2 + np.arange(part + 1, 2 * part + 1), 4
    neu_idxC, neu_mkrC = part // 2 + np.arange(2 * part + 1, 3 * part + 1), 5
    bL_idx, bL_mkr = np.arange(part // 2 + 3 * part + 1, npts), 6

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    def bean_curve(n):
        """
        Coordinates of a bean to be used
        as a subdomain.
        """
        ampl = 0.5
        x0 = 0.0
        y0 = 0.2
        angle = np.pi / 6.0
        t = np.linspace(0, np.pi, n, endpoint=False)
        r = ampl * (np.cos(t) ** 3 + np.sin(t) ** 3)
        x = r * np.cos(t) + x0
        y = r * np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_ * x - sin_ * y
        y_rot = sin_ * x + cos_ * y
        return x_rot, y_rot

    subdomain = np.column_stack(bean_curve(60))

    if rank == 0:
        np.save(test_path / "subdomain.npy", subdomain)

    filename = test_path / "domain0.msh"

    # Create the gmsh domain0.msh
    nbr_tri0 = fop.build_gmsh_model_2d(
        vertices,
        boundary_parts,
        0.6 * mesh_size,
        curves=[subdomain],
        filename=filename,
        plot=False,
    )

    # Read the domain0
    output = fop.read_gmsh(filename, comm, dim=2)

    domain0, _, boundary_tags = output

    # Set all connectivities on domain0
    fop.all_connectivities(domain0)

    # Space for data generation
    space0 = fop.create_space(domain0, "CG", rank_dim)
    # Forces
    forces = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]

    # Dirichlet boundary conditions
    dirbc_partial = fop.homogeneous_dirichlet(
        domain0, space0, boundary_tags, [dir_mkr], rank_dim
    )

    dirbc_total = fop.homogeneous_boundary(domain0, space0, dim, rank_dim)

    # Create measures to apply Neumman condition
    ds_parts = fop.marked_ds(
        domain0, boundary_tags, [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
    )

    # Measures for force application
    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]]

    # Measure for adjoint problem
    ds1 = sum(ds_parts[1:], start=ds_parts[0])

    # Instance for data generation
    # We need the method pde0
    md0 = InverseElasticity(
        dim,
        domain0,
        space0,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        test_path,
    )

    # Function that defines
    # the level set function
    # to generate the data
    def beam_equation(x):
        """
        Implicit equation :
        ((x-x0)^2 + (y-y0)^2)^2 = a((x-x0)^3 + (y-y0)^3)
        """
        ampl = 0.5
        x0 = 0.0
        y0 = 0.2
        angle = np.pi / 6.0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        # inverse rotation
        x_irot = cos_ * x[0] + sin_ * x[1] - x0
        y_irot = -sin_ * x[0] + cos_ * x[1] - y0
        left_part = (x_irot**2 + y_irot**2) ** 2
        right_part = ampl * (x_irot**3 + y_irot**3)
        values = left_part - right_part
        return np.log(25.0 * values + 1.0)

    # Dirichlet extensions
    extensions = fop.dir_extension_from(
        comm, domain0, space0, md0.pde0, beam_equation, test_path
    )

    npts = 80  # npts % 4 = 0
    part = npts // 4

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.2, npts))

    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part // 2 + 1), 2
    neu_idxA, neu_mkrA = part // 2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part // 2 + np.arange(part + 1, 2 * part + 1), 4
    neu_idxC, neu_mkrC = part // 2 + np.arange(2 * part + 1, 3 * part + 1), 5
    bL_idx, bL_mkr = np.arange(part // 2 + 3 * part + 1, npts), 6

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    dirbc_total = fop.homogeneous_boundary(domain, space, dim, rank_dim)
    # Boundary to apply Neumann conditions
    ds_parts = fop.marked_ds(
        domain, boundary_tags, [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
    )

    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]]
    ds1 = sum(ds_parts[1:], start=ds_parts[0])
    forces = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]
    # Create the model
    md = InverseElasticity(
        dim,
        domain,
        space,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        test_path,
    )

    # Space for interpolation (degree = 2)
    g_space = fop.create_space(domain, "CG", rank_dim, degree=2)
    # Interpolation between different spaces
    # from different domains
    g_funcs = fop.space_interpolation(
        from_space=space0, funcs=extensions, to_space=g_space
    )

    # To save as P1 functions
    g_space_1 = fop.create_space(domain, "CG", rank_dim)
    g_funcs_1 = fop.interpolate(funcs=g_funcs, to_space=g_space_1, name="g")
    fop.save_functions(comm, domain, g_funcs_1, test_path / "gP1.xdmf")

    md.gs = g_funcs

    # Initial guess: centers and radii
    centers = np.array([(0.0, 0.4)])
    radii = np.array([0.15])

    md.create_initial_level(centers, radii, factor=-1.0)
    md.save_initial_level(comm)

    # Run Data Parallelism
    md.runDP(
        niter=200,
        dfactor=1e-1,
        lv_time=(1e-3, 1.0),
        cost_tol=1e-1,
        random_pars=(26, 0.05),
    )


def test_07():
    """
    Elasticity Inverse Problem - Task Parallelism

    Run `mpirun -np 6 python test.py 07`.

    To save the output, append `> ../results/t07/out.txt`.
    To delete the images, enter `rm ../results/t07/*.png`.
    """

    # Verification
    task_nbr = 6
    if size != task_nbr:
        print(f"Nbr of processes must be = {task_nbr}")
        return

    comm_self = MPI.COMM_SELF

    test_name = "Elasticity Inverse Problem - Task Parallelism"
    test_path = Path("../results/t07/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps) / b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a * np.cos(t)
        y = b * np.sin(t) + (b - eps)
        return x, y

    npts = 80  # npts % 4 = 0
    part = npts // 4

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.2, npts))

    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part // 2 + 1), 2
    neu_idxA, neu_mkrA = part // 2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part // 2 + np.arange(part + 1, 2 * part + 1), 4
    neu_idxC, neu_mkrC = part // 2 + np.arange(2 * part + 1, 3 * part + 1), 5
    bL_idx, bL_mkr = np.arange(part // 2 + 3 * part + 1, npts), 6

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    def bean_curve(n):
        """
        Coordinates of a bean to be used
        as a subdomain.
        """
        ampl = 0.5
        x0 = 0.0
        y0 = 0.2
        angle = np.pi / 6.0
        t = np.linspace(0, np.pi, n, endpoint=False)
        r = ampl * (np.cos(t) ** 3 + np.sin(t) ** 3)
        x = r * np.cos(t) + x0
        y = r * np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_ * x - sin_ * y
        y_rot = sin_ * x + cos_ * y
        return x_rot, y_rot

    subdomain = np.column_stack(bean_curve(60))

    if rank == 0:
        np.save(test_path / "subdomain.npy", subdomain)

    filename = test_path / "domain0.msh"

    nbr_tri0 = fop.build_gmsh_model_2d(
        vertices,
        boundary_parts,
        0.6 * mesh_size,
        curves=[subdomain],
        filename=filename,
        plot=False,
    )

    if rank == 0:
        # Read the domain0 in rank = 0
        output = fop.read_gmsh(filename, comm_self, 2)

        domain0, _, boundary_tags = output
        fop.all_connectivities(domain0)

        # Space defined in rank = 0
        space0 = fop.create_space(domain0, "CG", rank_dim)
        # three forces
        forces = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]

        # Dirichlet boundary conditions
        dirbc_partial = fop.homogeneous_dirichlet(
            domain0, space0, boundary_tags, [dir_mkr], rank_dim
        )

        dirbc_total = fop.homogeneous_boundary(domain0, space0, dim, rank_dim)

        # Measures
        ds_parts = fop.marked_ds(
            domain0, boundary_tags, [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
        )

        ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]]
        ds1 = sum(ds_parts[1:], start=ds_parts[0])

        md0 = InverseElasticity(
            dim,
            domain0,
            space0,
            forces,
            ds_forces,
            ds1,
            dirbc_partial,
            dirbc_total,
            test_path,
        )

        def beam_equation(x):
            """
            Implicit equation :
            ((x-x0)^2 + (y-y0)^2)^2 = a((x-x0)^3 + (y-y0)^3)
            """
            ampl = 0.5
            x0 = 0.0
            y0 = 0.2
            angle = np.pi / 6.0
            cos_ = np.cos(angle)
            sin_ = np.sin(angle)
            # inverse rotation
            x_irot = cos_ * x[0] + sin_ * x[1] - x0
            y_irot = -sin_ * x[0] + cos_ * x[1] - y0
            vals = (x_irot**2 + y_irot**2) ** 2 - ampl * (x_irot**3 + y_irot**3)
            return np.log(25 * vals + 1.0)

        extensions = fop.dir_extension_from(
            comm_self, domain0, space0, md0.pde0, beam_equation, test_path
        )

    npts = 80  # npts % 4 = 0
    part = npts // 4

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.2, npts))

    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part // 2 + 1), 2
    neu_idxA, neu_mkrA = part // 2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part // 2 + np.arange(part + 1, 2 * part + 1), 4
    neu_idxC, neu_mkrC = part // 2 + np.arange(2 * part + 1, 3 * part + 1), 5
    bL_idx, bL_mkr = np.arange(part // 2 + 3 * part + 1, npts), 6

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    # Create gmsh domain for Task Parallelism
    output = fop.create_domain_2d_TP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    dirbc_total = fop.homogeneous_boundary(domain, space, dim, rank_dim)
    # Boundary to force application
    ds_parts = fop.marked_ds(
        domain, boundary_tags, [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
    )

    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]]
    ds1 = sum(ds_parts[1:], start=ds_parts[0])
    forces = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]

    # Create the model
    md = InverseElasticity(
        dim,
        domain,
        space,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        test_path,
    )

    g_space = fop.create_space(domain, "CG", rank_dim, degree=2)
    if rank == 0:
        # Interpolation
        g_funcs = fop.space_interpolation(
            from_space=space0, funcs=extensions, to_space=g_space
        )

        # Save
        g_space_1 = fop.create_space(domain, "CG", rank_dim)
        g_funcs_1 = fop.interpolate(funcs=g_funcs, to_space=g_space_1, name="g")
        fop.save_functions(comm_self, domain, g_funcs_1, test_path / "gP1.xdmf")

        g_values = np.vstack([g.x.array[:] for g in g_funcs])

    else:
        g_values = None

    g_values = comm.bcast(g_values, root=0)

    md.gs = fop.get_funcs_from(g_space, g_values)

    # Initial guess: centers and radii
    centers = np.array([(0.0, 0.4)])
    radii = np.array([0.15])

    md.create_initial_level(centers, radii, factor=-1.0)
    if rank == 0:
        md.save_initial_level(comm_self)

    # Run Task Parallelism
    md.runTP(niter=200, dfactor=1e-1, lv_time=(1e-3, 1.0), cost_tol=1e-1)


def test_08():
    """
    Elasticity Inverse Problem - Mixed Parallelism

    Run `mpirun -np <6n> python test.py 08`.
    For instance, `mpirun -np 12 python test.py 08`.

    To save the output, append `> ../results/t08/out.txt`.
    To delete the images, enter `rm ../results/t08/*.png`.
    """

    # Verification
    nbr_groups = 6
    if size % nbr_groups != 0:
        print(f"Nbr of processes must be divisible by {nbr_groups}")
        return

    # Subcommunicators
    color = rank * nbr_groups // size
    sub_comm = comm.Split(color, rank)

    test_name = "Elasticity Inverse Problem - Mixed Parallelism"
    test_path = Path("../results/t08/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps) / b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a * np.cos(t)
        y = b * np.sin(t) + (b - eps)
        return x, y

    npts = 80  # npts % 4 = 0
    part = npts // 4

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.2, npts))

    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part // 2 + 1), 2
    neu_idxA, neu_mkrA = part // 2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part // 2 + np.arange(part + 1, 2 * part + 1), 4
    neu_idxC, neu_mkrC = part // 2 + np.arange(2 * part + 1, 3 * part + 1), 5
    bL_idx, bL_mkr = np.arange(part // 2 + 3 * part + 1, npts), 6

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    def bean_curve(n):
        """
        Coordinates of a bean to be used
        as a subdomain.
        """
        ampl = 0.5
        x0 = 0.0
        y0 = 0.2
        angle = np.pi / 6.0
        t = np.linspace(0, np.pi, n, endpoint=False)
        r = ampl * (np.cos(t) ** 3 + np.sin(t) ** 3)
        x = r * np.cos(t) + x0
        y = r * np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_ * x - sin_ * y
        y_rot = sin_ * x + cos_ * y
        return x_rot, y_rot

    subdomain = np.column_stack(bean_curve(60))

    if rank == 0:
        np.save(test_path / "subdomain.npy", subdomain)

    filename = test_path / "domain0.msh"

    nbr_tri0 = fop.build_gmsh_model_2d(
        vertices,
        boundary_parts,
        0.6 * mesh_size,
        curves=[subdomain],
        filename=filename,
        plot=False,
    )

    if color == 0:
        # Read the domain0 in rank = 0
        output = fop.read_gmsh(filename, sub_comm, 2)

        domain0, _, boundary_tags = output
        fop.all_connectivities(domain0)

        # Space defined in rank = 0
        space0 = fop.create_space(domain0, "CG", rank_dim)
        # three forces
        forces = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]

        # Dirichlet boundary conditions
        dirbc_partial = fop.homogeneous_dirichlet(
            domain0, space0, boundary_tags, [dir_mkr], rank_dim
        )

        dirbc_total = fop.homogeneous_boundary(domain0, space0, dim, rank_dim)

        # Measures
        ds_parts = fop.marked_ds(
            domain0, boundary_tags, [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
        )

        ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]]
        ds1 = sum(ds_parts[1:], start=ds_parts[0])

        md0 = InverseElasticity(
            dim,
            domain0,
            space0,
            forces,
            ds_forces,
            ds1,
            dirbc_partial,
            dirbc_total,
            test_path,
        )

        def beam_equation(x):
            """
            Implicit equation :
            ((x-x0)^2 + (y-y0)^2)^2 = a((x-x0)^3 + (y-y0)^3)
            """
            ampl = 0.5
            x0 = 0.0
            y0 = 0.2
            angle = np.pi / 6.0
            cos_ = np.cos(angle)
            sin_ = np.sin(angle)
            # inverse rotation
            x_irot = cos_ * x[0] + sin_ * x[1] - x0
            y_irot = -sin_ * x[0] + cos_ * x[1] - y0
            left_part = (x_irot**2 + y_irot**2) ** 2
            right_part = ampl * (x_irot**3 + y_irot**3)
            values = left_part - right_part
            return np.log(25.0 * values + 1.0)

        extensions = fop.dir_extension_from(
            sub_comm, domain0, space0, md0.pde0, beam_equation, test_path
        )

    npts = 80  # npts % 4 = 0
    part = npts // 4

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.2, npts))

    dir_idx, dir_mkr = [npts], 1
    bR_idx, bR_mkr = np.arange(1, part // 2 + 1), 2
    neu_idxA, neu_mkrA = part // 2 + np.arange(1, part + 1), 3
    neu_idxB, neu_mkrB = part // 2 + np.arange(part + 1, 2 * part + 1), 4
    neu_idxC, neu_mkrC = part // 2 + np.arange(2 * part + 1, 3 * part + 1), 5
    bL_idx, bL_mkr = np.arange(part // 2 + 3 * part + 1, npts), 6

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (bR_idx, bR_mkr, "bRight"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (bL_idx, bL_mkr, "bLeft"),
    ]

    # Create gmsh domain for Mix Parallelism
    output = fop.create_domain_2d_MP(
        sub_comm, color, vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )
    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    dirbc_total = fop.homogeneous_boundary(domain, space, dim, rank_dim)

    # Boundary to force application
    ds_parts = fop.marked_ds(
        domain, boundary_tags, [bR_mkr, neu_mkrA, neu_mkrB, neu_mkrC, bL_mkr]
    )

    ds_forces = [ds_parts[1], ds_parts[2], ds_parts[3]]
    ds1 = sum(ds_parts[1:], start=ds_parts[0])
    forces = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]
    # Create the model
    md = InverseElasticity(
        dim,
        domain,
        space,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        test_path,
    )

    g_space = fop.create_space(domain, "CG", rank_dim, degree=2)

    if color == 0:
        # Interpolation
        g_funcs = fop.space_interpolation(
            from_space=space0, funcs=extensions, to_space=g_space
        )

        # Save
        g_space_1 = fop.create_space(domain, "CG", rank_dim)
        g_funcs_1 = fop.interpolate(funcs=g_funcs, to_space=g_space_1, name="g")
        fop.save_functions(sub_comm, domain, g_funcs_1, test_path / "gP1.xdmf")

        g_values_loc = np.vstack([g.x.array[:] for g in g_funcs])

    else:
        g_values_loc = None

    g_values = comm.allgather(g_values_loc)

    md.gs = fop.get_funcs_from(g_space, g_values[sub_comm.rank])

    # Initial guess: centers and radii
    centers = np.array([(0.0, 0.4)])
    radii = np.array([0.15])

    md.create_initial_level(centers, radii, -1.0)
    if color == 0:
        md.save_initial_level(sub_comm)

    # Run Mixed Parallelism
    md.runMP(sub_comm, niter=200, dfactor=1e-1, lv_time=(1e-3, 1.0), cost_tol=1e-1)


def test_09():
    """
    Heat conduction 1 - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 09`.
    For instance, `mpirun -np 2 python test.py 09`.

    To save the output, append `> ../results/t09/out.txt`.
    To delete the images, enter `rm ../results/t09/*.png`.
    """

    test_name = "Heat conduction 1 - Data Parallelism"
    test_path = Path("../results/t09/")
    dim = 2
    rank_dim = 1
    mesh_size = 5e-3

    vertices = np.array(
        [[0.0, 0.0], [0.4, 0.0], [0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )

    dir_idx, dir_mkr = [2], 1

    boundary_parts = [(dir_idx, dir_mkr, "dir")]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    area = 0.25
    # Create the model
    md = Heat(dim, domain, space, dirichlet_bcs, area, test_path)

    @fop.region_of(domain)
    def sub_domain(x):
        ineqs = [x[0] - 0.3, 0.7 - x[0], 0.05 - x[1]]
        return ineqs

    md.sub = [sub_domain.expression()]

    centers = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0)]

    centers += [(0.0, i * 0.1) for i in range(10)]
    centers += [(1.0, i * 0.1) for i in range(10)]
    centers += [(i * 0.1, 1.0) for i in range(11)]

    centers += [(0.25, i * 0.2) for i in range(1, 5)]
    centers += [(0.5, i * 0.2 + 0.1) for i in range(1, 5)]
    centers += [(0.75, i * 0.2) for i in range(1, 5)]

    centers = np.array(centers)
    radii = np.repeat(0.08, 49)

    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    # Run Data Parallelism
    md.runDP(niter=250, ctrn_tol=1e-3, lgrn_tol=1e-2)


def test_10():
    """
    Heat conduction 2 - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 10`.
    For instance, `mpirun -np 4 python test.py 10`.

    To save the output, append `> ../results/t10/out.txt`.
    To delete the images, enter `rm ../results/t10/*.png`.
    """

    test_name = "Heat conduction 2 - Data Parallelism"
    test_path = Path("../results/t10/")
    dim = 2
    rank_dim = 1
    mesh_size = 4e-3

    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    dir1_idx, dir1_mkr = [1], 1
    dir2_idx, dir2_mkr = [2], 2
    dir3_idx, dir3_mkr = [3], 3
    dir4_idx, dir4_mkr = [4], 4

    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2"),
        (dir3_idx, dir3_mkr, "dir3"),
        (dir4_idx, dir4_mkr, "dir4"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_boundary(domain, space, dim, rank_dim)
    area = 0.6
    # Create the model
    md = Heat(dim, domain, space, dirichlet_bcs, area, test_path)

    @fop.region_of(domain)
    def sub_domain1(x):
        return [0.1 - x[1]]

    @fop.region_of(domain)
    def sub_domain2(x):
        return [x[0] - 0.9]

    @fop.region_of(domain)
    def sub_domain3(x):
        return [x[1] - 0.9]

    @fop.region_of(domain)
    def sub_domain4(x):
        return [0.1 - x[0]]

    md.sub = [
        sub_domain1.expression(),
        sub_domain2.expression(),
        sub_domain3.expression(),
        sub_domain4.expression(),
    ]
    centers = []
    centers += [(0.3, i * 0.2 + 0.1) for i in range(1, 4)]
    centers += [(0.5, i * 0.2 + 0.1) for i in range(1, 4)]
    centers += [(0.7, i * 0.2 + 0.1) for i in range(1, 4)]
    centers += [(0.2, i * 0.2 + 0.2) for i in range(4)]
    centers += [(0.4, i * 0.2 + 0.2) for i in range(4)]
    centers += [(0.6, i * 0.2 + 0.2) for i in range(4)]
    centers += [(0.8, i * 0.2 + 0.2) for i in range(4)]

    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])

    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(niter=200, dfactor=1e-1, ctrn_tol=1e-3)


def test_11():
    """
    Heat conduction with one source - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 11`.
    For instance, `mpirun -np 6 python test.py 11`.

    To save the output, append `> ../results/t11/out.txt`.
    To delete the images, enter `rm ../results/t11/*.png`.
    """

    test_name = "Heat conduction with one source - Data Parallelism"
    test_path = Path("../results/t11/")
    dim = 2
    rank_dim = 1
    mesh_size = 1e-2

    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    dir1_idx, dir1_mkr = [1], 1
    dir2_idx, dir2_mkr = [2], 2
    dir3_idx, dir3_mkr = [3], 3
    dir4_idx, dir4_mkr = [4], 4

    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2"),
        (dir3_idx, dir3_mkr, "dir3"),
        (dir4_idx, dir4_mkr, "dir4"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_boundary(domain, space, dim, rank_dim)
    area = 0.5
    # Create the model
    md = Heat(dim, domain, space, dirichlet_bcs, area, test_path, "1Load")

    centers = []

    centers += [(i / 3.0, 1.0) for i in range(4)]
    centers += [(i / 3.0, 2.0 / 3.0) for i in range(4)]
    centers += [(i / 3.0, 1.0 / 3.0) for i in range(4)]
    centers += [(i / 3.0, 0.0) for i in range(4)]

    # Uncomment to add more holes to the initial condition
    # Spoilier: the result is different
    # centers += [(i/6.0, 5.0/6.0) for i in range(1, 6, 2)]
    # centers += [(i/6.0, 0.5) for i in range(1, 6, 2) if i != 3]
    # centers += [(i/6.0, 1.0/6.0) for i in range(1, 6, 2)]

    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])

    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(
        niter=250,
        dfactor=1.0,
        ctrn_tol=1e-3,
        lgrn_tol=1e-3,
        smooth=True,
        reinit_step=6,
        reinit_pars=(8, 0.01),
    )


def test_12():
    """
    Heat conduction with two sinks (single) - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 12`.
    For instance, `mpirun -np 2 python test.py 12`.

    To save the output, append `> ../results/t12/out.txt`.
    To delete the images, enter `rm ../results/t12/*.png`.
    """

    test_name = "Heat conduction with two sinks (single) - Data Parallelism"
    test_path = Path("../results/t12/")
    dim = 2
    rank_dim = 1
    mesh_size = 5e-3

    vertices = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.6, 0.0],
            [1.0, 0.0],
            [1.0, 0.4],
            [1.0, 0.6],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [5], 2

    boundary_parts = [(dir1_idx, dir1_mkr, "dir1"), (dir2_idx, dir2_mkr, "dir2")]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir1_mkr, dir2_mkr], rank_dim
    )

    area = 0.4
    # Create the model
    md = Heat(dim, domain, space, dirichlet_bcs, area, test_path)

    @fop.region_of(domain)
    def sub_domain1(x):
        # 0.3 < x < 0.7
        # y < 0.05
        ineqs = (x[0] - 0.3, 0.7 - x[0], 0.05 - x[1])
        return ineqs

    @fop.region_of(domain)
    def sub_domain2(x):
        # 0.95 < x
        # 0.3 < y < 0.7
        ineqs = (x[1] - 0.3, 0.7 - x[1], x[0] - 0.95)
        return ineqs

    md.sub = [sub_domain1.expression(), sub_domain2.expression()]

    centers_b = [
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.7, 0.0),
        (0.8, 0.0),
        (0.9, 0.0),
        (1.0, 0.0),
        (1.0, 0.1),
        (1.0, 0.2),
        (1.0, 0.3),
        (1.0, 0.7),
        (1.0, 0.8),
        (1.0, 0.9),
    ]

    centers_b += [(0.0, i * 0.1) for i in range(11)]
    centers_b += [(i * 0.1, 1.0) for i in range(1, 11)]
    centers_b += [(i * 0.1, 1.0) for i in range(1, 11)]

    centers_i = [(0.1, i * 0.2 + 0.3) for i in range(4)]
    centers_i += [(0.2, i * 0.2 + 0.2) for i in range(4)]
    centers_i += [(0.3, i * 0.2 + 0.3) for i in range(4)]
    centers_i += [(0.4, i * 0.2 + 0.2) for i in range(4)]
    centers_i += [(0.5, i * 0.2 + 0.3) for i in range(4)]
    centers_i += [(0.6, i * 0.2 + 0.2) for i in range(4)]
    centers_i += [(0.7, i * 0.2 + 0.3) for i in range(4)]
    centers_i += [(0.8, i * 0.2 + 0.2) for i in range(4)]

    centers = np.array(centers_b + centers_i)
    radii_b = np.repeat(0.08, len(centers_b))
    radii_i = np.repeat(0.05, len(centers_i))
    radii = np.concatenate((radii_b, radii_i))

    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(niter=250, ctrn_tol=1e-3, lgrn_tol=5e-2)


def test_13():
    """
    Heat conduction with two sinks (multiple) - Task Parallelism

    Run `mpirun -np <nbr of processes> python test.py 13`.
    For instance, `mpirun -np 2 python test.py 13`.

    To save the output, append `> ../results/t13/out.txt`.
    To delete the images, enter `rm ../results/t13/*.png`.
    """

    task_nbr = 2
    if size != task_nbr:
        print(f"Number of processes must be = {task_nbr}")
        return

    comm_self = MPI.COMM_SELF

    test_name = "Heat conduction with two sinks (multiple) - Task Parallelism"
    test_path = Path("../results/t13/")
    dim = 2
    rank_dim = 1
    mesh_size = 5e-3

    vertices = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.6, 0.0],
            [1.0, 0.0],
            [1.0, 0.4],
            [1.0, 0.6],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [5], 2

    boundary_parts = [(dir1_idx, dir1_mkr, "dir1"), (dir2_idx, dir2_mkr, "dir2")]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_TP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir1_mkr, dir2_mkr], rank_dim
    )

    dir_bcs = [[dirichlet_bcs[0]], [dirichlet_bcs[1]]]

    area = 0.4
    # Create the model
    md = HeatPlus(dim, domain, space, dir_bcs, area, test_path)

    @fop.region_of(domain)
    def sub_domain1(x):
        # 0.3 < x < 0.7
        # y < 0.05
        ineqs = (x[0] - 0.3, 0.7 - x[0], 0.05 - x[1])
        return ineqs

    @fop.region_of(domain)
    def sub_domain2(x):
        # 0.95 < x
        # 0.3 < y < 0.7
        ineqs = (x[1] - 0.3, 0.7 - x[1], x[0] - 0.95)
        return ineqs

    md.sub = [sub_domain1.expression(), sub_domain2.expression()]

    md.wt = [0.5, 0.5]

    centers_b = [
        (0.1, 0.0),
        (0.2, 0.0),
        (0.3, 0.0),
        (0.7, 0.0),
        (0.8, 0.0),
        (0.9, 0.0),
        (1.0, 0.0),
        (1.0, 0.1),
        (1.0, 0.2),
        (1.0, 0.3),
        (1.0, 0.7),
        (1.0, 0.8),
        (1.0, 0.9),
    ]

    centers_b += [(0.0, i * 0.1) for i in range(11)]
    centers_b += [(i * 0.1, 1.0) for i in range(1, 11)]
    centers_b += [(i * 0.1, 1.0) for i in range(1, 11)]

    centers_i = [(0.1, i * 0.2 + 0.3) for i in range(4)]
    centers_i += [(0.2, i * 0.2 + 0.2) for i in range(4)]
    centers_i += [(0.3, i * 0.2 + 0.3) for i in range(4)]
    centers_i += [(0.4, i * 0.2 + 0.2) for i in range(4)]
    centers_i += [(0.5, i * 0.2 + 0.3) for i in range(4)]
    centers_i += [(0.6, i * 0.2 + 0.2) for i in range(4)]
    centers_i += [(0.7, i * 0.2 + 0.3) for i in range(4)]
    centers_i += [(0.8, i * 0.2 + 0.2) for i in range(4)]

    centers = np.array(centers_b + centers_i)
    radii_b = np.repeat(0.08, len(centers_b))
    radii_i = np.repeat(0.05, len(centers_i))
    radii = np.concatenate((radii_b, radii_i))

    md.create_initial_level(centers, radii)
    if rank == 0:
        md.save_initial_level(comm_self)

    md.runTP(niter=250, ctrn_tol=1e-3, lgrn_tol=1e-2)


def test_14(test_path=Path("../results/t14/"), r=10.0):
    """
    Logistic equation (r = 10) - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 14`.
    For instance, `mpirun -np 2 python test.py 14`.

    To save the output, append `> ../results/t14/out.txt`.
    To delete the images, enter `rm ../results/t14/*.png`.
    """

    dim = 2
    rank_dim = 1
    mesh_size = 0.012
    vertices = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    output = fop.create_domain_2d_DP(
        vertices, [], mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)
    vol = 0.5
    # Initial guess for Newton
    u0 = lambda x: (1 + 0.2 * np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1]))
    # Create the model
    md = Logistic(dim, domain, space, vol, r, u0, test_path)

    # Initial guess: centers and radii
    centers = []
    centers += [(0, 0.25 * i) for i in range(5)]
    centers += [(0.25, 0.25 * i) for i in range(5)]
    centers += [(0.5, 0.25 * i) for i in range(5)]
    centers += [(0.75, 0.25 * i) for i in range(5)]
    centers += [(1, 0.25 * i) for i in range(5)]
    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])
    md.create_initial_level(centers, radii, factor=-1.0)
    md.save_initial_level(comm)

    md.runDP(
        niter=150,
        lv_iter=(10, 16),
        lv_time=(1e-3, 1.0),
        reinit_step=6,
        reinit_pars=(20, 0.1),
        ctrn_tol=1e-3,
        lgrn_tol=1e-3,
        dfactor=1.0,
        smooth=True,
    )


def test_15():
    """
    Logistic equation (r = 40) - Data Parallelism
    """

    test_path = Path("../results/t15/")
    test_14(test_path, r=40)


def test_16():
    """
    Logistic equation (r = 100) - Data Parallelism
    """

    test_path = Path("../results/t16/")
    test_14(test_path, r=100)


def test_17(test_path=Path("../results/t17/"), vol=0.3):

    dim = 2
    rank_dim = 1
    mesh_size = 0.012

    vertices = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    output = fop.create_domain_2d_DP(
        vertices, [], mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)

    # Create the model
    md = Logistic(dim, domain, space, test_path)

    md.ini_func = lambda x: (
        10 + 0.2 * np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1])
    )

    md.d = 0.01
    md.vol = vol
    md.name = "R"
    md.args = (1.0, 0.001)

    # Initial guess: centers and radii
    centers = []

    centers += [(0, 0.25 * i) for i in range(5)]
    # centers += [(0.25, 0.25*i+0.125) for i in range(4)]
    # centers += [(0.5, 0.25*i) for i in range(5)]
    # centers += [(0.75, 0.25*i+0.125) for i in range(4)]
    centers += [(1, 0.25 * i) for i in range(5)]

    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])

    md.create_initial_level(centers, radii, factor=-1.0)
    md.save_initial_level(comm)

    md.runDP(
        niter=300,
        lv_iter=(10, 16),
        lv_time=(1e-3, 0.1),
        reinit_step=6,
        reinit_pars=(20, 0.1),
        ctrn_tol=1e-3,
        lgrn_tol=1e-3,
        dfactor=100.0,
        smooth=True,
    )


def test_18():
    """
    Run: mpirun -np 2 python test.py 18
    """

    test_name = "Cantilever with two loads II - Data Parallelism"
    test_path = Path("../results/t18/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.0095

    vertices = np.array(
        [
            [0.0, 0.0],
            [0.95, 0.0],
            [1.05, 0.0],
            [2.0, 0.0],
            [2.0, 0.45],
            [2.0, 0.55],
            [2.0, 1.0],
            [0.0, 1.0],
        ]
    )

    dir_idx, dir_mkr = [8], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_right, neu_mkr_right = [5], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_right, neu_mkr_right, "neu_right"),
    ]

    # Create gmsh domain for Task Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr_bot, neu_mkr_right])
    area = 1.1
    g = [(0.0, -2.0), (0.0, 2.0)]
    # Create the model
    md = CompliancePlus(dim, domain, space, g, ds_g, dirichlet_bcs, area, test_path)

    @fop.region_of(domain)
    def sub_domain_right(x):
        # 0.42 < x[1] < 0.58
        # 1.95 < x[0]
        ineqs = [x[1] - 0.42, 0.58 - x[1], x[0] - 1.95]
        return ineqs

    @fop.region_of(domain)
    def sub_domain_bottom(x):
        # 0.94 < x[0] < 1.06
        # x[1] < 0.05
        ineqs = [x[0] - 0.94, 1.06 - x[0], 0.05 - x[1]]
        return ineqs

    md.sub = [sub_domain_right.expression(), sub_domain_bottom.expression()]

    # Initial guess: centers and radii
    centers = []

    centers += [(2.0, 0.0), (2.0, 0.35), (2.0, 0.65), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * 0.35, 0.0) for i in range(5) if i not in (2,)]
    centers += [(0.3 + i * 0.35, 0.5) for i in range(5)]
    centers += [(0.3 + i * 0.35, 1.0) for i in range(5)]
    centers += [(0.475 + i * 0.35, 0.25) for i in range(4)]
    centers += [(0.475 + i * 0.35, 0.75) for i in range(4)]
    centers += [(1.15, 0.0), (1.2, 0.0)]
    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])

    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    # Run Task Parallelism
    md.runDP(
        ctrn_tol=1e-3,
        lgrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(20, 0.01),
        smooth=True,
    )


def test_19():
    """
    Run: mpirun -np 2 python test.py 19
    """

    # Verification
    task_nbr = 2
    if size != task_nbr:
        print(f"Nbr of processes must be = {task_nbr}")
        return

    comm_self = MPI.COMM_SELF

    test_name = "Cantilever with two loads II (Task Parallelism)"
    test_path = Path("../results/t19/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.0095

    vertices = np.array(
        [
            [0.0, 0.0],
            [0.95, 0.0],
            [1.05, 0.0],
            [2.0, 0.0],
            [2.0, 0.45],
            [2.0, 0.55],
            [2.0, 1.0],
            [0.0, 1.0],
        ]
    )

    dir_idx, dir_mkr = [8], 1
    neu_idx_bot, neu_mkr_bot = [2], 2
    neu_idx_right, neu_mkr_right = [5], 3

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idx_bot, neu_mkr_bot, "neu_bot"),
        (neu_idx_right, neu_mkr_right, "neu_right"),
    ]

    # Create gmsh domain for Task Parallelism
    output = fop.create_domain_2d_TP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr_bot, neu_mkr_right])
    area = 1.1
    g = [(0.0, -2.0), (0.0, 2.0)]
    # Create the model
    md = CompliancePlus(dim, domain, space, g, ds_g, dirichlet_bcs, area, test_path)

    @fop.region_of(domain)
    def sub_domain_right(x):
        # 0.42 < x[1] < 0.58
        # 1.95 < x[0]
        ineqs = [x[1] - 0.42, 0.58 - x[1], x[0] - 1.95]
        return ineqs

    @fop.region_of(domain)
    def sub_domain_bottom(x):
        # 0.94 < x[0] < 1.06
        # x[1] < 0.05
        ineqs = [x[0] - 0.94, 1.06 - x[0], 0.05 - x[1]]
        return ineqs

    md.sub = [sub_domain_right.expression(), sub_domain_bottom.expression()]

    # Initial guess: centers and radii
    centers = []

    centers += [(2.0, 0.0), (2.0, 0.35), (2.0, 0.65), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * 0.35, 0.0) for i in range(5) if i not in (2,)]
    centers += [(0.3 + i * 0.35, 0.5) for i in range(5)]
    centers += [(0.3 + i * 0.35, 1.0) for i in range(5)]
    centers += [(0.475 + i * 0.35, 0.25) for i in range(4)]
    centers += [(0.475 + i * 0.35, 0.75) for i in range(4)]
    centers += [(1.15, 0.0), (1.2, 0.0)]
    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])

    md.create_initial_level(centers, radii)
    if rank == 0:
        md.save_initial_level(comm_self)

    # Run Task Parallelism
    md.runTP(
        ctrn_tol=1e-3,
        lgrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(20, 0.01),
        smooth=True,
    )


def test_20():
    """
    Symmetric Cantilever 2D (non-rectangular domain) - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 20`.
    For instance, `mpirun -np 2 python test.py 20`.

    To save the output, append `> ../results/t20/out.txt`.
    To delete the images, enter `rm ../results/t20/*.png`.
    """

    test_name = "Symmetric Cantilever 2D (non-rectangular domain) - Data Parallelism"
    test_path = Path("../results/t20/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    vertices = np.array(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.45), (2.0, 0.55), (1.0, 1.0), (0.0, 1.0)]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [(dir_idx, dir_mkr, "dir"), (neu_idx, neu_mkr, "neu")]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)

    # Dirichlet condition
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )

    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])

    area = 1.0
    g = (0.0, -2.0)
    # Create the model
    md = Compliance(dim, domain, space, g, ds_g[0], dirichlet_bcs, area, test_path)

    # Initial guess: centers and radii
    centers = [(0.0, 0.5)]
    centers += [((1 + i) * 0.25, 0.0) for i in range(4)]
    centers += [(i * 0.5, 0.25) for i in range(4)]
    centers += [(0.25 + i * 0.5, 0.5) for i in range(4)]
    centers += [(i * 0.5, 0.75) for i in range(4)]
    centers += [((1 + i) * 0.25, 1.0) for i in range(4)]

    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])

    # Create the initial level set function
    md.create_initial_level(centers, radii)
    # Save as initial.xdmf
    md.save_initial_level(comm)

    md.runDP(
        ctrn_tol=1e-3,
        dfactor=1e-1,
        lv_iter=(10, 16),
        reinit_step=8,
        reinit_pars=(20, 0.08),
        smooth=True,
        random_pars=(111, 0.05),
    )


def test_21():
    """
    Heat conduction with four sources (single) - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 21`.
    For instance, `mpirun -np 2 python test.py 21`.

    To save the output, append `> ../results/t21/out.txt`.
    To delete the images, enter `rm ../results/t21/*.png`.

    Description:
    This test considers four localized heat sources,
    represented as radial functions with compact support,
    and four sinks at the domain corners, imposed through
    homogeneous Dirichlet boundary conditions.

    Execution times:
    - 1 process (80 iterations):
        > Assembly time = 5.040334394000183 s
        > Resolution time = 26.744602871000097 s
    - 2 processes (80 iterations):
        > Assembly time = 7.185058718999926 s
        > Resolution time = 14.725885109999808 s
    - 3 processes (80 iterations):
        > Assembly time = 15.85057142100004 s
        > Resolution time = 10.170385355000008 s
    - 4 processes (80 iterations):
        > Assembly time = 7.924901005000038 s
        > Resolution time = 8.05341802199996 s
    - 5 processes (80 iterations):
        > Assembly time = 7.775461058000019 s
        > Resolution time = 6.686465177999935 s
    - 6 processes (80 iterations):
        > Assembly time = 16.48660050400008 s
        > Resolution time = 5.905231507000053 s
    - 7 processes (80 iterations):
        > Assembly time = 19.62543761699999 s
        > Resolution time = 10.18865386199991 s
    - 8 processes (80 iterations):
        > Assembly time = 20.597811266000008 s
        > Resolution time = 9.384790290999717 s
    - 9 processes (80 iterations):
        > Assembly time = 11.627410267999949 s
        > Resolution time = 8.650962873000026 s
    - 10 processes (80 iterations):
        > Assembly time = 11.565947376000167 s
        > Resolution time = 7.873313986000085 s

    resolution_times = [
        26.744602871000097,
        14.725885109999808,
        10.170385355000008,
        8.05341802199996,
        6.686465177999935,
        5.905231507000053,
        10.18865386199991,
        9.384790290999717,
        8.650962873000026,
        7.873313986000085
    ]
    """

    test_name = "Heat conduction with four sources (single) - Data Parallelism"
    test_path = Path("../results/t21/")
    dim = 2
    rank_dim = 1
    mesh_size = 1e-2

    vertices = np.array(
        [
            [0.05, 0.0],
            [0.95, 0.0],
            [1.0, 0.05],
            [1.0, 0.95],
            [0.95, 1.0],
            [0.05, 1.0],
            [0.0, 0.95],
            [0.0, 0.05],
        ]
    )

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [4], 2
    dir3_idx, dir3_mkr = [6], 3
    dir4_idx, dir4_mkr = [8], 4

    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2"),
        (dir3_idx, dir3_mkr, "dir3"),
        (dir4_idx, dir4_mkr, "dir4"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir1_mkr, dir2_mkr, dir3_mkr, dir4_mkr], rank_dim
    )

    area = 0.45
    # Create the model
    md = Heat(dim, domain, space, dirichlet_bcs, area, test_path, "4Loads")
    centers = []

    # Diagonal balls
    centers += [(0.2 + i * 0.12, 0.2 + i * 0.12) for i in range(6)]
    centers += [(0.8 - i * 0.12, 0.2 + i * 0.12) for i in range(6)]

    # Boundary balls
    centers += [(0.1 + i * 0.1, 0.0) for i in range(9)]
    centers += [(0.1 + i * 0.1, 1.0) for i in range(9)]
    centers += [(0.0, 0.1 + i * 0.1) for i in range(9)]
    centers += [(1.0, 0.1 + i * 0.1) for i in range(9)]

    centers += [(0.2, 0.32), (0.32, 0.2)]
    centers += [(0.8, 0.32), (0.68, 0.2)]
    centers += [(0.8, 0.68), (0.68, 0.8)]
    centers += [(0.2, 0.68), (0.32, 0.8)]

    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])

    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(
        dfactor=1.0, ctrn_tol=1e-3, smooth=True, reinit_step=4, reinit_pars=(12, 0.01)
    )


def test_22():
    """
    Heat conduction with four sources (multiple) - Task Parallelism

    Run `mpirun -np 4 python test.py 22`.

    To save the output, append `> ../results/t22/out.txt`.
    To delete the images, enter `rm ../results/t22/*.png`.

    Description:
    This test considers four problems, each with one
    localized heat source represented as a radial function with compact support.
    In all problems, four sinks are located at the domain corners and imposed
    through homogeneous Dirichlet boundary conditions.
    """

    task_nbr = 4
    if size != task_nbr:
        print(f"Number of processes must be = {task_nbr}")
        return

    comm_self = MPI.COMM_SELF

    test_name = "Heat conduction with four sources (multiple) - Task Parallelism"
    test_path = Path("../results/t22/")
    dim = 2
    rank_dim = 1
    mesh_size = 1e-2

    vertices = np.array(
        [
            [0.05, 0.0],
            [0.95, 0.0],
            [1.0, 0.05],
            [1.0, 0.95],
            [0.95, 1.0],
            [0.05, 1.0],
            [0.0, 0.95],
            [0.0, 0.05],
        ]
    )

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [4], 2
    dir3_idx, dir3_mkr = [6], 3
    dir4_idx, dir4_mkr = [8], 4

    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2"),
        (dir3_idx, dir3_mkr, "dir3"),
        (dir4_idx, dir4_mkr, "dir4"),
    ]

    # Create gmsh domain for Task Parallelism
    output = fop.create_domain_2d_TP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir1_mkr, dir2_mkr, dir3_mkr, dir4_mkr], rank_dim
    )
    dir_bcs = 4 * [dirichlet_bcs]
    area = 0.45
    # Create the model
    md = HeatPlus(dim, domain, space, dir_bcs, area, test_path)

    md.f = [
        md.source(0.5, 0.25, max_value=50.0, epsilon=0.1),
        md.source(0.75, 0.5, max_value=50.0, epsilon=0.1),
        md.source(0.5, 0.75, max_value=50.0, epsilon=0.1),
        md.source(0.25, 0.5, max_value=50.0, epsilon=0.1),
    ]
    md.wt = [0.25, 0.25, 0.25, 0.25]
    centers = []

    @fop.region_of(domain)
    def sub_domain1(x):
        return [0.12**2 - (x[0] - 0.5) ** 2 - (x[1] - 0.25) ** 2]

    @fop.region_of(domain)
    def sub_domain2(x):
        return [0.12**2 - (x[0] - 0.75) ** 2 - (x[1] - 0.5) ** 2]

    @fop.region_of(domain)
    def sub_domain3(x):
        return [0.12**2 - (x[0] - 0.5) ** 2 - (x[1] - 0.75) ** 2]

    @fop.region_of(domain)
    def sub_domain4(x):
        return [0.12**2 - (x[0] - 0.25) ** 2 - (x[1] - 0.5) ** 2]

    md.sub = [
        sub_domain1.expression(),
        sub_domain2.expression(),
        sub_domain3.expression(),
        sub_domain4.expression(),
    ]

    # Diagonal balls
    centers += [(0.2 + i * 0.12, 0.2 + i * 0.12) for i in range(6)]
    centers += [(0.8 - i * 0.12, 0.2 + i * 0.12) for i in range(6)]

    # Boundary balls
    centers += [(0.1 + i * 0.1, 0.0) for i in range(9)]
    centers += [(0.1 + i * 0.1, 1.0) for i in range(9)]
    centers += [(0.0, 0.1 + i * 0.1) for i in range(9)]
    centers += [(1.0, 0.1 + i * 0.1) for i in range(9)]

    centers += [(0.2, 0.32), (0.32, 0.2)]
    centers += [(0.8, 0.32), (0.68, 0.2)]
    centers += [(0.8, 0.68), (0.68, 0.8)]
    centers += [(0.2, 0.68), (0.32, 0.8)]

    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])

    md.create_initial_level(centers, radii)
    if rank == 0:
        md.save_initial_level(comm_self)

    md.runTP(
        dfactor=0.1, ctrn_tol=1e-3, smooth=True, reinit_step=4, reinit_pars=(12, 0.01)
    )


def test_23():
    """
    Heat conduction with four sources (multiple) - Mixed Parallelism

    Run `mpirun -np <4n> python test.py 23`.
    For instance, `mpirun -np 4 python test.py 23`.

    Execution times:
    - 4 processes (63 iterations):
        > Assembly time = 4.883218779999993 s
        > Resolution time = 19.407882885999996 s
    - 8 processes (63 iterations):
    """

    # Verification
    nbr_groups = 4
    if size % nbr_groups != 0:
        print(f"Nbr of processes must be divisible by {nbr_groups}")
        return

    # Subcommunicators
    color = rank * nbr_groups // size
    sub_comm = comm.Split(color, rank)

    test_name = "Heat conduction with four sources (multiple) - Mixed Parallelism"
    test_path = Path("../results/t23/")
    dim = 2
    rank_dim = 1
    mesh_size = 0.005

    vertices = np.array(
        [
            [0.05, 0.0],
            [0.95, 0.0],
            [1.0, 0.05],
            [1.0, 0.95],
            [0.95, 1.0],
            [0.05, 1.0],
            [0.0, 0.95],
            [0.0, 0.05],
        ]
    )

    dir1_idx, dir1_mkr = [2], 1
    dir2_idx, dir2_mkr = [4], 2
    dir3_idx, dir3_mkr = [6], 3
    dir4_idx, dir4_mkr = [8], 4

    boundary_parts = [
        (dir1_idx, dir1_mkr, "dir1"),
        (dir2_idx, dir2_mkr, "dir2"),
        (dir3_idx, dir3_mkr, "dir3"),
        (dir4_idx, dir4_mkr, "dir4"),
    ]

    output = fop.create_domain_2d_MP(
        sub_comm, color, vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet conditions
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir1_mkr, dir2_mkr, dir3_mkr, dir4_mkr], rank_dim
    )
    dir_bcs = 4 * [dirichlet_bcs]
    area = 0.45
    # Create the model
    md = HeatPlus(dim, domain, space, dir_bcs, area, test_path)

    md.f = [
        md.source(0.5, 0.25, max_value=50.0, epsilon=0.1),
        md.source(0.75, 0.5, max_value=50.0, epsilon=0.1),
        md.source(0.5, 0.75, max_value=50.0, epsilon=0.1),
        md.source(0.25, 0.5, max_value=50.0, epsilon=0.1),
    ]
    md.wt = [0.25, 0.25, 0.25, 0.25]
    centers = []

    @fop.region_of(domain)
    def sub_domain1(x):
        return [0.12**2 - (x[0] - 0.5) ** 2 - (x[1] - 0.25) ** 2]

    @fop.region_of(domain)
    def sub_domain2(x):
        return [0.12**2 - (x[0] - 0.75) ** 2 - (x[1] - 0.5) ** 2]

    @fop.region_of(domain)
    def sub_domain3(x):
        return [0.12**2 - (x[0] - 0.5) ** 2 - (x[1] - 0.75) ** 2]

    @fop.region_of(domain)
    def sub_domain4(x):
        return [0.12**2 - (x[0] - 0.25) ** 2 - (x[1] - 0.5) ** 2]

    md.sub = [
        sub_domain1.expression(),
        sub_domain2.expression(),
        sub_domain3.expression(),
        sub_domain4.expression(),
    ]

    # Diagonal balls
    yy = 0.2
    h = (1.0 - 2.0 * yy) / 5.0
    centers += [(yy + i * h, yy + i * h) for i in range(6)]
    centers += [((1 - yy) - i * h, yy + i * h) for i in range(6)]

    # Boundary balls
    centers += [(0.1 + i * 0.1, 0.0) for i in range(9)]
    centers += [(0.1 + i * 0.1, 1.0) for i in range(9)]
    centers += [(0.0, 0.1 + i * 0.1) for i in range(9)]
    centers += [(1.0, 0.1 + i * 0.1) for i in range(9)]

    centers += [(0.18, 0.34), (0.34, 0.18)]
    centers += [(0.82, 0.34), (0.66, 0.18)]
    centers += [(0.82, 0.66), (0.66, 0.82)]
    centers += [(0.18, 0.66), (0.34, 0.82)]
    centers += [(0.5, 0.5)]

    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])

    md.create_initial_level(centers, radii)
    if rank == 0:
        md.save_initial_level(sub_comm)

    md.runMP(
        niter=70,
        sub_comm=sub_comm,
        dfactor=0.1,
        lv_iter=(8, 25),
        lv_time=(0.001, 0.05),
        ctrn_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(20, 0.01),
    )


def test_31():
    """
    Mechanism 2D - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 31`.
    For instance, `mpirun -np 2 python test.py 31`.
    """

    test_name = "Mechanism 2D - Data Parallelism"
    test_path = Path("../results/t31/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.005

    vertices = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.47),
            (1.0, 0.53),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.95),
            (0.0, 0.53),
            (0.0, 0.47),
            (0.0, 0.05),
        ]
    )

    dir_idx, dir_mkr = [6], 1
    dir_idx2, dir_mkr2 = [10], 2
    rob_idx, rob_mkr = [3], 3
    neu_idx, neu_mkr = [8], 4

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (dir_idx2, dir_mkr2, "dir"),
        (rob_idx, rob_mkr, "rob"),
        (neu_idx, neu_mkr, "neu"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)

    # Dirichlet condition
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr, dir_mkr2], rank_dim
    )

    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [rob_mkr, neu_mkr])

    area = 0.2
    g = (0.5, 0.0)
    # Create the model
    md = Mechanism(dim, domain, space, g, ds_g, dirichlet_bcs, area, test_path)

    @fop.region_of(domain)
    def sub_domain1(x):
        # 0.42 < x[1] < 0.58
        # 0.90 < x[0]
        ineqs = [x[1] - 0.46, 0.54 - x[1], x[0] - 0.90]
        return ineqs

    @fop.region_of(domain)
    def sub_domain2(x):
        # 0.42 < x[1] < 0.58
        # x[0] < 0.1
        ineqs = [x[1] - 0.46, 0.54 - x[1], 0.1 - x[0]]
        return ineqs

    # md.sub = [sub_domain1.expression(), sub_domain2.expression()]

    centers = [
        (1.0, 0.10),
        (1.0, 0.20),
        (1.0, 0.80),
        (1.0, 0.90),
    ]
    centers += [(0.8 + i * 0.1, 0.0) for i in range(3)]
    centers += [(i / 3.0, 0.25) for i in range(5)]
    centers += [(0.5, 0.5), (0.75, 0.5)]
    centers += [(i / 3.0, 0.75) for i in range(5)]
    centers += [(0.8 + i * 0.1, 1.0) for i in range(3)]

    centers = np.array(centers)
    radii = np.repeat(0.075, centers.shape[0])

    # Create the initial level set function
    md.create_initial_level(centers, radii)
    # Save as initial.xdmf
    md.save_initial_level(comm)

    md.runDP(
        niter=400,
        ctrn_tol=1e-3,
        dfactor=0.1,
        lv_time=(0.001, 0.05),
        lv_iter=(12, 25),
        reinit_step=4,
        reinit_pars=(20, 0.01),
        smooth=True,
    )


def test_32():
    """
    Data paralellism
    ----------------
    Performance of tests 01, 03, and 21:
        please run `python load.py 32`
    """


def test_33():
    """
    Gripping mechanism 2D - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 33`.
    For instance, `mpirun -np 2 python test.py 33`.

    Reference:
    A consistent relaxation of optimal design problems
    for coupling shape and topological derivatives
    Samuel Amstutz et al., 2018
    doi.org/10.1007/s00211-018-0964-4
    """

    test_name = "Gripping mechanism 2D - Data Parallelism"
    test_path = Path("../results/t33/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.01

    vertices = np.array(
        [
            (0.0, 0.0),
            (0.9, 0.0),
            (1.0, 0.0),
            (1.0, 0.48),
            (1.0, 0.52),
            (1.0, 1.0),
            (0.9, 1.0),
            (0.0, 1.0),
            (0.0, 0.6),
            (0.1, 0.6),
            (0.1, 0.52),
            (0.1, 0.48),
            (0.1, 0.4),
            (0.0, 0.4),
        ]
    )

    dirR_idx, dirR_mkr = [4], 1
    dirL_idx, dirL_mkr = [11], 2
    neuRT_idx, neuRT_mkr = [6], 3
    neuRB_idx, neuRB_mkr = [2], 4
    neuLT_idx, neuLT_mkr = [9], 5
    neuLB_idx, neuLB_mkr = [13], 6

    boundary_parts = [
        (dirR_idx, dirR_mkr, "dir_right"),
        (dirL_idx, dirL_mkr, "dir_left"),
        (neuRT_idx, neuRT_mkr, "neu_right_top"),
        (neuRB_idx, neuRB_mkr, "neu_right_bottom"),
        (neuLT_idx, neuLT_mkr, "neu_left_top"),
        (neuLB_idx, neuLB_mkr, "neu_left_bottom"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)

    # Dirichlet condition
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dirR_mkr, dirL_mkr], rank_dim
    )

    # Boundary to force application
    ds_g = fop.marked_ds(
        domain, boundary_tags, [neuRT_mkr, neuRB_mkr, neuLT_mkr, neuLB_mkr]
    )

    bc_theta = (boundary_tags, [neuRT_mkr, neuRB_mkr, neuLT_mkr, neuLB_mkr])

    ff = 1.0
    gg = 1.0
    area = 0.3
    g = [(0.0, -ff * 10.0), (0.0, ff * 10.0), (0.0, ff * 1.0), (0.0, -ff * 1.0)]
    k = [(0.0, -gg * 1.0), (0.0, gg * 1.0), (0.0, gg * 2.0), (0.0, -gg * 2.0)]
    # Create the model
    md = Gripping(
        dim, domain, space, g, ds_g, k, dirichlet_bcs, bc_theta, area, test_path
    )

    @fop.region_of(domain)
    def sub_domain1(x):
        # 0.9 < x[0] < 1.0
        # 0.95 < x[1]
        ineqs = [x[0] - 0.9, 1.0 - x[0], x[1] - 0.95]
        return ineqs

    @fop.region_of(domain)
    def sub_domain2(x):
        # 0.9 < x[0] < 1.0
        # x[1] < 0.05
        ineqs = [x[0] - 0.9, 1.0 - x[0], 0.05 - x[1]]
        return ineqs

    @fop.region_of(domain)
    def sub_domain3(x):
        # 0.0 < x[0] < 0.1
        # 0.35 < x[1] < 0.5
        ineqs = [x[0], 0.1 - x[0], x[1] - 0.35, 0.5 - x[1]]
        return ineqs

    @fop.region_of(domain)
    def sub_domain4(x):
        # 0.0 < x[0] < 0.1
        # 0.65 > x[1] > 0.5
        ineqs = [x[0], 0.1 - x[0], 0.65 - x[1], x[1] - 0.5]
        return ineqs

    md.sub = [
        sub_domain1.expression(),
        sub_domain2.expression(),
        sub_domain3.expression(),
        sub_domain4.expression(),
    ]

    centers = []
    centers += [(i * 0.5 / 7, 1.0) for i in range(8)]
    centers += [(i * 0.5 / 7, 0.0) for i in range(8)]
    centers += [(1.0, 0.2 + i * 0.6 / 9) for i in range(10)]
    centers += [(0.0, i * 0.2 / 3) for i in range(4)]
    centers += [(0.0, 1.0 - i * 0.2 / 3) for i in range(4)]
    centers += [(0.1, 0.5), (0.1, 0.55), (0.1, 0.45), (0.5, 0.5)]
    centers += [(0.0, 0.0), (0.0, 1.0), (1.0, 0.5)]

    centers = np.array(centers)
    radii = np.repeat(0.05, centers.shape[0])
    radii[-4:] = 0.1
    md.create_initial_level(centers, radii)
    md.save_initial_level(comm)

    md.runDP(
        niter=600,
        ctrn_tol=1e-3,
        dfactor=0.01,
        lv_iter=(15, 25),
        lv_time=(0.01, 1.0),
        reinit_step=4,
        reinit_pars=(20, 0.001),
        smooth=True,
    )


def test_34():
    """
    Elasticity Inverse Problem (two inclusions) - Data Parallelism

    Run `mpirun -np <nbr of processes> python test.py 34`.
    For instance, `mpirun -np 16 python test.py 34`.
    """

    test_name = "Elasticity Inverse Problem (two inclusions) - Data Parallelism"
    test_path = Path("../results/t34/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps) / b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a * np.cos(t)
        y = b * np.sin(t) + (b - eps)
        return x, y

    npts = 80  # npts % 4 = 0
    part = npts // 8

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.15, npts))

    # 1 dirichlet boundary
    # 8 neumann boundaries
    dir_idx, dir_mkr = [npts], 1
    neu_idxA, neu_mkrA = np.arange(1, part + 1), 2
    neu_idxB, neu_mkrB = np.arange(part + 1, 2 * part + 1), 3
    neu_idxC, neu_mkrC = np.arange(2 * part + 1, 3 * part + 1), 4
    neu_idxD, neu_mkrD = np.arange(3 * part + 1, 4 * part + 1), 5
    neu_idxE, neu_mkrE = np.arange(4 * part + 1, 5 * part + 1), 6
    neu_idxF, neu_mkrF = np.arange(5 * part + 1, 6 * part + 1), 7
    neu_idxG, neu_mkrG = np.arange(6 * part + 1, 7 * part + 1), 8
    neu_idxH, neu_mkrH = np.arange(7 * part + 1, npts), 9

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (neu_idxD, neu_mkrD, "neuD"),
        (neu_idxE, neu_mkrE, "neuE"),
        (neu_idxF, neu_mkrF, "neuF"),
        (neu_idxG, neu_mkrG, "neuG"),
        (neu_idxH, neu_mkrH, "neuH"),
    ]

    def SubDomain1(n):
        # Lifted circle
        t = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        x0, y0 = -0.3, 0.3  # translation
        ef = 0.22  # scaled factor
        x = ef * np.cos(t) + x0
        y = ef * (np.sin(t) + (np.sin(t)) ** 2 / 2.0) + y0
        return x, y

    def SubDomain2(n):
        # Rotated ellipse
        t = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        angle = np.pi / 6.0
        x0, y0 = 0.5, 0.2
        a, b = 0.12, 0.28
        x = a * np.cos(t) + x0
        y = b * np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_ * x - sin_ * y
        y_rot = sin_ * x + cos_ * y
        return x_rot, y_rot

    subdomain1 = np.column_stack(SubDomain1(60))

    subdomain2 = np.column_stack(SubDomain2(60))

    if rank == 0:
        np.save(test_path / "inclusion1.npy", subdomain1)
        np.save(test_path / "inclusion2.npy", subdomain2)

    filename = test_path / "domain0.msh"

    # Create the gmsh domain0.msh
    nbr_tri0 = fop.build_gmsh_model_2d(
        vertices,
        boundary_parts,
        0.6 * mesh_size,
        curves=[subdomain1, subdomain2],
        filename=filename,
        plot=False,
    )

    # Read the domain0
    output = fop.read_gmsh(filename, comm, dim=2)

    domain0, _, boundary_tags = output

    # Set all connectivities on domain0
    fop.all_connectivities(domain0)

    # Space for data generation
    space0 = fop.create_space(domain0, "CG", rank_dim)
    # Forces
    forces = [
        (np.cos(np.pi / 16), np.sin(np.pi / 16)),
        (np.cos(np.pi / 16 + np.pi / 8), np.sin(np.pi / 16 + np.pi / 8)),
        (np.cos(np.pi / 16 + 2 * np.pi / 8), np.sin(np.pi / 16 + 2 * np.pi / 8)),
        (np.cos(np.pi / 16 + 3 * np.pi / 8), np.sin(np.pi / 16 + 3 * np.pi / 8)),
        (np.cos(np.pi / 16 + 4 * np.pi / 8), np.sin(np.pi / 16 + 4 * np.pi / 8)),
        (np.cos(np.pi / 16 + 5 * np.pi / 8), np.sin(np.pi / 16 + 5 * np.pi / 8)),
        (np.cos(np.pi / 16 + 6 * np.pi / 8), np.sin(np.pi / 16 + 6 * np.pi / 8)),
        (np.cos(np.pi / 16 + 7 * np.pi / 8), np.sin(np.pi / 16 + 7 * np.pi / 8)),
    ]

    # Dirichlet boundary conditions
    dirbc_partial = fop.homogeneous_dirichlet(
        domain0, space0, boundary_tags, [dir_mkr], rank_dim
    )

    dirbc_total = fop.homogeneous_boundary(domain0, space0, dim, rank_dim)

    # Create measures to apply Neumman condition
    ds_parts = fop.marked_ds(
        domain0,
        boundary_tags,
        [
            neu_mkrA,
            neu_mkrB,
            neu_mkrC,
            neu_mkrD,
            neu_mkrE,
            neu_mkrF,
            neu_mkrG,
            neu_mkrH,
        ],
    )

    # Measures for force application
    ds_forces = ds_parts

    # Measure for adjoint problem
    ds1 = sum(ds_parts[1:], start=ds_parts[0])

    # Instance for data generation
    # We need the method pde0
    md0 = InverseElasticity(
        dim,
        domain0,
        space0,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        test_path,
    )

    # Function that defines
    # the level set function
    # to generate the data
    def SubDomain1_eq(x):
        """
        Implicit equation :
        (x^2/2 + y - 1/2)^2 = 1 - x^2
        """
        x0, y0 = -0.3, 0.3  # translation
        ef = 0.22  # scaled factor
        xx = (x[0] - x0) / ef
        yy = (x[1] - y0) / ef
        values = xx**4 + 4 * xx**2 * yy + 2 * xx**2 + 4 * yy**2 - 4 * yy - 3
        return np.log(0.1 * values + 1.0)

    def SubDomain2_eq(x):
        # inverse rotation
        angle = np.pi / 6.0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        xi = cos_ * x[0] + sin_ * x[1]
        yi = -sin_ * x[0] + cos_ * x[1]
        x0, y0 = 0.5, 0.2
        a, b = 0.12, 0.28
        xx = (xi - x0) / a
        yy = (yi - y0) / b
        values = xx**2 + yy**2 - 1
        return np.log(0.4 * values + 1.0)

    def SubDomain_combined(x):
        return np.minimum(SubDomain1_eq(x), SubDomain2_eq(x))

    # Dirichlet extensions
    extensions = fop.dir_extension_from(
        comm, domain0, space0, md0.pde0, SubDomain_combined, test_path
    )

    npts = 80  # npts % 4 = 0
    part = npts // 8

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.15, npts))

    # 1 dirichlet boundary
    # 8 neumann boundaries
    dir_idx, dir_mkr = [npts], 1
    neu_idxA, neu_mkrA = np.arange(1, part + 1), 2
    neu_idxB, neu_mkrB = np.arange(part + 1, 2 * part + 1), 3
    neu_idxC, neu_mkrC = np.arange(2 * part + 1, 3 * part + 1), 4
    neu_idxD, neu_mkrD = np.arange(3 * part + 1, 4 * part + 1), 5
    neu_idxE, neu_mkrE = np.arange(4 * part + 1, 5 * part + 1), 6
    neu_idxF, neu_mkrF = np.arange(5 * part + 1, 6 * part + 1), 7
    neu_idxG, neu_mkrG = np.arange(6 * part + 1, 7 * part + 1), 8
    neu_idxH, neu_mkrH = np.arange(7 * part + 1, npts), 9

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (neu_idxD, neu_mkrD, "neuD"),
        (neu_idxE, neu_mkrE, "neuE"),
        (neu_idxF, neu_mkrF, "neuF"),
        (neu_idxG, neu_mkrG, "neuG"),
        (neu_idxH, neu_mkrH, "neuH"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    dirbc_total = fop.homogeneous_boundary(domain, space, dim, rank_dim)
    # Boundary to apply Neumann conditions
    ds_parts = fop.marked_ds(
        domain,
        boundary_tags,
        [
            neu_mkrA,
            neu_mkrB,
            neu_mkrC,
            neu_mkrD,
            neu_mkrE,
            neu_mkrF,
            neu_mkrG,
            neu_mkrH,
        ],
    )

    ds_forces = ds_parts
    ds1 = sum(ds_parts[1:], start=ds_parts[0])
    forces = [
        (np.cos(np.pi / 16), np.sin(np.pi / 16)),
        (np.cos(np.pi / 16 + np.pi / 8), np.sin(np.pi / 16 + np.pi / 8)),
        (np.cos(np.pi / 16 + 2 * np.pi / 8), np.sin(np.pi / 16 + 2 * np.pi / 8)),
        (np.cos(np.pi / 16 + 3 * np.pi / 8), np.sin(np.pi / 16 + 3 * np.pi / 8)),
        (np.cos(np.pi / 16 + 4 * np.pi / 8), np.sin(np.pi / 16 + 4 * np.pi / 8)),
        (np.cos(np.pi / 16 + 5 * np.pi / 8), np.sin(np.pi / 16 + 5 * np.pi / 8)),
        (np.cos(np.pi / 16 + 6 * np.pi / 8), np.sin(np.pi / 16 + 6 * np.pi / 8)),
        (np.cos(np.pi / 16 + 7 * np.pi / 8), np.sin(np.pi / 16 + 7 * np.pi / 8)),
    ]

    # Create the model
    md = InverseElasticity(
        dim,
        domain,
        space,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        test_path,
    )

    # Space for interpolation (degree = 2)
    g_space = fop.create_space(domain, "CG", rank_dim, degree=2)
    # Interpolation between different spaces
    # from different domains
    g_funcs = fop.space_interpolation(
        from_space=space0, funcs=extensions, to_space=g_space
    )

    # To save as P1 functions
    g_space_1 = fop.create_space(domain, "CG", rank_dim)
    g_funcs_1 = fop.interpolate(funcs=g_funcs, to_space=g_space_1, name="g")
    fop.save_functions(comm, domain, g_funcs_1, test_path / "gP1.xdmf")

    md.gs = g_funcs

    # Initial guess: centers and radii
    centers = np.array([(-0.3, 0.4), (0.3, 0.4)])
    radii = np.array([0.15, 0.15])

    md.create_initial_level(centers, radii, factor=-1.0)
    md.save_initial_level(comm)

    # Run Data Parallelism
    md.runDP(niter=200, dfactor=1e-1, lv_time=(1e-3, 1.0), cost_tol=1e-1)


def test_35():
    """
    Elasticity Inverse Problem (two inclusions) - Task Parallelism

    Run `mpirun -np 16 python test.py 35`.
    """

    # Verification
    task_nbr = 16
    if size != task_nbr:
        print(f"Nbr of processes must be = {task_nbr}")
        return

    comm_self = MPI.COMM_SELF

    test_name = "Elasticity Inverse Problem (two inclusions) - Task Parallelism"
    test_path = Path("../results/t35/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps) / b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a * np.cos(t)
        y = b * np.sin(t) + (b - eps)
        return x, y

    npts = 80  # npts % 4 = 0
    part = npts // 8

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.15, npts))

    # 1 dirichlet boundary
    # 8 neumann boundaries
    dir_idx, dir_mkr = [npts], 1
    neu_idxA, neu_mkrA = np.arange(1, part + 1), 2
    neu_idxB, neu_mkrB = np.arange(part + 1, 2 * part + 1), 3
    neu_idxC, neu_mkrC = np.arange(2 * part + 1, 3 * part + 1), 4
    neu_idxD, neu_mkrD = np.arange(3 * part + 1, 4 * part + 1), 5
    neu_idxE, neu_mkrE = np.arange(4 * part + 1, 5 * part + 1), 6
    neu_idxF, neu_mkrF = np.arange(5 * part + 1, 6 * part + 1), 7
    neu_idxG, neu_mkrG = np.arange(6 * part + 1, 7 * part + 1), 8
    neu_idxH, neu_mkrH = np.arange(7 * part + 1, npts), 9

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (neu_idxD, neu_mkrD, "neuD"),
        (neu_idxE, neu_mkrE, "neuE"),
        (neu_idxF, neu_mkrF, "neuF"),
        (neu_idxG, neu_mkrG, "neuG"),
        (neu_idxH, neu_mkrH, "neuH"),
    ]

    def SubDomain1(n):
        # Lifted circle
        t = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        x0, y0 = -0.3, 0.3  # translation
        ef = 0.22  # scaled factor
        x = ef * np.cos(t) + x0
        y = ef * (np.sin(t) + (np.sin(t)) ** 2 / 2.0) + y0
        return x, y

    def SubDomain2(n):
        # Rotated ellipse
        t = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        angle = np.pi / 6.0
        x0, y0 = 0.5, 0.2
        a, b = 0.12, 0.28
        x = a * np.cos(t) + x0
        y = b * np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_ * x - sin_ * y
        y_rot = sin_ * x + cos_ * y
        return x_rot, y_rot

    subdomain1 = np.column_stack(SubDomain1(60))

    subdomain2 = np.column_stack(SubDomain2(60))

    if rank == 0:
        np.save(test_path / "inclusion1.npy", subdomain1)
        np.save(test_path / "inclusion2.npy", subdomain2)

    filename = test_path / "domain0.msh"

    # Create the gmsh domain0.msh
    nbr_tri0 = fop.build_gmsh_model_2d(
        vertices,
        boundary_parts,
        0.6 * mesh_size,
        curves=[subdomain1, subdomain2],
        filename=filename,
        plot=False,
    )

    if rank == 0:
        # Read the domain0
        output = fop.read_gmsh(filename, comm_self, dim=2)

        domain0, _, boundary_tags = output

        # Set all connectivities on domain0
        fop.all_connectivities(domain0)

        # Space for data generation
        space0 = fop.create_space(domain0, "CG", rank_dim)
        # Forces
        forces = [
            (np.cos(np.pi / 16), np.sin(np.pi / 16)),
            (np.cos(np.pi / 16 + np.pi / 8), np.sin(np.pi / 16 + np.pi / 8)),
            (np.cos(np.pi / 16 + 2 * np.pi / 8), np.sin(np.pi / 16 + 2 * np.pi / 8)),
            (np.cos(np.pi / 16 + 3 * np.pi / 8), np.sin(np.pi / 16 + 3 * np.pi / 8)),
            (np.cos(np.pi / 16 + 4 * np.pi / 8), np.sin(np.pi / 16 + 4 * np.pi / 8)),
            (np.cos(np.pi / 16 + 5 * np.pi / 8), np.sin(np.pi / 16 + 5 * np.pi / 8)),
            (np.cos(np.pi / 16 + 6 * np.pi / 8), np.sin(np.pi / 16 + 6 * np.pi / 8)),
            (np.cos(np.pi / 16 + 7 * np.pi / 8), np.sin(np.pi / 16 + 7 * np.pi / 8)),
        ]

        # Dirichlet boundary conditions
        dirbc_partial = fop.homogeneous_dirichlet(
            domain0, space0, boundary_tags, [dir_mkr], rank_dim
        )

        dirbc_total = fop.homogeneous_boundary(domain0, space0, dim, rank_dim)

        # Create measures to apply Neumman condition
        ds_parts = fop.marked_ds(
            domain0,
            boundary_tags,
            [
                neu_mkrA,
                neu_mkrB,
                neu_mkrC,
                neu_mkrD,
                neu_mkrE,
                neu_mkrF,
                neu_mkrG,
                neu_mkrH,
            ],
        )

        # Measures for force application
        ds_forces = ds_parts

        # Measure for adjoint problem
        ds1 = sum(ds_parts[1:], start=ds_parts[0])

        # Instance for data generation
        # We need the method pde0
        md0 = InverseElasticity(
            dim,
            domain0,
            space0,
            forces,
            ds_forces,
            ds1,
            dirbc_partial,
            dirbc_total,
            test_path,
        )

        # Function that defines
        # the level set function
        # to generate the data
        def SubDomain1_eq(x):
            """
            Implicit equation :
            (x^2/2 + y - 1/2)^2 = 1 - x^2
            """
            x0, y0 = -0.3, 0.3  # translation
            ef = 0.22  # scaled factor
            xx = (x[0] - x0) / ef
            yy = (x[1] - y0) / ef
            values = xx**4 + 4 * xx**2 * yy + 2 * xx**2 + 4 * yy**2 - 4 * yy - 3
            return np.log(0.1 * values + 1.0)

        def SubDomain2_eq(x):
            # inverse rotation
            angle = np.pi / 6.0
            cos_ = np.cos(angle)
            sin_ = np.sin(angle)
            xi = cos_ * x[0] + sin_ * x[1]
            yi = -sin_ * x[0] + cos_ * x[1]
            x0, y0 = 0.5, 0.2
            a, b = 0.12, 0.28
            xx = (xi - x0) / a
            yy = (yi - y0) / b
            values = xx**2 + yy**2 - 1
            return np.log(0.4 * values + 1.0)

        def SubDomain_combined(x):
            return np.minimum(SubDomain1_eq(x), SubDomain2_eq(x))

        # Dirichlet extensions
        extensions = fop.dir_extension_from(
            comm_self, domain0, space0, md0.pde0, SubDomain_combined, test_path
        )

    npts = 80  # npts % 4 = 0
    part = npts // 8

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.15, npts))

    # 1 dirichlet boundary
    # 8 neumann boundaries
    dir_idx, dir_mkr = [npts], 1
    neu_idxA, neu_mkrA = np.arange(1, part + 1), 2
    neu_idxB, neu_mkrB = np.arange(part + 1, 2 * part + 1), 3
    neu_idxC, neu_mkrC = np.arange(2 * part + 1, 3 * part + 1), 4
    neu_idxD, neu_mkrD = np.arange(3 * part + 1, 4 * part + 1), 5
    neu_idxE, neu_mkrE = np.arange(4 * part + 1, 5 * part + 1), 6
    neu_idxF, neu_mkrF = np.arange(5 * part + 1, 6 * part + 1), 7
    neu_idxG, neu_mkrG = np.arange(6 * part + 1, 7 * part + 1), 8
    neu_idxH, neu_mkrH = np.arange(7 * part + 1, npts), 9

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (neu_idxD, neu_mkrD, "neuD"),
        (neu_idxE, neu_mkrE, "neuE"),
        (neu_idxF, neu_mkrF, "neuF"),
        (neu_idxG, neu_mkrG, "neuG"),
        (neu_idxH, neu_mkrH, "neuH"),
    ]

    # Create gmsh domain for Task Parallelism
    output = fop.create_domain_2d_TP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    dirbc_total = fop.homogeneous_boundary(domain, space, dim, rank_dim)
    # Boundary to apply Neumann conditions
    ds_parts = fop.marked_ds(
        domain,
        boundary_tags,
        [
            neu_mkrA,
            neu_mkrB,
            neu_mkrC,
            neu_mkrD,
            neu_mkrE,
            neu_mkrF,
            neu_mkrG,
            neu_mkrH,
        ],
    )

    ds_forces = ds_parts
    ds1 = sum(ds_parts[1:], start=ds_parts[0])
    forces = [
        (np.cos(np.pi / 16), np.sin(np.pi / 16)),
        (np.cos(np.pi / 16 + np.pi / 8), np.sin(np.pi / 16 + np.pi / 8)),
        (np.cos(np.pi / 16 + 2 * np.pi / 8), np.sin(np.pi / 16 + 2 * np.pi / 8)),
        (np.cos(np.pi / 16 + 3 * np.pi / 8), np.sin(np.pi / 16 + 3 * np.pi / 8)),
        (np.cos(np.pi / 16 + 4 * np.pi / 8), np.sin(np.pi / 16 + 4 * np.pi / 8)),
        (np.cos(np.pi / 16 + 5 * np.pi / 8), np.sin(np.pi / 16 + 5 * np.pi / 8)),
        (np.cos(np.pi / 16 + 6 * np.pi / 8), np.sin(np.pi / 16 + 6 * np.pi / 8)),
        (np.cos(np.pi / 16 + 7 * np.pi / 8), np.sin(np.pi / 16 + 7 * np.pi / 8)),
    ]

    # Create the model
    md = InverseElasticity(
        dim,
        domain,
        space,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        test_path,
    )

    # Space for interpolation (degree = 2)
    g_space = fop.create_space(domain, "CG", rank_dim, degree=2)
    if rank == 0:
        # Interpolation between different spaces
        # from different domains
        g_funcs = fop.space_interpolation(
            from_space=space0, funcs=extensions, to_space=g_space
        )

        # To save as P1 functions
        g_space_1 = fop.create_space(domain, "CG", rank_dim)
        g_funcs_1 = fop.interpolate(funcs=g_funcs, to_space=g_space_1, name="g")
        fop.save_functions(comm_self, domain, g_funcs_1, test_path / "gP1.xdmf")

        g_values = np.vstack([g.x.array[:] for g in g_funcs])

    else:
        g_values = None

    g_values = comm.bcast(g_values, root=0)

    md.gs = fop.get_funcs_from(g_space, g_values)

    # Initial guess: centers and radii
    centers = np.array([(-0.3, 0.4), (0.3, 0.4)])
    radii = np.array([0.15, 0.15])

    md.create_initial_level(centers, radii, factor=-1.0)
    if rank == 0:
        md.save_initial_level(comm_self)

    # Run Data Parallelism
    md.runTP(niter=300, dfactor=1e-1, lv_time=(1e-3, 1.0), cost_tol=1e-1)


def test_36():
    """
    Elasticity Inverse Problem (two inclusions) - Mixed Parallelism

    Run `mpirun -np <16n> python test.py 36`.
    For instance, `mpirun -np 32 python test.py 36`.
    """

    # Verification
    nbr_groups = 16
    if size % nbr_groups != 0:
        print(f"Nbr of processes must be divisible by {nbr_groups}")
        return

    # Subcommunicators
    color = rank * nbr_groups // size
    sub_comm = comm.Split(color, rank)

    test_name = "Elasticity Inverse Problem (two inclusions) - Mixed Parallelism"
    test_path = Path("../results/t36/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    # Data generation

    def semi_ellipse(a, b, eps, npts):
        """
        Coordinates of a ellipse
        crossing the x-axis
        """
        t_ = np.arcsin((b - eps) / b)
        t = np.linspace(-t_, np.pi + t_, npts)
        x = a * np.cos(t)
        y = b * np.sin(t) + (b - eps)
        return x, y

    npts = 80  # npts % 4 = 0
    part = npts // 8

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.15, npts))

    # 1 dirichlet boundary
    # 8 neumann boundaries
    dir_idx, dir_mkr = [npts], 1
    neu_idxA, neu_mkrA = np.arange(1, part + 1), 2
    neu_idxB, neu_mkrB = np.arange(part + 1, 2 * part + 1), 3
    neu_idxC, neu_mkrC = np.arange(2 * part + 1, 3 * part + 1), 4
    neu_idxD, neu_mkrD = np.arange(3 * part + 1, 4 * part + 1), 5
    neu_idxE, neu_mkrE = np.arange(4 * part + 1, 5 * part + 1), 6
    neu_idxF, neu_mkrF = np.arange(5 * part + 1, 6 * part + 1), 7
    neu_idxG, neu_mkrG = np.arange(6 * part + 1, 7 * part + 1), 8
    neu_idxH, neu_mkrH = np.arange(7 * part + 1, npts), 9

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (neu_idxD, neu_mkrD, "neuD"),
        (neu_idxE, neu_mkrE, "neuE"),
        (neu_idxF, neu_mkrF, "neuF"),
        (neu_idxG, neu_mkrG, "neuG"),
        (neu_idxH, neu_mkrH, "neuH"),
    ]

    def SubDomain1(n):
        # Lifted circle
        t = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        x0, y0 = -0.3, 0.3  # translation
        ef = 0.22  # scaled factor
        x = ef * np.cos(t) + x0
        y = ef * (np.sin(t) + (np.sin(t)) ** 2 / 2.0) + y0
        return x, y

    def SubDomain2(n):
        # Rotated ellipse
        t = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        angle = np.pi / 6.0
        x0, y0 = 0.5, 0.2
        a, b = 0.12, 0.28
        x = a * np.cos(t) + x0
        y = b * np.sin(t) + y0
        cos_ = np.cos(angle)
        sin_ = np.sin(angle)
        x_rot = cos_ * x - sin_ * y
        y_rot = sin_ * x + cos_ * y
        return x_rot, y_rot

    subdomain1 = np.column_stack(SubDomain1(60))

    subdomain2 = np.column_stack(SubDomain2(60))

    if rank == 0:
        np.save(test_path / "inclusion1.npy", subdomain1)
        np.save(test_path / "inclusion2.npy", subdomain2)

    filename = test_path / "domain0.msh"

    # Create the gmsh domain0.msh
    nbr_tri0 = fop.build_gmsh_model_2d(
        vertices,
        boundary_parts,
        0.6 * mesh_size,
        curves=[subdomain1, subdomain2],
        filename=filename,
        plot=False,
    )

    if color == 0:
        # Read the domain0
        output = fop.read_gmsh(filename, sub_comm, dim=2)

        domain0, _, boundary_tags = output

        # Set all connectivities on domain0
        fop.all_connectivities(domain0)

        # Space for data generation
        space0 = fop.create_space(domain0, "CG", rank_dim)
        # Forces
        forces = [
            (np.cos(np.pi / 16), np.sin(np.pi / 16)),
            (np.cos(np.pi / 16 + np.pi / 8), np.sin(np.pi / 16 + np.pi / 8)),
            (np.cos(np.pi / 16 + 2 * np.pi / 8), np.sin(np.pi / 16 + 2 * np.pi / 8)),
            (np.cos(np.pi / 16 + 3 * np.pi / 8), np.sin(np.pi / 16 + 3 * np.pi / 8)),
            (np.cos(np.pi / 16 + 4 * np.pi / 8), np.sin(np.pi / 16 + 4 * np.pi / 8)),
            (np.cos(np.pi / 16 + 5 * np.pi / 8), np.sin(np.pi / 16 + 5 * np.pi / 8)),
            (np.cos(np.pi / 16 + 6 * np.pi / 8), np.sin(np.pi / 16 + 6 * np.pi / 8)),
            (np.cos(np.pi / 16 + 7 * np.pi / 8), np.sin(np.pi / 16 + 7 * np.pi / 8)),
        ]

        # Dirichlet boundary conditions
        dirbc_partial = fop.homogeneous_dirichlet(
            domain0, space0, boundary_tags, [dir_mkr], rank_dim
        )

        dirbc_total = fop.homogeneous_boundary(domain0, space0, dim, rank_dim)

        # Create measures to apply Neumman condition
        ds_parts = fop.marked_ds(
            domain0,
            boundary_tags,
            [
                neu_mkrA,
                neu_mkrB,
                neu_mkrC,
                neu_mkrD,
                neu_mkrE,
                neu_mkrF,
                neu_mkrG,
                neu_mkrH,
            ],
        )

        # Measures for force application
        ds_forces = ds_parts

        # Measure for adjoint problem
        ds1 = sum(ds_parts[1:], start=ds_parts[0])

        # Instance for data generation
        # We need the method pde0
        md0 = InverseElasticity(
            dim,
            domain0,
            space0,
            forces,
            ds_forces,
            ds1,
            dirbc_partial,
            dirbc_total,
            test_path,
        )

        # Function that defines
        # the level set function
        # to generate the data
        def SubDomain1_eq(x):
            """
            Implicit equation :
            (x^2/2 + y - 1/2)^2 = 1 - x^2
            """
            x0, y0 = -0.3, 0.3  # translation
            ef = 0.22  # scaled factor
            xx = (x[0] - x0) / ef
            yy = (x[1] - y0) / ef
            values = xx**4 + 4 * xx**2 * yy + 2 * xx**2 + 4 * yy**2 - 4 * yy - 3
            return np.log(0.1 * values + 1.0)

        def SubDomain2_eq(x):
            # inverse rotation
            angle = np.pi / 6.0
            cos_ = np.cos(angle)
            sin_ = np.sin(angle)
            xi = cos_ * x[0] + sin_ * x[1]
            yi = -sin_ * x[0] + cos_ * x[1]
            x0, y0 = 0.5, 0.2
            a, b = 0.12, 0.28
            xx = (xi - x0) / a
            yy = (yi - y0) / b
            values = xx**2 + yy**2 - 1
            return np.log(0.4 * values + 1.0)

        def SubDomain_combined(x):
            return np.minimum(SubDomain1_eq(x), SubDomain2_eq(x))

        # Dirichlet extensions
        extensions = fop.dir_extension_from(
            sub_comm, domain0, space0, md0.pde0, SubDomain_combined, test_path
        )

    npts = 80  # npts % 4 = 0
    part = npts // 8

    vertices = np.column_stack(semi_ellipse(0.75, 0.5, 0.15, npts))

    # 1 dirichlet boundary
    # 8 neumann boundaries
    dir_idx, dir_mkr = [npts], 1
    neu_idxA, neu_mkrA = np.arange(1, part + 1), 2
    neu_idxB, neu_mkrB = np.arange(part + 1, 2 * part + 1), 3
    neu_idxC, neu_mkrC = np.arange(2 * part + 1, 3 * part + 1), 4
    neu_idxD, neu_mkrD = np.arange(3 * part + 1, 4 * part + 1), 5
    neu_idxE, neu_mkrE = np.arange(4 * part + 1, 5 * part + 1), 6
    neu_idxF, neu_mkrF = np.arange(5 * part + 1, 6 * part + 1), 7
    neu_idxG, neu_mkrG = np.arange(6 * part + 1, 7 * part + 1), 8
    neu_idxH, neu_mkrH = np.arange(7 * part + 1, npts), 9

    boundary_parts = [
        (dir_idx, dir_mkr, "dir"),
        (neu_idxA, neu_mkrA, "neuA"),
        (neu_idxB, neu_mkrB, "neuB"),
        (neu_idxC, neu_mkrC, "neuC"),
        (neu_idxD, neu_mkrD, "neuD"),
        (neu_idxE, neu_mkrE, "neuE"),
        (neu_idxF, neu_mkrF, "neuF"),
        (neu_idxG, neu_mkrG, "neuG"),
        (neu_idxH, neu_mkrH, "neuH"),
    ]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_MP(
        sub_comm, color, vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Nbr of triangles for data generation = {nbr_tri0}")

    # Space
    space = fop.create_space(domain, "CG", rank_dim)
    # Dirichlet boundary conditions
    dirbc_partial = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )
    dirbc_total = fop.homogeneous_boundary(domain, space, dim, rank_dim)
    # Boundary to apply Neumann conditions
    ds_parts = fop.marked_ds(
        domain,
        boundary_tags,
        [
            neu_mkrA,
            neu_mkrB,
            neu_mkrC,
            neu_mkrD,
            neu_mkrE,
            neu_mkrF,
            neu_mkrG,
            neu_mkrH,
        ],
    )

    ds_forces = ds_parts
    ds1 = sum(ds_parts[1:], start=ds_parts[0])
    forces = [
        (np.cos(np.pi / 16), np.sin(np.pi / 16)),
        (np.cos(np.pi / 16 + np.pi / 8), np.sin(np.pi / 16 + np.pi / 8)),
        (np.cos(np.pi / 16 + 2 * np.pi / 8), np.sin(np.pi / 16 + 2 * np.pi / 8)),
        (np.cos(np.pi / 16 + 3 * np.pi / 8), np.sin(np.pi / 16 + 3 * np.pi / 8)),
        (np.cos(np.pi / 16 + 4 * np.pi / 8), np.sin(np.pi / 16 + 4 * np.pi / 8)),
        (np.cos(np.pi / 16 + 5 * np.pi / 8), np.sin(np.pi / 16 + 5 * np.pi / 8)),
        (np.cos(np.pi / 16 + 6 * np.pi / 8), np.sin(np.pi / 16 + 6 * np.pi / 8)),
        (np.cos(np.pi / 16 + 7 * np.pi / 8), np.sin(np.pi / 16 + 7 * np.pi / 8)),
    ]

    # Create the model
    md = InverseElasticity(
        dim,
        domain,
        space,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        test_path,
    )

    # Space for interpolation (degree = 2)
    g_space = fop.create_space(domain, "CG", rank_dim, degree=2)

    if color == 0:
        # Interpolation between different spaces
        # from different domains
        g_funcs = fop.space_interpolation(
            from_space=space0, funcs=extensions, to_space=g_space
        )

        # To save as P1 functions
        g_space_1 = fop.create_space(domain, "CG", rank_dim)
        g_funcs_1 = fop.interpolate(funcs=g_funcs, to_space=g_space_1, name="g")
        fop.save_functions(sub_comm, domain, g_funcs_1, test_path / "gP1.xdmf")

        g_values_loc = np.vstack([g.x.array[:] for g in g_funcs])

    else:
        g_values_loc = None

    g_values = comm.allgather(g_values_loc)

    md.gs = fop.get_funcs_from(g_space, g_values[sub_comm.rank])

    # Initial guess: centers and radii
    centers = np.array([(-0.3, 0.4), (0.3, 0.4)])
    radii = np.array([0.15, 0.15])

    md.create_initial_level(centers, radii, factor=-1.0)
    if color == 0:
        md.save_initial_level(sub_comm)

    # Run Mixed Parallelism
    md.runMP(sub_comm, niter=200, dfactor=1e-1, lv_time=(1e-3, 1.0), cost_tol=1e-1)


def test_37():
    """
    Symmetric Cantilever 2D (SVK)

    Run `mpirun -np <nbr of processes> python test.py 37`.
    For instance, `mpirun -np 2 python test.py 37`.

    To save the output, append `> ../results/t37/out.txt`.
    To delete the images, enter `rm ../results/t37/*.png`.
    """

    test_path = Path("../results/t37/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.011

    vertices = np.array(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.45), (2.0, 0.55), (2.0, 1.0), (0.0, 1.0)]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [(dir_idx, dir_mkr, "dir"), (neu_idx, neu_mkr, "neu")]

    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    space = fop.create_space(domain, "CG", rank_dim)

    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )

    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])

    alpha = 0.25
    g = (0.0, -10.0)

    md = SVK(dim, domain, space, g, ds_g[0], dirichlet_bcs, alpha, test_path)

    @fop.region_of(domain)
    def sub_domain(x):
        # 0.42 < x[1] < 0.58
        # 1.95 < x[0]
        ineqs = [x[1] - 0.42, 0.58 - x[1], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * 0.7, 0.0) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.25) for i in range(2)]
    centers += [(0.3 + i * 0.7, 0.5) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.75) for i in range(2)]
    centers += [(0.3 + i * 0.7, 1.0) for i in range(3)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    md.create_initial_level(centers, radii)

    md.runDP(
        niter=150,
        dfactor=0.01,
        lv_iter=(10, 25),
        lv_time=(0.01, 0.1),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(6, 0.01),
    )


def test_38():

    test_path = Path("../results/t38/")
    dim = 2
    rank_dim = 2
    mesh_size = 0.012

    vertices = np.array(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.45), (2.0, 0.55), (2.0, 1.0), (0.0, 1.0)]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [(dir_idx, dir_mkr, "dir"), (neu_idx, neu_mkr, "neu")]

    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    space = fop.create_space(domain, "CG", rank_dim)

    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )

    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])

    alpha = 0.25
    g = (0.0, -10.0)

    md = NHK(dim, domain, space, g, ds_g[0], dirichlet_bcs, alpha, test_path)
    md.bc_theta = (boundary_tags, [neu_mkr])
    md.ds_theta = fop.marked_ds(domain, boundary_tags, [dir_mkr])[0]

    @fop.region_of(domain)
    def sub_domain(x):
        # 0.42 < x[1] < 0.58
        # 1.95 < x[0]
        ineqs = [x[1] - 0.42, 0.58 - x[1], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * 0.7, 0.0) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.25) for i in range(2)]
    centers += [(0.3 + i * 0.7, 0.5) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.75) for i in range(2)]
    centers += [(0.3 + i * 0.7, 1.0) for i in range(3)]
    centers = np.array(centers)
    radii = np.repeat(0.08, centers.shape[0])
    md.create_initial_level(centers, radii)

    md.runDP(
        niter=150,
        dfactor=0.01,
        lv_iter=(10, 25),
        lv_time=(0.01, 0.1),
        cost_tol=1e-3,
        smooth=True,
        reinit_step=4,
        reinit_pars=(6, 0.01),
    )


def test_42():
    """
    Symmetric cantilever 3D - Data Parallelism (1 process)

    Run: `mpirun -np 1 python test.py 42`
    This test is part of the set (executed laptop):
        test_42 with 1 process,
        test_43 with 2 process, and
        test_44 with 4 process.
    See the output with the time execution in
    the `out.txt` file of the corresponding folder.
    """

    test_name = "Symmetric cantilever 3D - Data Parallelism (1 process)"
    test_path = Path("../results/t42/")
    dim = 3
    rank_dim = 3
    mesh_size = 60

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")

    from dolfinx.mesh import create_box

    domain = create_box(
        comm, [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], [2 * mesh_size, mesh_size, mesh_size]
    )
    fop.all_connectivities(domain)
    fop.save_domain(comm, domain, test_path / "domain.xdmf")

    space = fop.create_space(domain, "CG", rank_dim)

    def boundary_dirichlet(x):
        return np.isclose(x[0], 0.0)

    def boundary_neumann(x):
        in_plane = np.isclose(x[0], 2.0)
        in_square = np.maximum(np.abs(x[1] - 0.5), np.abs(x[2] - 0.5)) <= 0.1
        return in_plane & in_square

    dirichlet_bcs = fop.homogeneous_dirichlet_fun(
        domain, space, [boundary_dirichlet], rank_dim
    )
    ds_g = fop.fun_ds(domain, [boundary_neumann])
    volume, g = 1.0, [0.0, 0.0, -4.0]
    md = Compliance(dim, domain, space, g, ds_g[0], dirichlet_bcs, volume, test_path)

    @fop.region_of(domain)
    def sub_domain(x):
        ineqs = [x[1] - 0.35, 0.65 - x[1], x[2] - 0.35, 0.65 - x[2], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    centers = np.array(
        [
            (2.0, 0.7, 0.3),
            (2.0, 0.7, 0.7),
            (2.0, 0.3, 0.3),
            (2.0, 0.3, 0.7),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
            (2.0, 0.5, 0.0),
            (2.0, 0.5, 1.0),
            (2.0, 0.0, 0.5),
            (2.0, 1.0, 0.5),
            (1.7, 0.0, 0.0),
            (1.7, 0.0, 1.0),
            (1.7, 1.0, 0.0),
            (1.7, 1.0, 1.0),
            (1.7, 0.5, 0.5),
            (1.7, 0, 0.5),
            (1.7, 1.0, 0.5),
            (1.7, 0.5, 0.0),
            (1.7, 0.5, 1.0),
            (1.35, 0.25, 0.25),
            (1.35, 0.75, 0.25),
            (1.35, 0.25, 0.75),
            (1.35, 0.75, 0.75),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.5, 0.5),
            (1.0, 0, 0.5),
            (1.0, 1.0, 0.5),
            (1.0, 0.5, 0.0),
            (1.0, 0.5, 1.0),
            (0.65, 0.25, 0.25),
            (0.65, 0.75, 0.25),
            (0.65, 0.25, 0.75),
            (0.65, 0.75, 0.75),
            (0.3, 0.0, 0.0),
            (0.3, 0.0, 1.0),
            (0.3, 1.0, 0.0),
            (0.3, 1.0, 1.0),
            (0.3, 0.5, 0.5),
            (0.3, 0, 0.5),
            (0.3, 1.0, 0.5),
            (0.3, 0.5, 0.0),
            (0.3, 0.5, 1.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.5, 1.0),
            (0.0, 0.0, 0.5),
            (0.0, 1.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.0, 0.25, 0.25),
            (0.0, 0.75, 0.25),
            (0.0, 0.25, 0.75),
            (0.0, 0.75, 0.75),
            (2.0, 0.5, 0.3),
            (2.0, 0.5, 0.7),
            (2.0, 0.3, 0.5),
            (2.0, 0.7, 0.5),
        ]
    )
    radii = np.repeat(0.1, centers.shape[0])
    md.create_initial_level(centers, radii, ord=np.inf)
    md.save_initial_level(comm)
    md.runDP(
        ctrn_tol=1e-3, dfactor=1e-1, reinit_step=4, reinit_pars=(4, 0.01), smooth=True
    )


def test_43():
    """
    Symmetric cantilever 3D - Data Parallelism (2 process)

    Run: `mpirun -np 2 python test.py 43`
    This test is part of the set (executed laptop):
        test_42 with 1 process,
        test_43 with 2 process, and
        test_44 with 4 process.
    See the output with the time execution in
    the `out.txt` file of the corresponding folder.
    """

    test_name = "Symmetric cantilever 3D - Data Parallelism (2 process)"
    test_path = Path("../results/t43/")
    dim = 3
    rank_dim = 3
    mesh_size = 60

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")

    from dolfinx.mesh import create_box

    domain = create_box(
        comm, [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], [2 * mesh_size, mesh_size, mesh_size]
    )
    fop.all_connectivities(domain)
    fop.save_domain(comm, domain, test_path / "domain.xdmf")

    space = fop.create_space(domain, "CG", rank_dim)

    def boundary_dirichlet(x):
        return np.isclose(x[0], 0.0)

    def boundary_neumann(x):
        in_plane = np.isclose(x[0], 2.0)
        in_square = np.maximum(np.abs(x[1] - 0.5), np.abs(x[2] - 0.5)) <= 0.1
        return in_plane & in_square

    dirichlet_bcs = fop.homogeneous_dirichlet_fun(
        domain, space, [boundary_dirichlet], rank_dim
    )
    ds_g = fop.fun_ds(domain, [boundary_neumann])
    volume, g = 1.0, [0.0, 0.0, -4.0]
    md = Compliance(dim, domain, space, g, ds_g[0], dirichlet_bcs, volume, test_path)

    @fop.region_of(domain)
    def sub_domain(x):
        ineqs = [x[1] - 0.35, 0.65 - x[1], x[2] - 0.35, 0.65 - x[2], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    centers = np.array(
        [
            (2.0, 0.7, 0.3),
            (2.0, 0.7, 0.7),
            (2.0, 0.3, 0.3),
            (2.0, 0.3, 0.7),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
            (2.0, 0.5, 0.0),
            (2.0, 0.5, 1.0),
            (2.0, 0.0, 0.5),
            (2.0, 1.0, 0.5),
            (1.7, 0.0, 0.0),
            (1.7, 0.0, 1.0),
            (1.7, 1.0, 0.0),
            (1.7, 1.0, 1.0),
            (1.7, 0.5, 0.5),
            (1.7, 0, 0.5),
            (1.7, 1.0, 0.5),
            (1.7, 0.5, 0.0),
            (1.7, 0.5, 1.0),
            (1.35, 0.25, 0.25),
            (1.35, 0.75, 0.25),
            (1.35, 0.25, 0.75),
            (1.35, 0.75, 0.75),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.5, 0.5),
            (1.0, 0, 0.5),
            (1.0, 1.0, 0.5),
            (1.0, 0.5, 0.0),
            (1.0, 0.5, 1.0),
            (0.65, 0.25, 0.25),
            (0.65, 0.75, 0.25),
            (0.65, 0.25, 0.75),
            (0.65, 0.75, 0.75),
            (0.3, 0.0, 0.0),
            (0.3, 0.0, 1.0),
            (0.3, 1.0, 0.0),
            (0.3, 1.0, 1.0),
            (0.3, 0.5, 0.5),
            (0.3, 0, 0.5),
            (0.3, 1.0, 0.5),
            (0.3, 0.5, 0.0),
            (0.3, 0.5, 1.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.5, 1.0),
            (0.0, 0.0, 0.5),
            (0.0, 1.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.0, 0.25, 0.25),
            (0.0, 0.75, 0.25),
            (0.0, 0.25, 0.75),
            (0.0, 0.75, 0.75),
            (2.0, 0.5, 0.3),
            (2.0, 0.5, 0.7),
            (2.0, 0.3, 0.5),
            (2.0, 0.7, 0.5),
        ]
    )
    radii = np.repeat(0.1, centers.shape[0])
    md.create_initial_level(centers, radii, ord=np.inf)
    md.save_initial_level(comm)
    md.runDP(
        ctrn_tol=1e-3, dfactor=1e-1, reinit_step=4, reinit_pars=(4, 0.01), smooth=True
    )


def test_44():
    """
    Symmetric cantilever 3D - Data Parallelism (4 process)

    Run: `mpirun -np 4 python test.py 44`
    This test is part of the set (executed laptop):
        test_42 with 1 process,
        test_43 with 2 process, and
        test_44 with 4 process.
    See the output with the time execution in
    the `out.txt` file of the corresponding folder.
    """

    test_name = "Symmetric cantilever 3D - Data Parallelism (4 process)"
    test_path = Path("../results/t44/")
    dim = 3
    rank_dim = 3
    mesh_size = 60

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")

    from dolfinx.mesh import create_box

    domain = create_box(
        comm, [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], [2 * mesh_size, mesh_size, mesh_size]
    )
    fop.all_connectivities(domain)
    fop.save_domain(comm, domain, test_path / "domain.xdmf")

    space = fop.create_space(domain, "CG", rank_dim)

    def boundary_dirichlet(x):
        return np.isclose(x[0], 0.0)

    def boundary_neumann(x):
        in_plane = np.isclose(x[0], 2.0)
        in_square = np.maximum(np.abs(x[1] - 0.5), np.abs(x[2] - 0.5)) <= 0.1
        return in_plane & in_square

    dirichlet_bcs = fop.homogeneous_dirichlet_fun(
        domain, space, [boundary_dirichlet], rank_dim
    )
    ds_g = fop.fun_ds(domain, [boundary_neumann])
    volume, g = 1.0, [0.0, 0.0, -4.0]
    md = Compliance(dim, domain, space, g, ds_g[0], dirichlet_bcs, volume, test_path)

    @fop.region_of(domain)
    def sub_domain(x):
        ineqs = [x[1] - 0.35, 0.65 - x[1], x[2] - 0.35, 0.65 - x[2], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    centers = np.array(
        [
            (2.0, 0.7, 0.3),
            (2.0, 0.7, 0.7),
            (2.0, 0.3, 0.3),
            (2.0, 0.3, 0.7),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
            (2.0, 0.5, 0.0),
            (2.0, 0.5, 1.0),
            (2.0, 0.0, 0.5),
            (2.0, 1.0, 0.5),
            (1.7, 0.0, 0.0),
            (1.7, 0.0, 1.0),
            (1.7, 1.0, 0.0),
            (1.7, 1.0, 1.0),
            (1.7, 0.5, 0.5),
            (1.7, 0, 0.5),
            (1.7, 1.0, 0.5),
            (1.7, 0.5, 0.0),
            (1.7, 0.5, 1.0),
            (1.35, 0.25, 0.25),
            (1.35, 0.75, 0.25),
            (1.35, 0.25, 0.75),
            (1.35, 0.75, 0.75),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.5, 0.5),
            (1.0, 0, 0.5),
            (1.0, 1.0, 0.5),
            (1.0, 0.5, 0.0),
            (1.0, 0.5, 1.0),
            (0.65, 0.25, 0.25),
            (0.65, 0.75, 0.25),
            (0.65, 0.25, 0.75),
            (0.65, 0.75, 0.75),
            (0.3, 0.0, 0.0),
            (0.3, 0.0, 1.0),
            (0.3, 1.0, 0.0),
            (0.3, 1.0, 1.0),
            (0.3, 0.5, 0.5),
            (0.3, 0, 0.5),
            (0.3, 1.0, 0.5),
            (0.3, 0.5, 0.0),
            (0.3, 0.5, 1.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.5, 1.0),
            (0.0, 0.0, 0.5),
            (0.0, 1.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.0, 0.25, 0.25),
            (0.0, 0.75, 0.25),
            (0.0, 0.25, 0.75),
            (0.0, 0.75, 0.75),
            (2.0, 0.5, 0.3),
            (2.0, 0.5, 0.7),
            (2.0, 0.3, 0.5),
            (2.0, 0.7, 0.5),
        ]
    )
    radii = np.repeat(0.1, centers.shape[0])
    md.create_initial_level(centers, radii, ord=np.inf)
    md.save_initial_level(comm)
    md.runDP(
        ctrn_tol=1e-3, dfactor=1e-1, reinit_step=4, reinit_pars=(4, 0.01), smooth=True
    )


def test_45(eps, test_path):

    test_name = "Symmetric Cantilever 2D with several Ersatz material"
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    vertices = np.array(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.45), (2.0, 0.55), (2.0, 1.0), (0.0, 1.0)]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [(dir_idx, dir_mkr, "dir"), (neu_idx, neu_mkr, "neu")]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Ersatz material = {eps}")

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)

    # Dirichlet condition
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )

    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])

    area = 1.0
    g = (0.0, -2.0)
    # Create the model
    md = Compliance(dim, domain, space, g, ds_g[0], dirichlet_bcs, area, test_path)

    md.updateA(eps)

    @fop.region_of(domain)
    def sub_domain(x):
        # 0.42 < x[1] < 0.58
        # 1.95 < x[0]
        ineqs = [x[1] - 0.42, 0.58 - x[1], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    # Initial guess: centers and radii

    # First set of centers (for example in manuscript):
    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * 0.7, 0.0) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.25) for i in range(2)]
    centers += [(0.3 + i * 0.7, 0.5) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.75) for i in range(2)]
    centers += [(0.3 + i * 0.7, 1.0) for i in range(3)]

    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])

    # Create the initial level set function
    md.create_initial_level(centers, radii)
    # Save as initial.xdmf
    md.save_initial_level(comm)

    md.runDP(
        ctrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(20, 0.1),
        smooth=True,
    )


def test_46():
    test_45(1e-2, Path("../results/t46/"))


def test_47():
    test_45(1e-3, Path("../results/t47/"))


def test_48():
    test_45(1e-4, Path("../results/t48/"))


def test_49():
    test_45(1e-5, Path("../results/t49/"))


def test_50():
    test_45(1e-6, Path("../results/t50/"))


def test_51():
    test_45(1e-7, Path("../results/t51/"))


def test_55(value, test_path):

    test_name = "Symmetric Cantilever 2D with several Ersatz material"
    dim = 2
    rank_dim = 2
    mesh_size = 0.015

    vertices = np.array(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 0.45), (2.0, 0.55), (2.0, 1.0), (0.0, 1.0)]
    )

    dir_idx, dir_mkr = [6], 1
    neu_idx, neu_mkr = [3], 2
    boundary_parts = [(dir_idx, dir_mkr, "dir"), (neu_idx, neu_mkr, "neu")]

    # Create gmsh domain for Data Parallelism
    output = fop.create_domain_2d_DP(
        vertices, boundary_parts, mesh_size, path=test_path, plot=False
    )

    domain, nbr_tri, boundary_tags = output

    if rank == 0:
        print("\n\t" + test_name + "\n")
        print(f"> Path = {test_path}")
        print(f"> Nbr of triangles = {nbr_tri}")
        print(f"> Value = {value}")

    # Space for the PDE solution
    space = fop.create_space(domain, "CG", rank_dim)

    # Dirichlet condition
    dirichlet_bcs = fop.homogeneous_dirichlet(
        domain, space, boundary_tags, [dir_mkr], rank_dim
    )

    # Boundary to force application
    ds_g = fop.marked_ds(domain, boundary_tags, [neu_mkr])

    area = 1.0
    g = (0.0, -2.0)
    # Create the model
    md = Compliance(dim, domain, space, g, ds_g[0], dirichlet_bcs, area, test_path)
    md.Coef = value

    @fop.region_of(domain)
    def sub_domain(x):
        # 0.42 < x[1] < 0.58
        # 1.95 < x[0]
        ineqs = [x[1] - 0.42, 0.58 - x[1], x[0] - 1.90]
        return ineqs

    md.sub = [sub_domain.expression()]

    # Initial guess: centers and radii

    # First set of centers (for example in manuscript):
    centers = [(2.0, 0.35), (2.0, 0.65), (2.0, 0.0), (2.0, 1.0)]
    centers += [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
    centers += [(0.3 + i * 0.7, 0.0) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.25) for i in range(2)]
    centers += [(0.3 + i * 0.7, 0.5) for i in range(3)]
    centers += [(0.65 + i * 0.7, 0.75) for i in range(2)]
    centers += [(0.3 + i * 0.7, 1.0) for i in range(3)]

    centers = np.array(centers)
    radii = np.repeat(0.1, centers.shape[0])

    # Create the initial level set function
    md.create_initial_level(centers, radii)
    # Save as initial.xdmf
    md.save_initial_level(comm)

    md.runDP(
        niter=250,
        ctrn_tol=1e-3,
        dfactor=1e-1,
        reinit_step=4,
        reinit_pars=(20, 0.1),
        smooth=True,
    )


def test_56():
    test_55(0.5, Path("../results/t56/"))


def test_57():
    test_55(1.0, Path("../results/t57/"))


def test_58():
    test_55(1.5, Path("../results/t58/"))


def test_59():
    test_55(2.0, Path("../results/t59/"))


def test_60():
    test_55(2.5, Path("../results/t60/"))


test_functions = {
    "01": test_01,
    "02": test_02,
    "03": test_03,
    "04": test_04,
    "05": test_05,
    "06": test_06,
    "07": test_07,
    "08": test_08,
    "09": test_09,
    "10": test_10,
    "11": test_11,
    "12": test_12,
    "13": test_13,
    "14": test_14,
    "15": test_15,
    "16": test_16,
    "17": test_17,
    "18": test_18,
    "19": test_19,
    "20": test_20,
    "21": test_21,
    "22": test_22,
    "23": test_23,
    "31": test_31,
    "32": test_32,
    "33": test_33,
    "34": test_34,
    "35": test_35,
    "36": test_36,
    "37": test_37,
    "38": test_38,
    "42": test_42,
    "43": test_43,
    "44": test_44,
    "46": test_46,
    "47": test_47,
    "48": test_48,
    "49": test_49,
    "50": test_50,
    "51": test_51,
    "55": test_55,
    "56": test_56,
    "57": test_57,
    "58": test_58,
    "59": test_59,
    "60": test_60,
}


def main():

    import sys

    if len(sys.argv) != 2:
        print("Usage: python tests.py <test_id>")
        print("Example: python test.py 01")
        return

    test_id = sys.argv[1]
    func = test_functions.get(test_id)

    if func:
        func()
    else:
        print(f"Test '{test_id}' not recognized.")


if __name__ == "__main__":
    main()
