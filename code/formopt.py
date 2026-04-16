"""
=================================================================================
Module: FormOpt
Article Title:
    FormOpt: A FEniCSx toolbox for level set-based shape optimization
    supporting parallel computing
Authors:
    Josué D. Díaz-Avalos and Antoine Laurain (University of Duisburg-Essen)
Description:
    FormOpt is a FEniCSx-based Python module for two- and three-dimensional
    shape optimization using a level set method and with parallel computing support.
Dependencies:
    - FEniCSx 0.9
    - MPI for parallel computing
    - Gmsh
    - NumPy, SciPy
    - Matplotlib
Citation:
    If you use this module in your work, please cite:
    Josué D. Díaz-Avalos and Antoine Laurain. FormOpt: A FEniCSx toolbox
    for level set-based shape optimization supporting parallel computing.
License:
    GNU Lesser General Public License v3.0 (LGPL-3.0)
=================================================================================
"""

from pathlib import Path
from h5py import File

import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmshio, XDMFFile

import numpy as np
from numpy.linalg import norm as Norm

from dolfinx.cpp.mesh import MeshTags_int32

from dolfinx import default_real_type

from dolfinx.mesh import (
    Mesh,
    locate_entities_boundary,
    exterior_facet_indices,
    meshtags,
)

from dolfinx.fem.petsc import (
    LinearProblem,
    DirichletBC,
    NonlinearProblem,
    assemble_matrix,
    assemble_vector,
    create_vector,
    apply_lifting,
    set_bc,
)


from dolfinx.fem import (
    FunctionSpace,
    Function,
    Constant,
    create_interpolation_data,
    locate_dofs_topological,
    locate_dofs_geometrical,
    assemble_scalar,
    form,
    functionspace,
    dirichletbc,
)

from ufl import (
    replace,
    Form,
    SpatialCoordinate,
    Measure,
    Coefficient,
    TrialFunction,
    TestFunction,
    system,
    conditional,
    inner,
    grad,
    dot,
    sqrt,
    gt,
    lt,
    MixedFunctionSpace,
    extract_blocks,
)

from ufl.argument import Argument

from dolfinx.nls.petsc import NewtonSolver

from abc import ABC, abstractmethod

import numpy.typing as npt
from typing import List, Tuple, Callable, Sequence, final
from ufl.core.expr import Expr

from plots import plot_domain

from mesh_scripts import compute_tags_measures  # phiFem

from basix.ufl import element, mixed_element

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
comm_self = MPI.COMM_SELF

dom_name = "domain"
dat_name = "data"
res_name = "results"
ini_name = "initial"
ext_name = "extensions"
inf_name = "info_solver"

ScalarFunc = Callable[[npt.NDArray[np.float64]], float]
VectorFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
GenericFunc = ScalarFunc | ScalarFunc


class Model(ABC):
    """
    Base class for all user-defined models.

    Attributes
    ----------
    dim : int
        Domain dimension.
    domain : Mesh
        Problem domain.
    space : FunctionSpace
        Space of functions for the solution and test functions.
    path : Path
        Test path to save the results.

    Methods
    -------
    pde(level_set_func: Function) -> List[Tuple[Expr, Sequence[DirichletBC]]] | List[Tuple[Expr, Sequence[DirichletBC], Expr, Coefficient, GenericFunc]]
        Weak form of the partial differential equations.
    adjoint(level_set_func: Function, states: List[Function]) -> List[Tuple[Expr, Sequence[DirichletBC]]] | List[Tuple[Expr, Sequence[DirichletBC], Expr, Coefficient, GenericFunc]]
        Weak form of the adjoint equations.
    cost(level_set_func: Function, states: List[Function]) -> Expr
        Cost functional.
    constraint(level_set_func: Function, states: List[Function]) -> List[Expr]
        Constraint functions.
    derivative(level_set_func: Function, states: List[Function], adjoints: List[Function]) -> Tuple[Tuple[Expr, List[Expr]], Tuple[Expr, List[Expr]]]
        Derivative components.
    bilinear_form(velocity_func: Function | Argument, test_func: Function | Argument) -> Tuple[Expr, bool]
        Bilinear form to compute the velocity field.
    """

    @abstractmethod
    def __init__(self, dim: int, domain: Mesh, space: FunctionSpace, path: Path):
        """
        Initializes the model.

        Parameters
        ----------
        dim : int
            Domain dimension.
        domain : Mesh
            Problem domain.
        space : FunctionSpace
            Space of functions for the solution and test functions.
        path : Path
            Test path to save the results.
        """

    def __init_subclass__(cls):
        cls._to_eval = {"function": {}, "quantity": {}}
        for attr in cls.__dict__.values():
            if callable(attr) and getattr(attr, "_to_eval", False):
                cls._to_eval[attr._kind][attr._name] = attr

    @abstractmethod
    def pde(
        self, level_set_func: Function
    ) -> (
        List[Tuple[Expr, Sequence[DirichletBC]]]
        | List[Tuple[Expr, Sequence[DirichletBC], Expr, Coefficient, GenericFunc]]
    ):
        """
        Weak form of the partial differential equations.

        Parameters
        ----------
        level_set_func : Function
            Level set function.

        Returns
        -------
        List[Tuple[Expr, List[DirichletBC]]] | List[Tuple[Expr, List[DirichletBC], Expr, Coefficient, GenericFunc]]
            List with elements of the form (`wk`, `bc`) or (`wk`, `bc`, `jac`, `unk`, `ini_func`),
            corresponding to linear or nonlinear problems, respectively, where

            - `wk` is the weak formulation of a state equation,
            - `bc` is a list with the dirichlet boundary conditions,
            - `jac` is the jacobian of `wk`,
            - `unk` is the unknown in the nonlinear equation,
            - `ini_func` is a callable function that defines the initial guess.
        """
        pass

    @abstractmethod
    def adjoint(
        self, level_set_func: Function, states: List[Function]
    ) -> (
        List[Tuple[Expr, Sequence[DirichletBC]]]
        | List[Tuple[Expr, Sequence[DirichletBC], Expr, Coefficient, GenericFunc]]
    ):
        """
        Weak form of the adjoint equations.

        Parameters
        ----------
        level_set_func : Function
            Level set function.
        states: List[Function]
            List of state solutions.

        Returns
        -------
        List[Tuple[Expr, List[DirichletBC]]] | List[Tuple[Expr, List[DirichletBC], Expr, Coefficient, GenericFunc]]
            List with elements of the form (`wk`, `bc`) or (`wk`, `bc`, `jac`, `unk`, `ini_func`),
            corresponding to linear or nonlinear problems, respectively, where

            - `wk` is the weak formulation of a adjoint equation,
            - `bc` is a list with the dirichlet boundary conditions,
            - `jac` is the jacobian of `wk`,
            - `unk` is the unknown in the nonlinear equation,
            - `ini_func` is a callable function that defines the initial guess.
        """
        pass

    @abstractmethod
    def cost(self, level_set_func: Function, states: List[Function]) -> Expr:
        """
        Cost functional `J`.

        Parameters
        ----------
        level_set_func : Function
            Level set function.
        states: List[Function]
            List of state solutions.

        Returns
        -------
        Expr
            UFL expresion of the cost functional.
        """
        pass

    @abstractmethod
    def constraint(
        self, level_set_func: Function, states: List[Function]
    ) -> List[Expr]:
        """
        Constraint functions `C = [C0, C1, ...]`.

        Parameters
        ----------
        level_set_func : Function
            Level set function.
        states: List[Function]
            List of state solutions.

        Returns
        -------
        List[Expr]
            List with the UFL expressions of the constraint functions.
        """
        pass

    @abstractmethod
    def derivative(
        self, level_set_func: Function, states: List[Function], adjoints: List[Function]
    ) -> Tuple[Tuple[Expr, List[Expr]], Tuple[Expr, List[Expr]]]:
        """
        Derivative components of the cost functional `J` and
        the constraint functions `C = [C0, C1, ...]`.

        Parameters
        ----------
        level_set_func : Function
            Level set function.
        states : List[Function]
            List of state solutions.
        adjoints : List[Function]
            List of adjoint solutions.

        Returns
        -------
        Tuple[Tuple[Expr, List[Expr]], Tuple[Expr, List[Expr]]]
            Two tuples

            (S0_J, [S0_C0, S0_C1, ...]), (S1_J, [S1_C0, S1_C1, ...])

            where

            `S0_J`, `S1_J` are the UFL expressions
            of the derivative components of `J`;

            `S0_Ci`, `S1_Ci` are the UFL expressions
            of the derivative components of `Ci`.
        """
        pass

    @abstractmethod
    def bilinear_form(
        self, velocity_func: Function | Argument, test_func: Function | Argument
    ) -> Tuple[Expr, bool]:
        """
        Bilinear form `B` to compute the velocity field.

        Parameters
        ----------
        velocity_func : Function | TrialFunction
            The trial-like function used in the bilinear form.
        test_func : Function | TestFunction
            The test-like function used in the bilinear form.

        Returns
        -------
        b : Expr
            The UFL expression representing the bilinear form.
        flag : bool
            Whether a homogeneous Dirichlet boundary condition
            should be applied (True) or not (False).
        """
        pass

    def __verification(self, required_attrs: List[str]) -> None:
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise NotImplementedError(f"Models must define the '{attr}' attribute.")

    @final
    def set_initial_level(self, func: Function) -> None:
        """
        Set a initial level set function as initial guess.
        """
        func_class = ObjFunc(func)
        self.ini_lvl = func_class

    @final
    def create_initial_level(
        self,
        centers: npt.NDArray[np.float64],
        radii: npt.NDArray[np.float64],
        factor: float = 1.0,
        ord: int = 2,
    ) -> None:
        """
        Creates a level set funtion to be used as initial guess,
        with ball shaped holes determined by centers and radii.

        Parameters
        ----------
        centers : npt.NDArray[np.float64]
            Array of center coordinates
            of shape (N, 2) or (N, 3).
        radii : npt.NDArray[np.float64]
            Array of radii of shape (N,).
        factor : float, default=1.0
            Scaling factor, positive or negative.
            The default is 1.0, which implies a domain with holes.
        ord : int, default=2
            Order of norm. Positive integer
            grater than 1 or infinity `np.inf`
            The dafault is 2, which implies Euclidean norm.
        """

        self.ini_lvl = InitialLevel(centers, radii, factor, ord)

    @final
    def add_paths_to_ini_lvl(self, list_of_paths):
        self.ini_lvl.add_paths(list_of_paths)

    @final
    def save_initial_level(self, comm: MPI.Comm) -> None:
        """
        Saves the initial level set function is a xdmf file.

        Parameters
        ----------
        comm : MPI.Comm
            Communicator.
        """

        self.__verification(["domain", "path"])
        self.ini_lvl.save(comm, self.domain, self.path)

    @final
    def _get_initial_level(self):
        """
        Return the callable initial level set function.
        """
        return self.ini_lvl.func

    @final
    def runDP(
        self,
        niter: int = 100,
        dfactor: float = 1e-2,
        lv_time: Tuple[float, float] = (1e-3, 1e-1),
        lv_iter: Tuple[int, int] = (8, 16),
        smooth: bool = False,
        reinit_step: int | bool = False,
        reinit_pars: Tuple[int, float] = (8, 1e-1),
        start_to_check: int = 30,
        ctrn_tol: float = 1e-2,
        lgrn_tol: float = 1e-2,
        cost_tol: float = 1e-2,
        prev: int = 10,
        random_pars: Tuple[int, float] = (1, 0.05),
    ) -> None:
        """
        This method implements Data Parallelism.

        Parameters
        ----------
        niter : int, default=100
            The number of iterations.
        dfactor : float, default=1e-2
            A positive scaling factor applied to the inverse of
            the derivative norm to estimate the final integration
            time of the level set equation.
            Values <= 1 are recommended.
        lv_time : Tuple[float, float], default=(1e-3, 1e-1)
            A tuple with the minimum and maximum time allowed
            for the integration of the level set equation.
        lv_iter : Tuple[int, int], default=(8, 16)
            A tuple with the minimum and maximum number of
            iterations allowed for the integration of the level
            set equation.
        smooth : bool, default=False
            If True, a diffusion term is added to the level set
            equation; if False, the equation is solved without
            diffusion.
        reinit_step : int or bool, default=False
            A positive integer to apply the reinitialization
            method when the condition
                iter>start_to_check and reinit_step%iter == 0
            holds. If False, no reinitialization is applied.
        reinit_pars : Tuple[int, float], default=(8, 1e-1)
            A tuple with the number of iterations and the final
            time for the integration of the reinitialization
            equation. The argument reinit_step must be a
            positive integer.
        start_to_check : int, default=30,
            A positive integer to verify the stopping condition
            when current iteration > start_to_check.
        ctrn_tol : float, default=1e-2,
            Tolerance for the error constraint.
        lgrn_tol : float, default=1e-2,
            Tolerance for the relative difference
            of the <prev>-th previous Lagrangian values.
        cost_tol : float, default=1e-2,
            Tolerance for the relative difference
            of the <prev>-th previous cost values.
        prev : int, default=10,
            Number of previous values to verify the tolerance
            of the Lagrangian and cost functionals.
        random_pars : Tuple[int, float], default=(1, 0.05)
            Seed and noise level in the line search method.
        """

        self.__verification(["dim", "domain", "space", "path"])

        self.phi = runDP(
            self,
            niter,
            reinit_step,
            reinit_pars,
            dfactor,
            lv_time,
            lv_iter,
            smooth,
            start_to_check,
            ctrn_tol,
            lgrn_tol,
            cost_tol,
            prev,
            random_pars,
        )

    @final
    def runTP(
        self,
        niter: int = 100,
        dfactor: float = 1e-2,
        lv_time: Tuple[float, float] = (1e-3, 1e-1),
        lv_iter: Tuple[int, int] = (8, 16),
        smooth: bool = False,
        reinit_step: int | bool = False,
        reinit_pars: Tuple[int, float] = (8, 1e-1),
        start_to_check: int = 30,
        ctrn_tol: float = 1e-2,
        lgrn_tol: float = 1e-2,
        cost_tol: float = 1e-2,
        prev: int = 10,
        random_pars: Tuple[int, float] = (1, 0.05),
    ) -> None:
        """
        This method implements Data Parallelism.

        Parameters
        ----------
        niter : int, default=100
            The number of iterations.
        dfactor : float, default=1e-2
            A positive scaling factor applied to the inverse of
            the derivative norm to estimate the final integration
            time of the level set equation.
            Values <= 1 are recommended.
        lv_time : Tuple[float, float], default=(1e-3, 1e-1)
            A tuple with the minimum and maximum time allowed
            for the integration of the level set equation.
        lv_iter : Tuple[int, int], default=(8, 16)
            A tuple with the minimum and maximum number of
            iterations allowed for the integration of the level
            set equation.
        smooth : bool, default=False
            If True, a diffusion term is added to the level set
            equation; if False, the equation is solved without
            diffusion.
        reinit_step : int or bool, default=False
            A positive integer to apply the reinitialization
            method when the condition
                iter>start_to_check and reinit_step%iter == 0
            holds. If False, no reinitialization is applied.
        reinit_pars : Tuple[int, float], default=(8, 1e-1)
            A tuple with the number of iterations and the final
            time for the integration of the reinitialization
            equation. The argument reinit_step must be a
            positive integer.
        start_to_check : int, default=30,
            A positive integer to verify the stopping condition
            when current iteration > start_to_check.
        ctrn_tol : float, default=1e-2,
            Tolerance for the error constraint.
        lgrn_tol : float, default=1e-2,
            Tolerance for the relative difference
            of the <prev>-th previous Lagrangian values.
        cost_tol : float, default=1e-2,
            Tolerance for the relative difference
            of the <prev>-th previous cost values.
        prev : int, default=10,
            Number of previous values to verify the tolerance
            of the Lagrangian and cost functionals.
        random_pars : Tuple[int, float], default=(1, 0.05)
            Seed and noise level in the line search method.
        """

        self.__verification(["dim", "domain", "space", "path"])

        self.phi = runTP(
            self,
            niter,
            reinit_step,
            reinit_pars,
            dfactor,
            lv_time,
            lv_iter,
            smooth,
            start_to_check,
            ctrn_tol,
            lgrn_tol,
            cost_tol,
            prev,
            random_pars,
        )

    @final
    def runMP(
        self,
        sub_comm: MPI.Comm,
        niter: int = 100,
        dfactor: float = 1e-2,
        lv_time: Tuple[float, float] = (1e-3, 1e-1),
        lv_iter: Tuple[int, int] = (8, 16),
        smooth: bool = False,
        reinit_step: int | bool = False,
        reinit_pars: Tuple[int, float] = (8, 1e-1),
        start_to_check: int = 30,
        ctrn_tol: float = 1e-2,
        lgrn_tol: float = 1e-2,
        cost_tol: float = 1e-2,
        prev: int = 10,
        random_pars: Tuple[int, float] = (1, 0.05),
    ) -> None:
        """
        This method implements Mixed Parallelism.

        Parameters
        ----------
        niter : int, default=100
            The number of iterations.
        dfactor : float, default=1e-2
            A positive scaling factor applied to the inverse of
            the derivative norm to estimate the final integration
            time of the level set equation.
            Values <= 1 are recommended.
        lv_time : Tuple[float, float], default=(1e-3, 1e-1)
            A tuple with the minimum and maximum times allowed
            for the integration of the level set equation.
        lv_iter : Tuple[int, int], default=(8, 16)
            A tuple with the minimum and maximum number of
            iterations allowed for the integration of the level
            set equation.
        smooth : bool, default=False
            If True, a diffusion term is added to the level set
            equation; if False, the equation is solved without
            diffusion.
        reinit_step : int or bool, default=False
            A positive integer to apply the reinitialization
            method when the condition
                iter>start_to_check and reinit_step%iter == 0
            holds. If False, no reinitialization is applied.
        reinit_pars : Tuple[int, float], default=(8, 1e-1)
            A tuple with the number of iterations and the final
            time for the integration of the reinitialization
            equation. The argument reinit_step must be a
            positive integer.
        start_to_check : int, default=30,
            A positive integer to verify the stopping condition
            when current iteration > start_to_check.
        ctrn_tol : float, default=1e-2,
            Tolerance for the error constraint.
        lgrn_tol : float, default=1e-2,
            Tolerance for the relative difference
            of the <prev>-th previous Lagrangian values.
        cost_tol : float, default=1e-2,
            Tolerance for the relative difference
            of the <prev>-th previous cost values.
        prev : int, default=10,
            Number of previous values to verify the tolerance
            of the Lagrangian and cost functionals.
        random_pars : Tuple[int, float], default=(1, 0.05)
            Seed and noise level in the line search method.
        """

        self.__verification(["dim", "domain", "space", "path"])

        self.phi = runMP(
            sub_comm,
            self,
            niter,
            reinit_step,
            reinit_pars,
            dfactor,
            lv_time,
            lv_iter,
            smooth,
            start_to_check,
            ctrn_tol,
            lgrn_tol,
            cost_tol,
            prev,
            random_pars,
        )

    @final
    def phifem_run(
        self,
        niter: int = 100,
        dfactor: float = 1e-2,
        lv_time: Tuple[float, float] = (1e-3, 1e-1),
        lv_iter: Tuple[int, int] = (8, 16),
        smooth: bool = False,
        reinit_step: int | bool = False,
        reinit_pars: Tuple[int, float] = (8, 1e-1),
        start_to_check: int = 30,
        ctrn_tol: float = 1e-2,
        lgrn_tol: float = 1e-2,
        cost_tol: float = 1e-2,
        prev: int = 10,
        random_pars: Tuple[int, float] = (1, 0.05),
    ) -> None:
        """
        This method implements Data Parallelism.

        Parameters
        ----------
        niter : int, default=100
            The number of iterations.
        dfactor : float, default=1e-2
            A positive scaling factor applied to the inverse of
            the derivative norm to estimate the final integration
            time of the level set equation.
            Values <= 1 are recommended.
        lv_time : Tuple[float, float], default=(1e-3, 1e-1)
            A tuple with the minimum and maximum time allowed
            for the integration of the level set equation.
        lv_iter : Tuple[int, int], default=(8, 16)
            A tuple with the minimum and maximum number of
            iterations allowed for the integration of the level
            set equation.
        smooth : bool, default=False
            If True, a diffusion term is added to the level set
            equation; if False, the equation is solved without
            diffusion.
        reinit_step : int or bool, default=False
            A positive integer to apply the reinitialization
            method when the condition
                iter>start_to_check and reinit_step%iter == 0
            holds. If False, no reinitialization is applied.
        reinit_pars : Tuple[int, float], default=(8, 1e-1)
            A tuple with the number of iterations and the final
            time for the integration of the reinitialization
            equation. The argument reinit_step must be a
            positive integer.
        start_to_check : int, default=30,
            A positive integer to verify the stopping condition
            when current iteration > start_to_check.
        ctrn_tol : float, default=1e-2,
            Tolerance for the error constraint.
        lgrn_tol : float, default=1e-2,
            Tolerance for the relative difference
            of the <prev>-th previous Lagrangian values.
        cost_tol : float, default=1e-2,
            Tolerance for the relative difference
            of the <prev>-th previous cost values.
        prev : int, default=10,
            Number of previous values to verify the tolerance
            of the Lagrangian and cost functionals.
        random_pars : Tuple[int, float], default=(1, 0.05)
            Seed and noise level in the line search method.
        """

        self.__verification(["dim", "domain", "space", "path"])

        self.phi = phifem_run(
            self,
            niter,
            reinit_step,
            reinit_pars,
            dfactor,
            lv_time,
            lv_iter,
            smooth,
            start_to_check,
            ctrn_tol,
            lgrn_tol,
            cost_tol,
            prev,
            random_pars,
        )


def _register(kind, name):
    def decorator(func):
        func._to_eval = True
        func._kind = kind
        func._name = name
        return func

    return decorator


def func_to_eval(name):
    # decorator for functions
    # to be evaluated
    return _register("function", name)


def qtty_to_eval(name):
    # decorator for quantities
    # to be evaluated
    return _register("quantity", name)


class ObjFunc:
    """
    Class to save and pass a level set function
    """

    def __init__(self, func):
        self.func = func


class Subdomain:
    """
    This class creates an indicator function
    from a list o inequalities that determine
    a subdomain.

    Attributtes
    -----------
    domain : Mesh
        Problem domain.
    cond_func : Callable[[npt.NDArray[np.float64]], List[Expr]]
        Callable function that returns a list of
        inequalities of the form g(x) > 0.
        For example, the subdomain
        {(x,y) | 0.42 < y < 0.58, 1.95 < x}
        is represented with the function
        ```
        def sub_domain(x):
            ineqs = [
                x[1] - 0.42,
                0.58 - x[1],
                x[0] - 1.95
            ]
            return ineqs
        ```

    Methods
    -------
    expression() -> Expr
        Returns the ufl expresion correspoding
        to the indicator function with subdomain
        determinied by cond_func.
    """

    def __init__(
        self, domain: Mesh, cond_func: Callable[[npt.NDArray[np.float64]], List[Expr]]
    ) -> None:
        self.domain = domain
        self.cond_func = cond_func

    def expression(self) -> Expr:
        """
        Returns the ufl expresion correspoding
        to the indicator function with subdomain
        determinied by cond_func.
        The subdomains determined by each inequality
        are intercepted by multiplying the conditionals.
        """
        x = SpatialCoordinate(self.domain)
        conditions = self.cond_func(x)
        chi = 1.0

        for cond in conditions:
            chi *= conditional(gt(cond, 0.0), 1.0, 0.0)

        return chi


def region_of(domain: Mesh):
    """
    Wrapper for the Subdomain class.
    """

    def wrapper(func):
        return Subdomain(domain, func)

    return wrapper


class PPL:
    """
    This class implements the Perturbed Proximal Lagrangian method
    developed in the paper "A new Lagrangian-based ﬁrst-order
    method for nonconvex constrained optimization" (Jong Gwang Kim, 2023).

    Attributes
    ----------

    Methods
    -------
    run(cost: float, constraints: List[float]) -> npt.NDArray[np.float64]
        Runs one iteration of the method.
    _lagrangian(cost: float) -> float
        Returns the value of the Lagrangian functional.
    see() -> str
        Returns a string with some variable values.
    """

    def __init__(self, n: int, ini_cost: float, ini_ctrs: List[float]) -> None:
        """
        Parameters
        ----------
        n : int
            Number of constraint functions.
        ini_cost : float
            Cost value at the first iteration.
        ini_ctrs : float
            Constraint values at the first iteration.
        """

        self.n = n
        self.ones = np.ones(n)
        self.lm = np.repeat(0.0, n)
        self.mu = np.repeat(0.0, n)
        self.zs = np.repeat(0.0, n)
        self.alpha = 2000.0
        self.beta = 0.5
        self.rho = self.alpha / (1.0 + self.alpha * self.beta)
        self.r = 0.999
        self.dl = 0.5
        self.Lg = ini_cost
        self.ct = np.array(ini_ctrs)

        self.list_Lg = [self.Lg]
        self.list_lm = [self.lm]
        self.list_mu = [self.mu]
        self.list_zs = [self.zs]
        self.list_dl = [self.dl]
        self.list_ct = [self.ct]

    def run(self, cost: float, constraints: List[float]) -> npt.NDArray[np.float64]:
        """
        Runs one iteration of the method.
        """

        self.ct = np.array(constraints)
        self.lmu = self.lm - self.mu
        self.mu = self.mu + self.dl * self.lmu / (np.inner(self.lmu, self.lmu) + 1)
        self.lm = self.mu + self.rho * (self.ct - self.ones)
        self.zs = (self.lm - self.mu) / self.alpha
        self.dl = self.r * self.dl

        self.Lg = self._lagrangian(cost)

        self.list_Lg.append(self.Lg)
        self.list_lm.append(self.lm)
        self.list_mu.append(self.mu)
        self.list_zs.append(self.zs)
        self.list_dl.append(self.dl)
        self.list_ct.append(self.ct)

        return self.lm[:]

    def _lagrangian(self, cost: float) -> float:
        """
        Returns the value of the Lagrangian functional.
        """
        return (
            cost
            + np.inner(self.lm, self.ct - self.ones - self.zs)
            + np.inner(self.mu, self.zs)
            + self.alpha * np.inner(self.zs, self.zs) / 2.0
            - self.beta * np.inner(self.lm - self.mu, self.lm - self.mu) / 2.0
        )

    def see(self) -> str:
        """
        Returns a string with some variable values.
        """
        return (
            f"lagr = {self.Lg:.4f} | "
            f"lm = {', '.join(f'{val:.4f}' for val in self.lm)} | "
            f"mu = {', '.join(f'{val:.4f}' for val in self.mu)} | ",
        )[0]


class Save:
    """
    This class saves scalar numerical results:
    cost values, derivative norms, Lagrange multipliers, etc.

    Attributes
    ----------
    cost : List[float]
        List with cost values.
    nder : List[float]
        List with the derivative norms.
    ppl_obj : PPL
        Instance of PPL.
    times : List[float]
        List with the assembly and resolution times.

    Methods
    -------
    add(cost: float, nder: float) -> None
        Adds cost and norm derivative values.
    add_times(assembly: float, resolution: float) -> None
        Adds assemply and resolution times.
    add_ppl(ppl_obj: PPL) -> None
        Adds a PPL object.
    save(path: Path) -> None
        Saves the data.
    """

    def __init__(self) -> None:
        self.cost = []
        self.nder = []
        self.quantities = []
        self.q_names = None
        self.ppl_obj = None

    def add(self, cost: float, nder: float) -> None:
        self.cost.append(cost)
        self.nder.append(nder)

    def add_quantities(self, qs: List[float]) -> None:
        self.quantities.append(qs)

    def add_qtty_names(self, names):
        self.q_names = names

    def add_times(self, assembly: float, resolution: float) -> None:
        self.times = [assembly, resolution]

    def add_ppl(self, ppl_obj: PPL) -> None:
        self.ppl_obj = ppl_obj

    def save(self, path: Path) -> None:
        data_to_save = None

        if self.ppl_obj is None:
            data_to_save = {
                "cost": np.array(self.cost),
                "nder": np.array(self.nder),
                "times": self.times,
            }

        else:
            data_to_save = {
                "cost": np.array(self.cost),
                "ctrs": np.array(self.ppl_obj.list_ct),
                "nder": np.array(self.nder),
                "Lg": np.array(self.ppl_obj.list_Lg),
                "lm": np.array(self.ppl_obj.list_lm),
                "mu": np.array(self.ppl_obj.list_mu),
                "z": np.array(self.ppl_obj.list_zs),
                "delta": np.array(self.ppl_obj.list_dl),
                "alpha": np.array(self.ppl_obj.alpha),
                "beta": np.array(self.ppl_obj.beta),
                "rho": np.array(self.ppl_obj.rho),
                "r": np.array(self.ppl_obj.r),
                "times": self.times,
            }

        if len(self.quantities) > 0:
            qtts = np.array(self.quantities)
            for i in range(len(self.q_names)):
                data_to_save[self.q_names[i]] = qtts[:, i]

        np.savez(path / f"{dat_name}.npz", **data_to_save)


class InitialLevel:
    """
    Creates the initial level set function
    with ball shaped holes determined by
    centers and radii.

    Attributes
    ----------
    centers : npt.NDArray[np.float64]
        Array of center coordinates.
    radii : npt.NDArray[np.float64]
        Array of radii.
    dim : int
        Problem dimension.
    factor : float
        Scaling factor.
    ord : int
        Order of the norm.

    Methods
    -------
    func(x) :
        Callable function to be interpolated
        by a dolfinx function.
    save() :
        Interpolates func and saves it into
        a xdmf file.
    """

    def __init__(
        self,
        centers: npt.NDArray[np.float64],
        radii: npt.NDArray[np.float64],
        factor: float = 1.0,
        ord: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        centers : npt.NDArray[np.float64]
            Array of center coordinates
            of shape (N, 2) or (N, 3).
        radii : npt.NDArray[np.float64]
            Array of radii of shape (N,).
        factor : float
            Scaling factor, positive or negative.
        ord : int
            Order of norm. Positive integer
            grater than 1 or infinity (np.inf).
        """
        self.centers = centers
        self.radii = radii
        self.dim = self.centers.shape[1]
        self.factor = factor
        self.ord = ord
        self.paths = False

    def func(self, x):
        """
        Callable function to be interpolated
        by a dolfinx function.
        """

        xT = (x[: self.dim].T)[None, :, :]
        # comps = (self.centers[:, None, :] - xT)**2
        # norms = np.sqrt(np.sum(comps, axis = 2))
        norms = Norm(self.centers[:, None, :] - xT, ord=self.ord, axis=2)
        distances = self.radii[:, None] - norms
        values = np.max(distances, axis=0)

        return self.factor * values

    def add_paths(self, list_of_paths):
        for ini, end, r in list_of_paths:
            distance = Norm(end - ini)
            np.linspace(ini, end, distance % r + 1)
            # pendiente

    def save(self, comm, domain, save_path):
        """
        Interpolates func and saves it into
        a xdmf file.
        """

        space = functionspace(domain, ("CG", 1))
        intp_func = interpolate([self.func], space, name="phi0")
        save_functions(comm, domain, intp_func, save_path / f"{ini_name}.xdmf")


def get_funcs_from(
    space: FunctionSpace, values: npt.NDArray[np.float64]
) -> List[Function]:
    """
    Returns a list of functions of the given
    values and defined on the given space.
    """
    funcs = [Function(space) for _ in range(len(values))]
    for f, v in zip(funcs, values):
        f.x.array[:] = v

    return funcs


def space_interpolation(from_space, funcs, to_space):

    to_domain = to_space.mesh
    new_funcs = [Function(to_space) for _ in range(len(funcs))]
    dim = to_domain.topology.dim
    fine_mesh_cell_map = to_domain.topology.index_map(dim)
    num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interp_data = create_interpolation_data(to_space, from_space, cells, padding=1e-14)

    for nf, f in zip(new_funcs, funcs):
        nf.interpolate_nonmatching(f, cells, interpolation_data=interp_data)

    return new_funcs


def all_connectivities(domain: Mesh) -> None:
    """
    Creates all basic connectivities.
    """
    topology = domain.topology
    dim = topology.dim

    pairs = [(dim, dim), (0, dim), (dim, 0), (1, dim), (dim, 1)]

    if dim == 3:
        pairs += [(2, dim), (dim, 2)]

    for d0, d1 in pairs:
        topology.create_connectivity(d0, d1)


def dirichlet_extension_from_bcs(domain, space, list_bcs):

    dx = Measure("dx", domain=domain)
    n = len(list_bcs)
    ext = [Function(space) for _ in range(n)]
    for i in range(n):
        ext[i].name = "g" + str(i)

    h = Function(space)
    h.x.array[:] = 0
    eta = TrialFunction(space)
    zeta = TestFunction(space)
    a = inner(grad(eta), grad(zeta)) * dx
    L = h * zeta * dx  # Right-hand side is zero

    for bcs, g in zip(list_bcs, ext):
        basic_solver(form(a), form(L), bcs, g)

    return ext


def dirichlet_extension(
    domain: Mesh, space: FunctionSpace, funcs: List[Function]
) -> List[Function]:
    """
    Compute the Dirichlet extensions of a
    list of functions.

    Parameters
    ----------
    domain : Mesh
        Problem domain.
    space : FunctionSpace
        Space of functions where the extensions are computed.
    funcs : List[Function]
        List of functions to be used as Dirichlet conditions.
        These functions are defined on the domain.

    Returns
    -------
    extensions : List[Function]
        The Dirichlet extensions of the input functions.
    """
    dx = Measure("dx", domain=domain)
    nbr_fcs = len(funcs)
    extensions = [Function(space) for _ in range(nbr_fcs)]
    for i in range(nbr_fcs):
        extensions[i].name = "g" + str(i)

    dim = domain.topology.dim
    boundary_dofs = locate_dofs_topological(
        space, dim - 1, exterior_facet_indices(domain.topology)
    )

    h = Function(space)
    h.x.array[:] = 0
    eta = TrialFunction(space)
    zeta = TestFunction(space)
    a = inner(grad(eta), grad(zeta)) * dx
    L = dot(zeta, h) * dx  # Right-hand side is zero

    for f, g in zip(funcs, extensions):
        basic_solver(form(a), form(L), [dirichletbc(f, boundary_dofs)], g)

    return extensions


def build_gmsh_model_2d(
    vertices,
    boundary_parts,
    mesh_size,
    holes=None,
    curves=None,
    filename="domain.msh",
    plot=False,
    quad=False,
):
    """
    Parameters
    ---------
        vertices :
            ...
        boundary_parts :
            ...
        mesh_size :
            ...
        holes :
            ...
        curve :
            ...
        path :
            ...
        plot :
            ...

    Returns
    -------
        ...

    """
    rank_to_build = 0
    # Tags:
    # point tags = 1, 2, ..., K
    # line tags = 1, 2, ..., K
    if rank == rank_to_build:
        gmsh.initialize()
        gmsh.model.add("2D_Mesh")
        gmsh.option.setNumber("General.Terminal", 0)

        K = len(vertices)
        line_tags = [k for k in range(1, K + 1)]
        curve_tag = 1
        surface_tag = 1

        # boundary lines
        gmsh.model.geo.addPoint(
            x=vertices[0][0], y=vertices[0][1], z=0.0, meshSize=0.0, tag=1
        )
        for k in range(1, K):
            gmsh.model.geo.addPoint(
                x=vertices[k][0], y=vertices[k][1], z=0.0, meshSize=0.0, tag=k + 1
            )
            gmsh.model.geo.addLine(startTag=k, endTag=k + 1, tag=k)
        gmsh.model.geo.addLine(startTag=K, endTag=1, tag=K)
        gmsh.model.geo.addCurveLoop(curveTags=line_tags, tag=curve_tag)

        last_tag = K

        if holes is not None:
            list_holes = []
            curve_hole_tag = 2  # 2, 3, ...
            for hole in holes:
                list_holes.append(curve_hole_tag)
                J = len(hole)
                hole_tags = [last_tag + j for j in range(1, J + 1)]
                gmsh.model.geo.addPoint(
                    x=hole[0][0], y=hole[0][1], z=0.0, meshSize=0.0, tag=last_tag + 1
                )
                for j in range(1, J):
                    gmsh.model.geo.addPoint(
                        x=hole[j][0],
                        y=hole[j][1],
                        z=0.0,
                        meshSize=0.0,
                        tag=last_tag + j + 1,
                    )
                    gmsh.model.geo.addLine(
                        startTag=last_tag + j, endTag=last_tag + j + 1, tag=last_tag + j
                    )
                gmsh.model.geo.addLine(
                    startTag=last_tag + J, endTag=last_tag + 1, tag=last_tag + J
                )
                gmsh.model.geo.addCurveLoop(curveTags=hole_tags, tag=curve_hole_tag)
                last_tag = last_tag + J  # update last tag
                curve_hole_tag = curve_hole_tag + 1

            # create surface
            gmsh.model.geo.addPlaneSurface([curve_tag] + list_holes, tag=surface_tag)
        else:
            # create surface
            gmsh.model.geo.addPlaneSurface([curve_tag], tag=surface_tag)

        # --------------------------------------------
        # Code for add a interior curve that define
        # a subdomain. This part is used to generate
        # data for inverse problems.
        if curves is not None:
            for curve in curves:
                L = len(curve)
                gmsh.model.geo.addPoint(
                    x=curve[0][0], y=curve[0][1], z=0.0, meshSize=0.1, tag=last_tag + 1
                )
                for l in range(1, L):
                    gmsh.model.geo.addPoint(
                        x=curve[l][0],
                        y=curve[l][1],
                        z=0.0,
                        meshSize=0.1,
                        tag=last_tag + l + 1,
                    )
                    gmsh.model.geo.addLine(
                        startTag=last_tag + l, endTag=last_tag + l + 1, tag=last_tag + l
                    )
                gmsh.model.geo.addLine(
                    startTag=last_tag + L, endTag=last_tag + 1, tag=last_tag + L
                )

                # We have to synchronize before embedding the lines
                gmsh.model.geo.synchronize()

                gmsh.model.mesh.embed(
                    dim=1,
                    tags=[last_tag + l for l in range(1, L + 1)],
                    inDim=2,
                    inTag=surface_tag,
                )
                last_tag = last_tag + L

        # if curve is not None:
        #     L = len(curve)
        #     gmsh.model.geo.addPoint(
        #         x = curve[0][0],
        #         y = curve[0][1],
        #         z = 0.,
        #         meshSize = 0.1,
        #         tag = last_tag + 1
        #     )
        #     for l in range(1, L):
        #         gmsh.model.geo.addPoint(
        #             x = curve[l][0],
        #             y = curve[l][1],
        #             z = 0.,
        #             meshSize = 0.1,
        #             tag = last_tag + l + 1
        #         )
        #         gmsh.model.geo.addLine(
        #             startTag = last_tag + l,
        #             endTag = last_tag + l + 1,
        #             tag = last_tag + l
        #         )
        #     gmsh.model.geo.addLine(
        #         startTag = last_tag + L,
        #         endTag = last_tag + 1,
        #         tag = last_tag + L
        #     )

        #     # We have to synchronize before embedding the lines
        #     gmsh.model.geo.synchronize()

        #     gmsh.model.mesh.embed(
        #         dim = 1,
        #         tags = [last_tag + l for l in range(1, L + 1)],
        #         inDim = 2,
        #         inTag = surface_tag
        #     )

        gmsh.model.geo.synchronize()

        # physical groups
        # domain
        gmsh.model.addPhysicalGroup(dim=2, tags=[surface_tag], tag=1, name="domain")
        # boundary
        if boundary_parts:
            # add mark to subsets of the boundary
            for facets, marker, name in boundary_parts:
                gmsh.model.addPhysicalGroup(dim=1, tags=facets, tag=marker, name=name)
        else:
            # add mark to the whole boundary
            gmsh.model.addPhysicalGroup(dim=1, tags=line_tags, tag=1, name="boundary")

        # size mesh
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        if quad:
            gmsh.model.mesh.setRecombine(2, surface_tag)

        gmsh.model.mesh.generate(dim=2)

        if quad:
            nbr_triangles = len(gmsh.model.mesh.getElementsByType(3)[0])
        else:
            nbr_triangles = len(gmsh.model.mesh.getElementsByType(2)[0])

        # Plot the mesh
        if plot:
            gmsh.fltk.run()

        # Write the mesh in a *.msh file
        gmsh.write(str(filename))
        gmsh.clear()
        gmsh.finalize()

    else:

        nbr_triangles = None

    nbr_triangles = comm.bcast(nbr_triangles, root=rank_to_build)

    return nbr_triangles


def interpolate(funcs, to_space, name="f"):

    n = len(funcs)
    new_funcs = [Function(to_space) for _ in range(n)]

    if n == 1:
        new_funcs[0].name = name
        new_funcs[0].interpolate(funcs[0])
    else:
        for i in range(n):
            new_funcs[i].name = name + str(i)
            new_funcs[i].interpolate(funcs[i])

    return new_funcs


def save_domain(comm, domain, filename, facet_tags=None):

    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        if facet_tags:
            xdmf.write_meshtags(facet_tags, domain.geometry)


def save_initial(comm, initial_guess, domain, filename):

    initial_level = Function(functionspace(domain, ("CG", 1)))
    initial_level.name = "phi"
    initial_level.interpolate(InitialLevel(*initial_guess).func)

    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(initial_level, 0)


def save_functions(comm, domain, funcs, filename):
    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        for f in funcs:
            xdmf.write_function(f, 0)


def save_initial_level(comm, domain, func, filename):

    space = functionspace(domain, ("CG", 1))
    intp_func = interpolate([func], space, name="phi")
    save_functions(comm, domain, intp_func, filename)


def read_gmsh(
    filename: Path, comm: MPI.Comm, dim: int
) -> Tuple[Mesh, MeshTags_int32, MeshTags_int32]:
    """
    This is a simple wrapper around `dolfinx.io.gmshio.read_from_msh`
    to read a mesh from a Gmsh `.msh` file.

    Parameters
    ----------
    filename : Path
        Path to the Gmsh `.msh` file.
    comm : MPI.Comm
        The MPI communicator.
    dim : int
        The geometric dimension of the mesh.

    Returns
    -------
    mesh : Mesh
        The mesh object.
    cell_tags : MeshTags_int32
        Integer tags associated with mesh cells (top-dimensional entities).
    facet_tags : MeshTags_int32
        Integer tags associated with mesh facets (co-dimension 1 entities).
    """
    return gmshio.read_from_msh(filename, comm=comm, gdim=dim)


def create_domain_2d_DP(
    vertices: npt.NDArray[np.float64],
    boundary_parts: List[Tuple[Sequence[int], int, str]],
    mesh_size: float,
    holes: List[npt.NDArray[np.float64]] = None,
    curve: List[npt.NDArray[np.float64]] = None,
    path: Path = Path(""),
    plot: bool = False,
) -> Tuple[Mesh, int, MeshTags_int32]:
    """
    This function creates a polygonal domain with holes
    and interior closed curves for data parallelism (DP).
    The finite elements are triangles.

    Parameters
    ----------
    vertices : npt.NDArray[np.float64]
        Nx2 numpy array of vertices to define the polygonal domain.
    boundary_parts : List[Tuple[Sequence[int], int, str]]
        List of tuples of the form
        (`Array`, `Mark`, `Name`)
        to define boundary subsets.
        `Array` contains the indices (starting at 1) of the
        boundary faces.
        `Mark` is a representative integer grater then 0
        to identify the subset.
        `Name` is just a tag name for the subset.
    mesh_size : float
        Representative mesh size for Gmsh.
    holes : List[npt.NDArray[np.float64]]
        List of Nx2 numpy arrays to define holes.
    curve : List[npt.NDArray[np.float64]]
        List of Nx2 numpy arrays to define interior curves.
    path : Path, default=Path("")
        Test path.
    plot : bool, default=False
        Flag to plot the mesh.

    Returns
    -------
    domain : Mesh
        Problem domain.
    nbr_triangles : int
        Number of finite elements (triangles).
    facet_tags : MeshTags_int32
        Integer tags associated with mesh facets (co-dimension 1 entities).
    """

    filename = path / f"{dom_name}.msh"
    nbr_triangles = build_gmsh_model_2d(
        vertices, boundary_parts, mesh_size, holes, curve, filename, plot
    )

    # To read and save the mesh (.xmdf format) =====================
    comm.barrier()
    domain, _, facet_tags = read_gmsh(filename, comm, 2)
    all_connectivities(domain)
    save_domain(comm, domain, path / f"{dom_name}.xdmf", facet_tags)
    # ==============================================================

    # Plot the distributed mesh
    if plot:
        plot_domain(domain, f"rank = {rank}")

    return domain, nbr_triangles, facet_tags


def create_domain_2d_TP(
    vertices, boundary_parts, mesh_size, holes=None, curve=None, path="", plot=False
):
    """
    Create a polygonal domain with holes
    and an interior closed curve for
    task parallelism (TP).
    """
    filename = Path(path) / f"{dom_name}.msh"
    nbr_triangles = build_gmsh_model_2d(
        vertices, boundary_parts, mesh_size, holes, curve, filename, plot
    )

    # Read and save (.xmdf format) the mesh ===============================
    comm.barrier()
    domain, _, facet_tags = read_gmsh(filename, comm_self, 2)
    all_connectivities(domain)
    if rank == 0:
        save_domain(comm_self, domain, path / f"{dom_name}.xdmf", facet_tags)
    # =====================================================================

    # Plot the identical meshes
    if plot:
        plot_domain(domain, f"rank = {rank}")

    return domain, nbr_triangles, facet_tags


def create_domain_2d_MP(
    sub_comm: MPI.Comm,
    color: int,
    vertices,
    boundary_parts,
    mesh_size,
    holes=None,
    curve=None,
    path="",
    plot=False,
):
    """
    Create a polygonal domain with holes
    and an interior closed curve for
    mix parallelism (MP).
    """

    filename = Path(path) / f"{dom_name}.msh"
    nbr_triangles = build_gmsh_model_2d(
        vertices, boundary_parts, mesh_size, holes, curve, filename, plot
    )

    # Read and save (.xmdf format) the mesh ================================
    comm.barrier()
    domain, _, facet_tags = read_gmsh(filename, sub_comm, 2)
    all_connectivities(domain)
    if color == 0:
        save_domain(sub_comm, domain, path / f"{dom_name}.xdmf", facet_tags)
    # ======================================================================

    # Plot the identically distributed meshes
    if plot:
        plot_domain(domain, f"rank = {rank}")

    return domain, nbr_triangles, facet_tags


def print0(i, cost, const_vals, der_norm, steps, extra=""):
    print(
        f"i = {i:3.0f} | "
        f"cost = {cost:.4f} | "
        f"cstr = {', '.join(f'{abs(v):.4f}' for v in const_vals)} | "
        f"nder = {der_norm:.4f} | "
        f"steps = {steps:2.0f} | " + extra
    )


def print1(i, cost, der_norm, steps, extra=""):
    print(
        f"i = {i:3.0f} | "
        f"cost = {cost:.6f} | "
        f"nder = {der_norm:.4f} | "
        f"steps = {steps:2.0f} | " + extra
    )


def assemble_mtx(ufl_form, bcs=[]):
    return assemble_matrix(form(ufl_form), bcs)


def eval(formula):
    return assemble_scalar(form(formula))


def const(domain, value):
    return Constant(domain, PETSc.ScalarType(value))


def volume(domain, comm):
    """
    Return the area (2D) or volume (3D).
    """
    dx = Measure("dx", domain=domain)
    local = eval(const(domain, 1) * dx)
    total = comm.allreduce(local, op=MPI.SUM)
    return total


def nbr_fems(domain, dim, comm):
    """
    Return the number of finite elements.
    """
    local = domain.topology.index_map(dim).size_local
    total = comm.allreduce(local, op=MPI.SUM)
    return total


def get_diam2(dim, vol, nfems):
    if dim == 2:
        return 4.0 * vol / nfems / np.sqrt(3.0)
    elif dim == 3:
        return (6.0 * np.sqrt(2.0) * vol / nfems) ** (2.0 / 3.0)
    else:
        raise ValueError(f"> Unsupported dimension: {dim}.")


def create_space(
    domain: Mesh, family: str, rank: int | tuple, degree: int = 1
) -> FunctionSpace:
    """
    Wrapper to create a function space.

    Parameters
    ----------
    domain : Mesh
        Problem domain.
    family : str
        The finite element type.
    rank : int | tuple
        - 0 for scalar functions.
        - An integer `n` for vector-valued functions.
        - A tuple `(m, n, ...)` for tensor-valued functions.
    degree : int
        The polynomial degree.

    Returns
    -------
        A function space in the given domain.
    """

    if rank == 1:  # Scalar function space
        element = (family, degree)
    elif isinstance(rank, int):
        # Vector-valued function space of dimension `rank`
        element = (family, degree, (rank,))
    elif isinstance(rank, tuple):
        # Tensor-valued function space of arbitrary shape
        element = (family, degree, rank)
    else:
        raise ValueError("The 'rank' argument must be an integer or a tuple.")

    return functionspace(domain, element)


def create_mixed_space(
    domain: Mesh, families: List[str], ranks: List[tuple], degrees: List[int]
) -> FunctionSpace:
    fe = domain.basix_cell()  # finite element
    els = []
    for fy, rk, dg in zip(families, ranks, degrees):
        els.append(element(fy, fe, dg, dtype=default_real_type, shape=rk))
    mix_el = mixed_element(els)
    return functionspace(domain, mix_el)


def build_solver(
    domain: Mesh, bilinear_form: Form, dirichlet_bcs: Sequence[DirichletBC]
) -> PETSc.KSP:
    """
    Build and configure a PETSc Krylov Subspace Solver
    for a given bilinear form.

    Parameters
    ----------
    domain : Mesh
        Problem domain.
    bilinear_form : Form
        Bilinear form.
    dirichlet_bcs : Sequence[DirichletBC]
        A sequence of Dirichlet boundary conditions.

    Returns
    -------
    KSP
        A configured PETSc Krylov Subspace Solver.
    """

    A = assemble_matrix(bilinear_form, dirichlet_bcs)
    A.assemble()
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)  # or PREONLY
    solver.getPC().setType(PETSc.PC.Type.HYPRE)  # or LU
    solver.setTolerances(rtol=1e-6, atol=1e-10, max_it=1000)

    return solver


def create_solver(a, L, bcs, uh):
    """
    Creates a solver. To re-assemble the system and solve it,
    Run problem.sove()

    Reference:
    https://jsdokken.com/FEniCS-workshop/src/form_compilation.html
    """

    petsc_options = {
        "ksp_type": "gmres",  # Krylov method: GMRES (good for non-symmetric matrices)
        "pc_type": "hypre",  # Uses Hypre multigrid preconditioner (efficient for parallel computing)
        "ksp_rtol": 1e-6,  # Relative tolerance for convergence
        "ksp_atol": 1e-10,  # Absolute tolerance for convergence
        "ksp_max_it": 1000,  # Maximum number of iterations
        # "ksp_monitor": None,
    }

    problem = LinearProblem(a, L, u=uh, bcs=bcs, petsc_options=petsc_options)

    return problem


def basic_solver(a, L, bcs, uh):
    """
    This configuration provides a good balance
    between performance and parallel scalability,
    without requiring an expensive LU factorization.

    See for instance:
    https://jsdokken.com/FEniCS-workshop/src/form_compilation.html
    """
    petsc_options = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-10,
        "ksp_max_it": 1000,
        # "ksp_monitor": None
    }

    problem = LinearProblem(a, L, u=uh, bcs=bcs, petsc_options=petsc_options)
    problem.solve()


class SaveInfoSolver:

    def __init__(self, ksp):
        self.ksp = ksp
        self.residuals = []
        self.iterations = []
        self.rates = []
        self.ksp.setMonitor(self.monitor)

    def monitor(self, ksp, its, rnorm):
        self.residuals.append(rnorm)

    def add(self):
        its = self.ksp.getIterationNumber()
        residuals = np.array(self.residuals)
        if len(residuals) > 3:
            rate = np.mean(np.diff(np.log10(residuals))[2:])
        else:
            rate = np.nan
        self.residuals.clear()
        self.iterations.append(its)
        self.rates.append(rate)

    def save(self, path):
        data_to_save = {
            "iterations": np.array(self.iterations, dtype=np.int32),
            "rates": np.array(self.rates),
        }
        np.savez(path / f"{inf_name}.npz", **data_to_save)


class Smooth:

    def __init__(
        self, domain: Mesh, space: FunctionSpace, f: Function, eps: float = 1e-5
    ) -> None:

        dx = Measure("dx", domain=domain)
        u = TrialFunction(space)
        v = TestFunction(space)
        self.a = form(dot(u, v) * dx + eps * inner(grad(u), grad(v)) * dx)
        self.L = form(dot(f, v) * dx)
        self.solver = build_solver(domain, self.a, [])

    def run(self, fun: Function) -> None:
        L = assemble_vector(self.L)
        apply_lifting(L, [self.a], [[]])
        L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(L, [])
        self.solver.solve(L, fun.x.petsc_vec)
        fun.x.scatter_forward()


class Velocity:
    """
    This class build and solve the velocity equation.

    Attributes
    ----------
    biform : Form
        Bilinear form.
    liform : Form
        Linear functional.
    bc : List[DirichletBC]
        List with the homogeneous Dirichlet condition.
    solver : PETSc.KSP
        Krylov Subspace Solver

    Methods
    -------
    run(theta: Function) -> None
        Solves the velocity equation.
    """

    def __init__(
        self,
        dim: int,
        domain: Mesh,
        space: FunctionSpace,
        biform: Callable[[Argument, Argument], Tuple[Expr, bool]],
        S0: Expr,
        S1: Expr,
    ) -> None:
        """
        Sets up the linear system for the velocity.
        Since the bilinear part remains unchanged,
        it is compiled here.

        Parameters
        ----------
        dim : int
            Domain dimension.
        domain : Mesh
            Problem domain.
        space : FunctionSpace
            Space of functions.
        biform : Callable[[Argument, Argument], Tuple[Expr, bool]]
            Bilinear form method of a subclass of Model.
        S0 : Expr
            S0 component of the shape derivative.
        S1 : Expr
            S1 component of the shape derivative.
        """

        th = TrialFunction(space)
        xi = TestFunction(space)
        dx = Measure("dx", domain=domain)

        b, dirbc = biform(th, xi)

        self.biform = form(b)
        self.liform = form(-(dot(S0, xi) + inner(S1, grad(xi))) * dx)
        self.bc = None

        if dirbc == True:
            # Homogeneous Dirichlet boundary condition on the whole boundary
            # Recall that domain dimension = velocity rank
            self.bc = homogeneus_boundary(domain, space, dim, dim)
        elif isinstance(dirbc, tuple):
            # Homogeneous Dirichlet boundary condition on a subset of the boundary.
            # dirbc is a tuple with two components,
            # boundary tags and a list of marks. For instance,
            # dirbc = (tags, [mark0, mark1])
            self.bc = homogeneous_dirichlet(domain, space, dirbc[0], dirbc[1], dim)
        else:
            # By defult,
            # homogeneous Neumann boundary condition on the whole boundary
            self.bc = []

        self.solver = build_solver(domain, self.biform, self.bc)

    def run(self, theta: Function) -> None:
        """
        Solves the velocity equation.
        Only the linear part must be updated.
        """

        L = assemble_vector(self.liform)
        apply_lifting(L, [self.biform], [self.bc])
        L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(L, self.bc)

        self.solver.solve(L, theta.x.petsc_vec)
        theta.x.scatter_forward()


class Velocity_Mixed:
    """
    This class build and solve the velocity equation.

    Attributes
    ----------
    biform : Form
        Bilinear form.
    liform : Form
        Linear functional.
    bc : List[DirichletBC]
        List with the homogeneous Dirichlet condition.
    solver : PETSc.KSP
        Krylov Subspace Solver

    Methods
    -------
    run(theta: Function) -> None
        Solves the velocity equation.
    """

    def __init__(
        self,
        dim: int,
        domain: Mesh,
        space: FunctionSpace,
        biform: Callable[[Argument, Argument], Tuple[Expr, bool]],
        S0: Expr,
        S1: Expr,
        dx: Measure,
    ) -> None:
        """
        Sets up the linear system for the velocity.
        Since the bilinear part remains unchanged,
        it is compiled here.

        Parameters
        ----------
        dim : int
            Domain dimension.
        domain : Mesh
            Problem domain.
        space : FunctionSpace
            Space of functions.
        biform : Callable[[Argument, Argument], Tuple[Expr, bool]]
            Bilinear form method of a subclass of Model.
        S0 : Expr
            S0 component of the shape derivative.
        S1 : Expr
            S1 component of the shape derivative.
        """

        th = TrialFunction(space)
        xi = TestFunction(space)

        self.biform, dirbc = biform(th, xi)

        self.bc = None

        if dirbc == True:
            self.bc = homogeneus_boundary(domain, space, dim, dim)
        else:
            self.bc = []

        self.liform = -(dot(S0, xi) + inner(S1, grad(xi))) * dx((1, 2))

    def run(self, theta: Function) -> None:
        """
        Solves the velocity equation.
        Only the linear part must be updated.
        """
        basic_solver(form(self.biform), form(self.liform), self.bc, theta)


class Level:
    """
    This class implements the Petrov Galerkin and
    Crank-Nicolson methods to solve
    the transport equation corresponding
    to the level set function.

    Attributes
    ----------
    dt : Constant
        Time step.
    domain : Mesh
        Problem domain.
    a : Form
        Bilinear part of the iterative scheme.
    L : Form
        Linear part of the iterative scheme.

    Methods
    -------
    run(phi: Function, steps: int, tend: float) -> None
        Run the iterative method over a fixed number of time steps.
    """

    def __init__(
        self,
        domain: Mesh,
        space: FunctionSpace,
        phi: Function,
        tht: Function,
        diam2: float,
        smooth: bool,
    ) -> None:
        """
        Set up the iterative method to solve the level set equation.
        The bilinear and linear parts are pre-compiled.

        Parameters
        ----------
        domain : Mesh
            Problem domain.
        space : FunctionSpace
            Space of functions.
        phi : Function
            Level set function.
        tht : Function
            Velocity function.
        diam2 : float
            Square of the mesh diameter.
        smooth : bool
            Flag to add diffusion.
        """

        self.domain = domain
        self.dt = const(domain, 0.0)
        dx = Measure("dx", domain=domain)

        u = TrialFunction(space)
        v = TestFunction(space)

        # # ----
        # sign_phi = conditional(
        #     lt(phi, -0.25), -0.25, conditional(gt(phi, 0.25), 0.25, phi)
        # )
        # # ----

        # Petrov-Galerkin test function
        tau = 2.0 * sqrt(1.0 / self.dt**2 + dot(tht, tht) / diam2)
        new_v = v + dot(grad(v), tht) / tau

        # Crank-Nicolson weak formulation
        a = (u + (self.dt / 2.0) * dot(tht, grad(u))) * new_v * dx
        L = (phi - (self.dt / 2.0) * dot(tht, grad(phi))) * new_v * dx

        # Add diffusion
        if smooth:
            a += (self.dt / 2.0) * diam2 * dot(grad(u), grad(v)) * dx
            L -= (self.dt / 2.0) * diam2 * dot(grad(phi), grad(v)) * dx

        # Pre-compilation
        self.a = form(a)
        self.L = form(L)

    def run(self, phi: Function, steps: int, tend: float) -> None:
        """
        Run the iterative method over a fixed number of time steps.
        Since the bilinear part remains unchanged across iterations,
        it is compiled only once. Therefore, the linear part must be
        correctly updated.

        Parameters
        ----------
        phi : Function
            Level set function.
        steps : int
            Number of time steps.
        tend : float
            Final time.
        """
        self.dt.value = tend / steps

        solver = build_solver(self.domain, self.a, [])
        b = create_vector(self.L)

        for _ in range(steps):
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, self.L)
            b.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            solver.solve(b, phi.x.petsc_vec)
            phi.x.scatter_forward()


class Reinit:
    """
    This class implements the Petrov Galerkin and
    two-step Adams-Bashforth methods to solve
    the diffusive Eikonal equation with
    fictitious time derivative.

    Attributes
    ----------
    dt : Constant
        Time step.
    phi_ini : Function
        Store the initial level set function to be reinitialized.
    phi_prev : Function
        Store the previous iteration.
    w : Function
        Auxiliary function.
    uh : Function
        Solution function.
    problem0 : LinearProblem
        Solver used in the first iteration.
    problem : LinearProblem
        Solver used in subsequent iterations.

    Methods
    -------
    run(phi: Function, steps: int, tend: float) -> None
        Run the iterative method over a fixed number of time steps.
    """

    def __init__(
        self, domain: Mesh, space: FunctionSpace, phi: Function, diam2: float
    ) -> None:
        """
        Set up the reinitialization method.
        Since the bilinear parts change across iterations
        (due to the Petrov-Galerkin test function),
        two solvers are created, which compile
        the equations at each call.

        Parameters
        ----------
        domain : Mesh
            Problem domain.
        space : FunctionSpace
            Space of functions.
        phi : Function
            Level set function.
        diam2 : float
            Square of the mesh diameter.
        """

        self.dt = const(domain, 0.0)
        dx = Measure("dx", domain=domain)

        self.phi_ini = Function(space)
        self.phi_prev = Function(space)

        self.uh = Function(space)
        self.w = Function(space)

        # Sign function
        sign_phi_ini = self.phi_ini / sqrt(self.phi_ini**2 + diam2)
        # Hamiltonian
        H = lambda p: sign_phi_ini * sqrt(dot(p, p))
        # Gradient of H
        GradH = lambda p: sign_phi_ini * p / sqrt(dot(p, p))

        u = TrialFunction(space)
        v = TestFunction(space)

        # Petrov-Galerkin test function
        tau = 2.0 * sqrt(1.0 / self.dt**2 + sign_phi_ini**2 / diam2)
        new_v = v + dot(grad(v), GradH(grad(phi))) / tau

        # Adams-Bashforth weak formulation
        a = u * new_v * dx
        L = phi * new_v * dx
        L += self.dt * sign_phi_ini * new_v * dx
        L += (
            (self.dt / 2.0) * (H(grad(self.phi_prev)) - 3.0 * H(grad(phi))) * new_v * dx
        )
        # Add diffusion
        L += (
            (self.dt / 2.0) * diam2 * dot(grad(self.phi_prev - 3.0 * phi), grad(v)) * dx
        )

        self.problem = create_solver(form(a), form(L), [], self.uh)

        # Explicit Euler weak formulation
        a0 = u * new_v * dx
        L0 = self.dt * sign_phi_ini * new_v * dx
        L0 += (self.phi_ini - self.dt * H(grad(self.phi_ini))) * new_v * dx
        # Add diffusion
        L0 -= self.dt * diam2 * dot(grad(self.phi_ini), grad(v)) * dx

        self.problem0 = create_solver(form(a0), form(L0), [], self.uh)

    def run(self, phi: Function, steps: int, tend: float) -> None:
        """
        Run the iterative method over a fixed number of time steps.
        The solver for the first iteration is called only once,
        and then the main solver is called at each time step.

        Parameters
        ----------
        phi : Function
            Level set function.
        steps : int
            Number of time steps.
        tend : float
            Final time.
        """

        self.dt.value = tend / steps
        self.phi_ini.interpolate(phi)

        self.problem0.solve()
        self.phi_prev.x.array[:] = self.uh.x.array[:]

        for _ in range(steps):
            self.w.x.array[:] = phi.x.array[:]
            self.problem.solve()
            phi.x.array[:] = self.uh.x.array[:]
            self.phi_prev.x.array[:] = self.w.x.array[:]


def homogeneus_boundary(domain, space, dim, rank_dimension):
    """
    Create homgeneous boundary conditions over the whole
    boundary. Used to calculate the velocity field.
    """
    boundary_facets = exterior_facet_indices(domain.topology)
    if rank_dimension == 1:
        u_zero = PETSc.ScalarType(0)
    else:
        u_zero = np.zeros(rank_dimension, dtype=PETSc.ScalarType)
    bc = dirichletbc(
        u_zero, locate_dofs_topological(space, dim - 1, boundary_facets), space
    )
    return [bc]


def homogeneous_dirichlet_point(domain, space, points, rank_dimension, values=None):
    """
    Create homogeneous Dirichlet boundary conditions at specific points
    for scalar or vector functions in 2D or 3D.
    It works for Lagrange P1!

    Parameters:
    - domain: The computational domain.
    - space: The function space.
    - points: A list of coordinate lists representing boundary points.
    - rank_dimension: The rank of the function (1 for scalar, >1 for vector).

    Returns:
    - A list of Dirichlet boundary conditions.
    """
    dim = domain.geometry.dim

    if values is not None:
        bcs = []
        for p, v in zip(points, values):
            dofs = locate_dofs_geometrical(
                space,
                lambda x: np.logical_and.reduce(
                    [np.isclose(x[i], p[i]) for i in range(dim)]
                ),
            )
            bcs.append(dirichletbc(PETSc.ScalarType(v), dofs, space))

        return bcs

    if rank_dimension == 1:
        u_zero = PETSc.ScalarType(0)
    else:
        u_zero = np.zeros(rank_dimension, dtype=PETSc.ScalarType)

    bcs = []
    for p in points:
        dofs = locate_dofs_geometrical(
            space,
            lambda x: np.logical_and.reduce(
                [np.isclose(x[i], p[i]) for i in range(dim)]
            ),
        )
        bcs.append(dirichletbc(u_zero, dofs, space))

    return bcs


def homogeneous_dirichlet_point_coordinate(domain, space, points, coordinates):
    """
    Create homogeneous Dirichlet boundary conditions at specific points
    and specific coordinate (either x, y, or z)
    It works for Lagrange P1!

    Parameters:
    - domain: The computational domain.
    - space: The function space.
    - points: A list of coordinate lists representing boundary points.
    - coordinates: A list of the coordinate (0, 1, 2) to apply the condition.

    Returns:
    - A list of Dirichlet boundary conditions.
    """

    dim = domain.geometry.dim
    bcs = []
    for p, c in zip(points, coordinates):
        vertex = locate_entities_boundary(
            domain,
            0,
            lambda x: np.logical_and.reduce(
                [np.isclose(x[i], p[i]) for i in range(dim)]
            ),
        )
        dofs = locate_dofs_topological(space.sub(c), 0, vertex)
        bcs.append(dirichletbc(PETSc.ScalarType(0.0), dofs, space.sub(c)))
    return bcs


def dirichlet_with_values(domain, space, boundary_tags, mrks_dirichlet, values):

    bcs = []
    for mk, v in zip(mrks_dirichlet, values):
        dofs = locate_dofs_topological(
            space,
            domain.geometry.dim - 1,
            boundary_tags.indices[boundary_tags.values == mk],
        )
        bcs.append(dirichletbc(PETSc.ScalarType(v), dofs, space))

    return bcs


def homogeneous_dirichlet(
    domain: Mesh, space: FunctionSpace, boundary_tags, mrks_dirichlet, rank_dimension
) -> List[DirichletBC]:
    """
    Creates homogeneous Dirichlet boundary conditions
    over faces of dimension domain.geometry.dim - 1.

    Parameters
    ----------
    domain:
        Problem domain.
    space:
        Space of functions.
    tags:
        Markers for the facets of the domain.
    mrks_dirichlet (List[int]):
        List of tags corresponding to
        Dirichlet boundary conditions.
    rank_dimension:
        1 scalar, 2 vector 2D, 3 vector 3D,
        (3, 3) rank-2 tensor in 3D, etc

    Returns
    -------
        A list of Dirichlet boundary conditions.
    """
    if rank_dimension == 1:
        u_zero = PETSc.ScalarType(0)
    else:
        u_zero = np.zeros(rank_dimension, dtype=PETSc.ScalarType)

    bcs = []
    for mk in mrks_dirichlet:
        dofs = locate_dofs_topological(
            space,
            domain.geometry.dim - 1,
            boundary_tags.indices[boundary_tags.values == mk],
        )
        bcs.append(dirichletbc(u_zero, dofs, space))
    return bcs


def homogeneous_dirichlet_mixed(
    domain: Mesh,
    subspace: FunctionSpace,
    boundary_tags: MeshTags_int32,
    markers: List[int],
):
    """
    Create homogeneous Dirichlet boundary conditions for a subspace of a mixed space.

    This function constructs zero Dirichlet boundary conditions on the facets
    identified by the given boundary markers. It is intended for use with
    subspaces extracted from a mixed finite element space in FEniCSx.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The computational mesh.

    subspace : dolfinx.fem.FunctionSpace
        Subspace of a mixed function space (e.g., obtained via
        ``mix_space.sub(i)``) on which the Dirichlet condition will be imposed.

    boundary_tags : dolfinx.mesh.MeshTags
        Mesh tags identifying boundary facets. Typically created with
        ``meshtags`` and used to mark parts of the boundary.

    markers : iterable of int
        Boundary markers where the homogeneous Dirichlet condition should
        be applied.

    Returns
    -------
    list of dolfinx.fem.DirichletBC
        List of Dirichlet boundary condition objects corresponding to the
        specified boundary markers.

    Notes
    -----
    The function internally collapses the given subspace to create a standalone
    function space required for defining the boundary values. The zero boundary
    function is then mapped back to the original mixed subspace when constructing
    the Dirichlet boundary conditions.

        Examples
    --------
    >>> bcs = homogeneous_dirichlet_mixed(
    ...     domain,
    ...     mix_space.sub(0),
    ...     boundary_tags,
    ...     [1, 3]
    ... )
    """

    space, _ = subspace.collapse()

    u0 = Function(space)
    u0.x.array[:] = 0.0

    facet_dim = domain.topology.dim - 1
    bcs = []

    for mk in markers:
        facets = boundary_tags.indices[boundary_tags.values == mk]

        dofs = locate_dofs_topological(
            (subspace, space),
            facet_dim,
            facets,
        )

        bcs.append(dirichletbc(u0, dofs, subspace))

    return bcs


def homogeneous_dirichlet_y_coord(domain, space, boundary_tags, mrks_dirichlet):
    bcs = []
    for mk in mrks_dirichlet:
        dofs = locate_dofs_topological(
            space.sub(1),
            domain.geometry.dim - 1,
            boundary_tags.indices[boundary_tags.values == mk],
        )
        bcs.append(dirichletbc(PETSc.ScalarType(0), dofs, space.sub(1)))
    return bcs


def homogeneous_dirichlet_fun(domain, space, funcs, rank_dimension):
    if rank_dimension == 1:
        u_zero = PETSc.ScalarType(0)
    else:
        u_zero = np.zeros(rank_dimension, dtype=PETSc.ScalarType)
    bcs = []
    for f in funcs:
        dofs = locate_dofs_geometrical(space, f)
        bcs.append(dirichletbc(u_zero, dofs, space))
    return bcs


def fun_ds(domain, funcs):
    dim_faces = domain.geometry.dim - 1
    facet_indices_list = []
    facet_values_list = []
    for i, f in enumerate(funcs, start=1):
        indices = locate_entities_boundary(domain, dim=dim_faces, marker=f)
        facet_indices_list.append(indices)
        facet_values_list.append(np.full(len(indices), i, dtype=np.int32))

    facet_indices = np.concatenate(facet_indices_list)
    facet_values = np.concatenate(facet_values_list)

    facet_tag = meshtags(domain, dim_faces, facet_indices, facet_values)

    indexed_ds = Measure("ds", domain=domain, subdomain_data=facet_tag)

    return [indexed_ds(i) for i in range(1, len(funcs) + 1)]


def marked_ds(domain, boundary_tags, marks):
    indexed_ds = Measure("ds", domain=domain, subdomain_data=boundary_tags)
    return [indexed_ds(i) for i in marks]


class AdapTime:
    """
    Line search utility to estimate (with randomness)
    the number of steps and final time
    based on the norm of the velocity field.

    Attributes
    ----------
    t_min: float
        Minimum time.
    t_max: float
        Maximum time.
    s_min: int
        Minimum number of steps.
    s_max: int
        Maximum number of steps.
    factor: float
        Scaling factor used to estimate final time.
    noise: float
        Random variation factor, e.g., 0.05 for 5% variation.

    Methods
    -------
    _nbr_steps_func(t: float):
        Power function to estimate the number of steps.
    get(derivative_norm: float):
        Computes the number of steps and final time.
    """

    def __init__(
        self,
        time_bounds: Tuple[float, float],
        step_bounds: Tuple[int, int],
        factor: float,
        noise: float = 0.05,
    ):
        """
        Parameters
        ---------
        time_bounds:  List[float, float]
            Minimum and maximum allowed final times.
        step_bounds: List[int, int]
            Minimum and maximum allowed number of steps.
        factor: float
            Scaling factor used to estimate final time.
        noise: float
            Random variation factor (e.g., 0.05 for 5% variation).
        """
        self.t_min, self.t_max = time_bounds
        self.s_min, self.s_max = step_bounds
        self.factor = factor
        self.noise = noise

    def _estimate_steps_from_time(self, t: float):
        """
        Power function to estimate the number of steps.

        Parameters
        ---------
        t: float
            Final time.

        Returns
        -------
        float: estimated number of steps
        """

        scaled_t = (t - self.t_min) / (self.t_max - self.t_min)
        nbr_steps = (self.s_max - self.s_min) * scaled_t ** (1.0 / 6.0) + self.s_min

        return nbr_steps

    def get(self, derivative_norm: float):
        """
        Computes the number of steps and final time.
        Reference:
        https://docu.ngsolve.org/latest/i-tutorials/unit-7-optimization/01_Shape_Derivative_Levelset.html

        Parameters
        ---------
        derivative_norm : float
            Norm of the velocity field.

        Returns
        -------
        Tuple[int, float]:
            Number of steps and final time.
        """

        # To prevent division by zero
        safe_norm = max(derivative_norm, 1e-8)
        # Final time
        tend = self.factor / safe_norm
        # Randomness
        tend = tend * np.random.uniform(1.0 - self.noise, 1.0 + self.noise)
        # To guarantee tend in [t_min, t_max]
        tend = max(self.t_min, min(tend, self.t_max))

        steps = self._estimate_steps_from_time(tend)
        # Randomness
        steps = np.random.normal(steps, self.noise * steps)
        # Conversion to integer
        steps = round(steps)
        # To guarantee steps in [s_min, s_max]
        steps = max(self.s_min, min(steps, self.s_max))

        return steps, tend


def get_rank_dimension(shape):
    if len(shape) == 0:
        return 1
    elif len(shape) == 1:
        return shape[0]
    else:
        print("Tensor rank!")
        return shape


def SolveLinearProblem(space, pde, name):
    u = Function(space)
    u.name = name
    weak_form, bcs = pde
    bi, li = system(weak_form)
    basic_solver(form(bi), form(li), bcs, u)
    return u


def solve_pde(space, pde, phi):

    eqs = pde(phi)
    nbr_eqs = len(eqs)
    solutions = [Function(space) for _ in range(nbr_eqs)]
    for i in range(nbr_eqs):
        solutions[i].name = "u" + str(i)

    for (weak_form, bcs), u in zip(eqs, solutions):
        bi, li = system(weak_form)
        basic_solver(form(bi), form(li), bcs, u)

    return solutions


def dir_extension_from(
    comm: MPI.Comm,
    domain: Mesh,
    space: FunctionSpace,
    pde: Callable[[Function], List[Tuple[Expr, List[DirichletBC]]]],
    func: Callable[[npt.NDArray[np.float64]], float],
    path: Path,
):
    """
    It computes the Dirichlet extension of the solutions
    to a set of linear partial differential equations.

    Parameters
    ---------
    comm : MPI.Comm.
        Communicator.
    domain : Mesh
        Problem domain.
    space : FunctionSpace
        Space of functions.
    pde : Callable[[Function], List[Tuple[Expr, List[DirichletBC]]]]
        Linear partial differential equations.
    func : Callable[[npt.NDArray[np.float64]], float]
        Level set function.
    path : Path
        Test path.

    Returns
    -------
    extensions : List[Function]
        Dirichlet extension functions.
    """

    phi = interpolate(funcs=[func], to_space=create_space(domain, "CG", 1), name="phi")[
        0
    ]

    solutions = solve_pde(space, pde, phi)
    extensions = dirichlet_extension(domain, space, solutions)

    save_functions(
        comm, domain, [phi] + solutions + extensions, path / f"{ext_name}.xdmf"
    )

    return extensions


def global_scalar(value, comm, postprocess=None):

    local_val = assemble_scalar(value)
    global_val = comm.allreduce(local_val, op=MPI.SUM)

    return postprocess(global_val) if postprocess else global_val


def global_scalar_list(values, comm):

    local_vals = [assemble_scalar(v) for v in values]

    return [comm.allreduce(v, op=MPI.SUM) for v in local_vals]


class NonlinearSolverWrapper:
    """
    Wrapper for solving nonlinear problems.
    """

    def __init__(self, solver, u, initial):
        """
        Parameters
        ---------
        solver : dolfinx.nls.petsc.NewtonSolver
            Non-linear solver.
        u : dolfinx.fem.Function
            Function to save the solution.
        initial : callable
            Initial guess. A function that can be
            evaluated at mesh points. For instance,
            a lambda function.
        bcs : List[DirichletBC]
            List of Dirichlet boundary conditions.
        """
        self.solver = solver
        self.u = u
        self.initial = initial
        self.solver_initial = None
        self.niter = 0

    def solve(self):
        self.u.interpolate(self.initial)
        self.niter, converged = self.solver.solve(self.u)
        if not converged:
            print(f"> Newton solver did not converge!")

    def see(self) -> str:
        """
        Returns a string with maximum number of iterations
        """
        return (f"newt = {self.niter:2.0f} | ",)[0]


class NonlinearSolverbySteeping:
    def __init__(self, solver, u, initial, factor, nbr_steps, func=None):
        self.solver = solver
        self.u = u
        self.ini = initial
        self.factor = factor
        self.vals = np.linspace(0, 1.0, nbr_steps)[1:]
        # self.vals = self.vals**2.5
        self.max_niter = 0
        self.func = func

    def solve(self):

        self.u.interpolate(self.ini)
        self.max_niter = 0

        for fc in self.vals:
            self.factor.value = PETSc.ScalarType(fc)
            # print(self.factor.value)
            try:
                niter, converged = self.solver.solve(self.u)
            except RuntimeError as e:
                print(f"Solver failed at load factor {fc}")
                print(f"Max iterations so far: {self.max_niter}")
                raise

            if self.func is not None:
                self.func(self.u)

            self.max_niter = max(self.max_niter, niter)

    def see(self) -> str:
        """
        Returns a string with maximum number of iterations
        """
        return (f"newt = {self.max_niter:2.0f} | ",)[0]


class NonlinearSolverbyIniLinAndSteeping:
    """
    Experimental
    """

    def __init__(self, solver, u, ini_linear, factor, nbr_steps, bcs):
        self.solver = solver
        self.u = u
        self.factor = factor
        self.vals = np.linspace(0, 1.0, nbr_steps)[1:]
        self.max_niter = 0

        bi, li = system(ini_linear)
        self.ini_solver = create_solver(form(bi), form(li), bcs, self.u)
        print("Here!")

    def solve(self):

        self.u.interpolate(lambda x: 0.0 * x[:2])

        for fc in self.vals:
            self.factor.value = PETSc.ScalarType(fc)
            try:
                niter, converged = self.solver.solve(self.u)
                print(converged, niter)
            except RuntimeError as e:
                # if converged == False:
                print(f"> Linear solution!, fc={fc}")
                self.ini_solver.solve()
                break


def create_nonlinear_solver_with_ini_linear_and_factor(
    comm, F, bcs, J, u, ini_linear, factor, nbr_steps, func=None
):
    problem = NonlinearProblem(F, u, bcs=bcs, J=J)
    solver = NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
    solver.max_it = 40

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()

    # Lower computational cost
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "hypre"
    opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"

    ksp.setFromOptions()

    return NonlinearSolverbyIniLinAndSteeping(
        solver, u, ini_linear, factor, nbr_steps, bcs
    )


def create_nonlinear_solver_with_factor(
    comm, F, bcs, J, u, initial, factor, nbr_steps, func=None
):

    problem = NonlinearProblem(F, u, bcs=bcs, J=J)
    solver = NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
    solver.max_it = 40

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()

    # Higher computational cost
    # opts[f"{option_prefix}ksp_type"] = "preonly"
    # opts[f"{option_prefix}pc_type"] = "lu"
    # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"

    # Lower computational cost
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "hypre"
    opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"

    ksp.setFromOptions()

    return NonlinearSolverbySteeping(solver, u, initial, factor, nbr_steps, func)


def create_nonlinear_solver(comm, F, bcs, J, u, initial):
    """
    Creates a Newton solver for nonlinear problems.
    """
    problem = NonlinearProblem(F, u, bcs=bcs, J=J)
    solver = NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
    solver.max_it = 40

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()

    # Higher computational cost
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"

    # Lower computational cost
    # opts[f"{option_prefix}ksp_type"] = "gmres"
    # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    # opts[f"{option_prefix}pc_type"] = "hypre"
    # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"

    ksp.setFromOptions()

    return NonlinearSolverWrapper(solver, u, initial)


def L2_interpolation(domain, space, expr, diam2):
    dx = Measure("dx", domain=domain)
    uh = Function(space)
    v, w = TrialFunction(space), TestFunction(space)
    problem = LinearProblem(
        (inner(v, w) + diam2 * dot(grad(v), grad(w))) * dx,
        inner(expr, w) * dx,
        u=uh,
        petsc_options={"ksp_type": "cg"},
    ).solve()
    return uh


class Save_Functions:

    def __init__(self, space, names):

        self.functions = [Function(space) for _ in range(len(names))]
        for i in range(len(names)):
            self.functions[i].name = names[i]
        self.idx = 0

    def save(self, f):
        self.functions[self.idx].interpolate(f)
        self.idx += 1


def SolveNonlinearOnce(domain, space, ste_problem, names):

    sf = Save_Functions(space, names)
    solution = Function(space)

    weak_form, bcs, jacobian, seudo_state, ini_func, factor_pars = ste_problem
    factor, nbr_newton_steps = factor_pars
    true_factor = const(domain, 0.0)
    true_weak_form = replace(weak_form, {seudo_state: solution, factor: true_factor})
    true_jacobian = replace(jacobian, {seudo_state: solution, factor: true_factor})

    problem = create_nonlinear_solver_with_factor(
        comm,
        true_weak_form,
        bcs,
        true_jacobian,
        solution,
        ini_func,
        true_factor,
        nbr_newton_steps,
        sf.save,
    )

    problem.solve()

    return sf.functions


def read_level_set_function(path: Path, domain: Mesh, niter: int) -> Function:
    """
    Read a level set function from the results.
    Especifically, the h5 file called res_name.h5.
    Parallelism does not work with this function.
    It is assumed that the domain corresponds to the level set function.
    """
    with File(path / f"{res_name}.h5", "r") as data_file:
        phi_group = data_file[f"/Function/phi"]
        phi_vals = phi_group[str(niter)][:, 0]
        space = create_space(domain, "CG", 1)
        phi = Function(space)
        phi.name = "phi"
        phi.x.array[:] = phi_vals

        return phi

    return None


def phifem_solver_mixed(a, L, bcs, uh, map0):
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
            "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_25": 0,
        },
    )
    w = problem.solve()
    uh.x.array[:] = w.x.array[map0]


def phifem_solver(
    a: Expr,
    L: Expr,
    bcs: Sequence[DirichletBC],
    uh: Function,
    phi: Function,
    rank_dim: int,
    comm: MPI.Comm,
) -> None:
    """
    This function was created to use phiFem method in formopt.
    It solves the linear system using LU and compute the phiFem
    solution.
    """
    A = assemble_matrix(form(a), bcs=bcs)
    A.assemble()
    b = assemble_vector(form(L))

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    F = pc.getFactorMatrix()
    F.setMumpsIcntl(icntl=24, ival=1)
    F.setMumpsIcntl(icntl=25, ival=0)
    ksp.solve(b, uh.x.petsc_vec)
    ksp.destroy()

    uh.x.scatter_forward()
    for i in range(rank_dim):
        uh.x.array[i::rank_dim] *= phi.x.array[:]
    uh.x.scatter_forward()


def phifem_solve(
    nbr_eq: int,
    equations: List[Tuple[Expr, Sequence[DirichletBC]]],
    solutions: List[Function],
    phi: Function,
    rank_dim: int,
    comm: MPI.Comm,
):
    """
    This function was created to use phiFem method in formopt.
    """
    for i in range(nbr_eq):
        weak_form, bcs = equations[i]
        bi, li = system(weak_form)
        phifem_solver(bi, li, bcs, solutions[i], phi, rank_dim, comm)


def phifem_solve_mixed(
    nbr_eq: int,
    equations: List[Tuple[Expr, Sequence[DirichletBC]]],
    solutions: List[Function],
    map,
):
    """
    This function was created to use phiFem method in formopt.
    """
    for i in range(nbr_eq):
        weak_form, bcs = equations[i]
        bi, li = system(weak_form)
        phifem_solver_mixed(bi, li, bcs, solutions[i], map)


def get_initial_level(
    domain: Mesh,
    centers: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
    factor: float = 1.0,
    ord: int = 2,
):
    """
    This function was created to use phiFem method in `formopt`.
    """
    ini_lvl = InitialLevel(centers, radii, factor, ord)
    sp_lset = create_space(domain, "CG", 1)
    phi0 = Function(sp_lset)
    phi0.interpolate(ini_lvl.func)
    return phi0


def phifem_run(
    model: Model,
    niter: int,
    reinit_step: int | bool,
    reinit_pars: Tuple[int, float],
    dfactor: float,
    lv_time: Tuple[float, float],
    lv_iter: Tuple[int, int],
    smooth: bool,
    start_to_check: int,
    ctrn_tol: float,
    lgrn_tol: float,
    cost_tol: float,
    prev: int,
    random_pars: Tuple[int, float],
) -> Function:
    """
    This function was created to use phiFem method in formopt.
    It is based on the `runDP` function.
    The parallelism scheme is implemented in the hope
    that phifem functions will run in parallel someday.
    """

    start_assembly = MPI.Wtime()

    rein_steps, rein_end = reinit_pars
    min_time, max_time = lv_time
    min_iter, max_iter = lv_iter
    seed, noise = random_pars

    # Constants ===========================
    filename = model.path / f"{res_name}.xdmf"
    stop_flag = False

    dim = model.dim
    domain = model.domain
    rank_dim = model.rank_dim

    vol = volume(domain, comm)
    nfems = nbr_fems(domain, dim, comm)

    if rank == 0:
        diam2 = get_diam2(dim, vol, nfems)
        np.random.seed(seed)
        lsearch = AdapTime((min_time, max_time), (min_iter, max_iter), dfactor, noise)
        tosave = Save()
    else:
        diam2 = None

    diam2 = comm.bcast(diam2, root=0)

    # List of variables
    # -----------------
    # sp_lset : space of level set functions
    # sp_vlty : space of velocity fields
    # phi : level set function
    # tht : velocity field
    # ste_eqs : list of state equations
    # adj_eqs : list of adjoint equations
    # ste_fcs : list of state functions
    # adj_fcs : list of adjoint functions
    # ste_pbs : list of state problems
    # adj_pbs : list of adjoint problems
    # eq_ctrs : list of equality constraints
    # nbr_ste : number of states
    # nbr_adj : number of adjoints
    # nbr_ctr : number of constraints
    # J : cost functional
    # C : list of constraints
    # L : list of lagrange multipliers
    # S0, S1 : derivative components

    # Level set function
    sp_lset = create_space(domain, "CG", 1)
    phi = Function(sp_lset)
    phi.name = "phi"
    # Velocity field
    sp_vlty = create_space(domain, "CG", dim)
    tht = Function(sp_vlty)
    tht.name = "tht"

    # State equations/functions
    ste_eqs = model.pde(phi)
    nbr_ste = len(ste_eqs)
    ste_fcs = [Function(model.space) for _ in range(nbr_ste)]
    for i in range(nbr_ste):
        ste_fcs[i].name = "u" + str(i)

    # Adjoint equations/functions
    adj_eqs = model.adjoint(phi, ste_fcs)
    nbr_adj = len(adj_eqs)
    if nbr_adj > 0:
        adj_fcs = [Function(model.space) for _ in range(nbr_adj)]
        for i in range(nbr_adj):
            adj_fcs[i].name = "p" + str(i)
    else:
        adj_fcs = []

    # to be evaluated: quantities
    nbr_to_ev_qs = len(model._to_eval["quantity"])
    if nbr_to_ev_qs > 0:
        to_ev_qs = []
        qs_names = []

        for name, fc in model._to_eval["quantity"].items():
            to_ev_qs.append(form(fc(model, phi, ste_fcs, adj_fcs)))
            qs_names.append(name)

        if rank == 0:
            tosave.add_qtty_names(qs_names)

    # Derivative components
    S0_cts, S1_cts = model.derivative(phi, ste_fcs, adj_fcs)
    S0 = S0_cts[0]
    S1 = S1_cts[0]
    # Equality constraints
    eq_ctrs = model.constraint(phi, ste_fcs)
    nbr_ctr = len(eq_ctrs)
    if nbr_ctr > 0:
        # Compilation of the constraints
        C = [form(c) for c in eq_ctrs]
        # Lagrange multipliers
        L = [const(domain, 0) for _ in range(nbr_ctr)]
        # Creation of the derivatives components
        for i in range(nbr_ctr):
            S0 += L[i] * S0_cts[1][i]
            S1 += L[i] * S1_cts[1][i]
    # Derivative norm
    nDJ = form((model.bilinear_form(tht, tht))[0])

    # To calculate the level set function
    cls_lset = Level(domain, sp_lset, phi, tht, diam2, smooth)

    # Reinicialization
    if reinit_step:
        cls_rein = Reinit(domain, sp_lset, phi, diam2)

    local_assembly = MPI.Wtime() - start_assembly
    max_assembly = comm.allreduce(local_assembly, op=MPI.MAX)

    # Iteration i = 0 =======
    start_solve = MPI.Wtime()

    phi.interpolate(model._get_initial_level())

    #######################################################################
    # Update model.dx, model.dS, model.ds_out
    cells_tags, facets_tags, _, model.ds_out, _, _ = compute_tags_measures(
        domain, phi, 1, box_mode=True
    )
    model.dx = Measure("dx", domain=domain, subdomain_data=cells_tags)
    model.dS = Measure("dS", domain=domain, subdomain_data=facets_tags)

    #######################################################################
    # To calculate the velocity field
    cls_vlty = Velocity(dim, domain, sp_vlty, model.bilinear_form, S0, S1)

    cls_smt = Smooth(domain, sp_lset, phi, diam2)
    cls_smt.run(phi)

    ste_eqs = model.pde(phi)
    if not model.mixed_space:
        phifem_solve(nbr_ste, ste_eqs, ste_fcs, phi, rank_dim, comm)
    else:
        phifem_solve_mixed(nbr_ste, ste_eqs, ste_fcs, model.map)

    comm.barrier()

    cost = global_scalar(form(model.cost(phi, ste_fcs)), comm)

    if nbr_ctr > 0:
        eq_ctrs = model.constraint(phi, ste_fcs)
        ctrs = global_scalar_list([form(c) for c in eq_ctrs], comm)
        if rank == 0:
            cls_meth = PPL(nbr_ctr, cost, ctrs)

    if nbr_adj > 0:
        adj_eqs = model.adjoint(phi, ste_fcs)
        if not model.mixed_space:
            phifem_solve(nbr_adj, adj_eqs, adj_fcs, phi, rank_dim, comm)
        else:
            phifem_solve_mixed(nbr_adj, adj_eqs, adj_fcs, model.map)
        comm.barrier()

    if nbr_to_ev_qs > 0:
        qs_eval = global_scalar_list(to_ev_qs, comm)

    cls_vlty.run(tht)  # model.dx
    nder = global_scalar(nDJ, comm, np.sqrt)

    if rank == 0:
        lset_steps, lset_end = lsearch.get(nder)
    else:
        lset_steps, lset_end = None, None
    lset_steps = comm.bcast(lset_steps, root=0)
    lset_end = comm.bcast(lset_end, root=0)

    # print ==============================================
    if rank == 0:
        print("> Iterations:")
        if nbr_ctr > 0:
            print0(0, cost, ctrs, nder, 0, cls_meth.see())
        else:
            print1(0, cost, nder, 0)
        tosave.add(cost, nder)
        if nbr_to_ev_qs > 0:
            tosave.add_quantities(qs_eval)
    # ====================================================
    spaceP1 = None
    degree_space = model.space.ufl_element().degree
    if degree_space > 1:
        rank_space = model.space.value_shape
        if len(rank_space) == 1:
            rank_space = rank_space[0]
        spaceP1 = create_space(domain, "CG", rank=rank_space, degree=1)

    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi, 0)

        if degree_space == 1:
            for f in ste_fcs:
                xdmf.write_function(f, 0)
            for f in adj_fcs:
                xdmf.write_function(f, 0)
        else:
            ste_fcsP1 = interpolate(ste_fcs, spaceP1, name="u")
            adj_fcsP1 = interpolate(adj_fcs, spaceP1, name="p")
            for f in ste_fcsP1:
                xdmf.write_function(f, 0)
            for f in adj_fcsP1:
                xdmf.write_function(f, 0)

        xdmf.write_function(tht, 0)

        for iter in range(1, niter + 1):

            cls_lset.run(phi, lset_steps, lset_end)
            # Reinitialization
            if reinit_step and iter > start_to_check:
                if iter % reinit_step == 0:
                    cls_rein.run(phi, rein_steps, rein_end)

            #######################################################################
            # Update model.dx, model.dS, model.ds_out
            cells_tags, facets_tags, _, model.ds_out, _, _ = compute_tags_measures(
                domain, phi, 1, box_mode=True
            )
            model.dx = Measure("dx", domain=domain, subdomain_data=cells_tags)
            model.dS = Measure("dS", domain=domain, subdomain_data=facets_tags)

            #######################################################################

            # cls_smt.run(phi) # smooth

            ste_eqs = model.pde(phi)
            if not model.mixed_space:
                phifem_solve(nbr_ste, ste_eqs, ste_fcs, phi, rank_dim, comm)
            else:
                phifem_solve_mixed(nbr_ste, ste_eqs, ste_fcs, model.map)

            comm.barrier()

            cost = global_scalar(form(model.cost(phi, ste_fcs)), comm)

            if nbr_ctr > 0:
                eq_ctrs = model.constraint(phi, ste_fcs)
                ctrs = global_scalar_list([form(c) for c in eq_ctrs], comm)

                if rank == 0:
                    lm = cls_meth.run(cost, ctrs)
                else:
                    lm = None
                lm = comm.bcast(lm, root=0)
                for i in range(nbr_ctr):
                    L[i].value = lm[i]

            if nbr_adj > 0:
                adj_eqs = model.adjoint(phi, ste_fcs)
                if not model.mixed_space:
                    phifem_solve(nbr_adj, adj_eqs, adj_fcs, phi, rank_dim, comm)
                else:
                    phifem_solve_mixed(nbr_adj, adj_eqs, adj_fcs, model.map)
                comm.barrier()

            if nbr_to_ev_qs > 0:
                qs_eval = global_scalar_list(to_ev_qs, comm)

            cls_vlty.run(tht)  # model.dx

            nder = global_scalar(nDJ, comm, np.sqrt)

            if rank == 0:
                lset_steps, lset_end = lsearch.get(nder)
            else:
                lset_steps, lset_end = None, None
            lset_steps = comm.bcast(lset_steps, root=0)
            lset_end = comm.bcast(lset_end, root=0)

            xdmf.write_function(phi, iter)
            if degree_space == 1:
                for f in ste_fcs:
                    xdmf.write_function(f, iter)
                for f in adj_fcs:
                    xdmf.write_function(f, iter)
            else:
                ste_fcsP1 = interpolate(ste_fcs, spaceP1, name="u")
                adj_fcsP1 = interpolate(adj_fcs, spaceP1, name="p")
                for f in ste_fcsP1:
                    xdmf.write_function(f, iter)
                for f in adj_fcsP1:
                    xdmf.write_function(f, iter)

            xdmf.write_function(tht, iter)

            if rank == 0:
                if nbr_ctr > 0:
                    print0(iter, cost, ctrs, nder, lset_steps, cls_meth.see())
                else:
                    print1(iter, cost, nder, lset_steps)
                tosave.add(cost, nder)
                if nbr_to_ev_qs > 0:
                    tosave.add_quantities(qs_eval)

                if iter > start_to_check:
                    if nbr_ctr > 0:
                        ctrn_errs = [c - 1.0 for c in ctrs]
                        lgrn_last = cls_meth.list_Lg[-1]
                        lgrn_errs = [l - lgrn_last for l in cls_meth.list_Lg[-prev:-1]]
                        cond1 = Norm(ctrn_errs, np.inf) < ctrn_tol
                        cond2 = Norm(lgrn_errs, np.inf) < lgrn_tol * abs(lgrn_last)
                        stop_flag = cond1 and cond2
                    else:
                        cost_last = tosave.cost[-1]
                        cost_diff = [j - cost_last for j in tosave.cost[-prev:-1]]
                        stop_flag = Norm(cost_diff, np.inf) < cost_tol * abs(cost_last)

            stop_flag = comm.bcast(stop_flag, root=0)

            if stop_flag:
                if rank == 0:
                    print("> Stopping condition reached!")
                break

    local_solve = MPI.Wtime() - start_solve
    max_solve = comm.allreduce(local_solve, op=MPI.MAX)

    if rank == 0:
        if nbr_ctr > 0:
            tosave.add_ppl(cls_meth)
        tosave.add_times(max_assembly, max_solve)
        tosave.save(model.path)
        print(f"> Assembly time = {max_assembly} s")
        print(f"> Resolution time = {max_solve} s")

    return phi


def runDP(
    model: Model,
    niter: int,
    reinit_step: int | bool,
    reinit_pars: Tuple[int, float],
    dfactor: float,
    lv_time: Tuple[float, float],
    lv_iter: Tuple[int, int],
    smooth: bool,
    start_to_check: int,
    ctrn_tol: float,
    lgrn_tol: float,
    cost_tol: float,
    prev: int,
    random_pars: Tuple[int, float],
) -> Function:
    """
    Implements Data Parallelism.
    """

    start_assembly = MPI.Wtime()

    rein_steps, rein_end = reinit_pars
    min_time, max_time = lv_time
    min_iter, max_iter = lv_iter
    seed, noise = random_pars

    # Constants ===========================
    filename = model.path / f"{res_name}.xdmf"
    stop_flag = False

    dim = model.dim
    domain = model.domain

    vol = volume(domain, comm)
    nfems = nbr_fems(domain, dim, comm)

    if rank == 0:
        diam2 = get_diam2(dim, vol, nfems)
        np.random.seed(seed)
        lsearch = AdapTime((min_time, max_time), (min_iter, max_iter), dfactor, noise)
        tosave = Save()
    else:
        diam2 = None

    diam2 = comm.bcast(diam2, root=0)

    # List of variables
    # -----------------
    # sp_lset : space of level set functions
    # sp_vlty : space of velocity fields
    # phi : level set function
    # tht : velocity field
    # ste_eqs : list of state equations
    # adj_eqs : list of adjoint equations
    # ste_fcs : list of state functions
    # adj_fcs : list of adjoint functions
    # ste_pbs : list of state problems
    # adj_pbs : list of adjoint problems
    # eq_ctrs : list of equality constraints
    # nbr_ste : number of states
    # nbr_adj : number of adjoints
    # nbr_ctr : number of constraints
    # J : cost functional
    # C : list of constraints
    # L : list of lagrange multipliers
    # S0, S1 : derivative components

    # Level set function
    sp_lset = create_space(domain, "CG", 1)
    phi = Function(sp_lset)
    phi.name = "phi"
    # Velocity field
    sp_vlty = create_space(domain, "CG", dim)
    tht = Function(sp_vlty)
    tht.name = "tht"

    # State equations/functions
    ste_eqs = model.pde(phi)
    nbr_ste = len(ste_eqs)
    ste_fcs = [Function(model.space) for _ in range(nbr_ste)]
    for i in range(nbr_ste):
        ste_fcs[i].name = "u" + str(i)

    # Solvers creation
    ste_pbs = []
    for i in range(nbr_ste):
        ste_problem = ste_eqs[i]
        if len(ste_problem) == 2:
            # linea problem
            weak_form, bcs = ste_problem
            bi, li = system(weak_form)
            ste_pbs.append(create_solver(form(bi), form(li), bcs, ste_fcs[i]))
            # save_info_solver = SaveInfoSolver(ste_pbs[0].solver)
        elif len(ste_problem) == 5:
            # nonlinear: newton with initial guess
            weak_form, bcs, jacobian, seudo_state, ini_func = ste_problem
            true_weak_form = replace(weak_form, {seudo_state: ste_fcs[i]})
            true_jacobian = replace(jacobian, {seudo_state: ste_fcs[i]})
            ste_pbs.append(
                create_nonlinear_solver(
                    comm, true_weak_form, bcs, true_jacobian, ste_fcs[i], ini_func
                )
            )
        elif model.ini_linear:
            weak_form, bcs, jacobian, seudo_state, ini_linear, factor_pars = ste_problem
            factor, nbr_newton_steps = factor_pars
            true_factor = const(domain, 0.0)
            true_weak_form = replace(
                weak_form, {seudo_state: ste_fcs[i], factor: true_factor}
            )
            true_jacobian = replace(
                jacobian, {seudo_state: ste_fcs[i], factor: true_factor}
            )
            ste_pbs.append(
                create_nonlinear_solver_with_ini_linear_and_factor(
                    comm,
                    true_weak_form,
                    bcs,
                    true_jacobian,
                    ste_fcs[i],
                    ini_linear,
                    true_factor,
                    nbr_newton_steps,
                )
            )
        else:
            # nonlinear: newton with initial guess and stepping method
            weak_form, bcs, jacobian, seudo_state, ini_func, factor_pars = ste_problem
            factor, nbr_newton_steps = factor_pars
            true_factor = const(domain, 0.0)
            true_weak_form = replace(
                weak_form, {seudo_state: ste_fcs[i], factor: true_factor}
            )
            true_jacobian = replace(
                jacobian, {seudo_state: ste_fcs[i], factor: true_factor}
            )
            ste_pbs.append(
                create_nonlinear_solver_with_factor(
                    comm,
                    true_weak_form,
                    bcs,
                    true_jacobian,
                    ste_fcs[i],
                    ini_func,
                    true_factor,
                    nbr_newton_steps,
                )
            )

    # Adjoint equations/functions
    adj_eqs = model.adjoint(phi, ste_fcs)
    nbr_adj = len(adj_eqs)
    if nbr_adj > 0:
        adj_fcs = [Function(model.space) for _ in range(nbr_adj)]
        for i in range(nbr_adj):
            adj_fcs[i].name = "p" + str(i)

        # Solvers creation
        adj_pbs = []
        for i in range(nbr_adj):
            adj_problem = adj_eqs[i]
            weak_form, bcs = adj_problem
            bi, li = system(weak_form)
            adj_pbs.append(create_solver(form(bi), form(li), bcs, adj_fcs[i]))
    else:
        adj_fcs = []

    # to be evaluated: functions
    nbr_to_ev_fs = len(model._to_eval["function"])
    if nbr_to_ev_fs > 0:
        fcs_to_eval = [Function(sp_lset) for _ in range(nbr_to_ev_fs)]
        for _, name in zip(fcs_to_eval, model._to_eval["function"]):
            _.name = name

    # to be evaluated: quantities
    nbr_to_ev_qs = len(model._to_eval["quantity"])
    if nbr_to_ev_qs > 0:
        to_ev_qs = []
        qs_names = []

        for name, fc in model._to_eval["quantity"].items():
            to_ev_qs.append(form(fc(model, phi, ste_fcs, adj_fcs)))
            qs_names.append(name)

        if rank == 0:
            tosave.add_qtty_names(qs_names)

    # Cost functional
    J = form(model.cost(phi, ste_fcs))
    # Derivative components
    S0_cts, S1_cts = model.derivative(phi, ste_fcs, adj_fcs)
    S0 = S0_cts[0]
    S1 = S1_cts[0]
    # Equality constraints
    eq_ctrs = model.constraint(phi, ste_fcs)
    nbr_ctr = len(eq_ctrs)
    if nbr_ctr > 0:
        # Compilation of the constraints
        C = [form(c) for c in eq_ctrs]
        # Lagrange multipliers
        L = [const(domain, 0) for _ in range(nbr_ctr)]
        # Creation of the derivatives components
        for i in range(nbr_ctr):
            S0 += L[i] * S0_cts[1][i]
            S1 += L[i] * S1_cts[1][i]
    # Derivative norm
    nDJ = form((model.bilinear_form(tht, tht))[0])

    # To calculate the velocity field
    cls_vlty = Velocity(dim, domain, sp_vlty, model.bilinear_form, S0, S1)

    # To calculate the level set function
    cls_lset = Level(domain, sp_lset, phi, tht, diam2, smooth)

    # Reinicialization
    if reinit_step:
        cls_rein = Reinit(domain, sp_lset, phi, diam2)

    local_assembly = MPI.Wtime() - start_assembly
    max_assembly = comm.allreduce(local_assembly, op=MPI.MAX)

    # Iteration i = 0 =======
    start_solve = MPI.Wtime()

    phi.interpolate(model._get_initial_level())

    [p.solve() for p in ste_pbs]
    comm.Barrier()
    # save_info_solver.add()

    cost = global_scalar(J, comm)

    if nbr_ctr > 0:
        ctrs = global_scalar_list(C, comm)
        if rank == 0:
            cls_meth = PPL(nbr_ctr, cost, ctrs)

    if nbr_adj > 0:
        [p.solve() for p in adj_pbs]
        comm.barrier()

    if nbr_to_ev_fs > 0:
        [
            fc.interpolate(
                L2_interpolation(
                    domain, sp_lset, func(model, phi, ste_fcs, adj_fcs), diam2
                )
            )
            for fc, func in zip(fcs_to_eval, model._to_eval["function"].values())
        ]

    if nbr_to_ev_qs > 0:
        qs_eval = global_scalar_list(to_ev_qs, comm)

    cls_vlty.run(tht)
    nder = global_scalar(nDJ, comm, np.sqrt)

    if rank == 0:
        lset_steps, lset_end = lsearch.get(nder)
    else:
        lset_steps, lset_end = None, None
    lset_steps = comm.bcast(lset_steps, root=0)
    lset_end = comm.bcast(lset_end, root=0)

    # print ==============================================
    if rank == 0:
        print("> Iterations:")
        if nbr_ctr > 0:
            print0(0, cost, ctrs, nder, 0, cls_meth.see())
        else:
            print1(0, cost, nder, 0)
        tosave.add(cost, nder)
        if nbr_to_ev_qs > 0:
            tosave.add_quantities(qs_eval)
    # ====================================================
    spaceP1 = None
    degree_space = model.space.ufl_element().degree
    if degree_space > 1:
        rank_space = model.space.value_shape
        if len(rank_space) == 1:
            rank_space = rank_space[0]
        spaceP1 = create_space(domain, "CG", rank=rank_space, degree=1)

    with XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(phi, 0)

        if degree_space == 1:
            for f in ste_fcs:
                xdmf.write_function(f, 0)
            for f in adj_fcs:
                xdmf.write_function(f, 0)
        else:
            ste_fcsP1 = interpolate(ste_fcs, spaceP1, name="u")
            adj_fcsP1 = interpolate(adj_fcs, spaceP1, name="p")
            for f in ste_fcsP1:
                xdmf.write_function(f, 0)
            for f in adj_fcsP1:
                xdmf.write_function(f, 0)

        xdmf.write_function(tht, 0)

        if nbr_to_ev_fs > 0:
            for f in fcs_to_eval:
                xdmf.write_function(f, 0)

        for iter in range(1, niter + 1):

            cls_lset.run(phi, lset_steps, lset_end)
            # Reinitialization
            if reinit_step and iter > start_to_check:
                if iter % reinit_step == 0:
                    cls_rein.run(phi, rein_steps, rein_end)

            [p.solve() for p in ste_pbs]
            comm.barrier()
            # save_info_solver.add()

            cost = global_scalar(J, comm)

            if nbr_ctr > 0:
                ctrs = global_scalar_list(C, comm)
                if rank == 0:
                    lm = cls_meth.run(cost, ctrs)
                else:
                    lm = None
                lm = comm.bcast(lm, root=0)
                for i in range(nbr_ctr):
                    L[i].value = lm[i]

            if nbr_adj > 0:
                [p.solve() for p in adj_pbs]
                comm.barrier()

            if nbr_to_ev_fs > 0:
                [
                    fc.interpolate(
                        L2_interpolation(
                            domain, sp_lset, func(model, phi, ste_fcs, adj_fcs), diam2
                        )
                    )
                    for fc, func in zip(
                        fcs_to_eval, model._to_eval["function"].values()
                    )
                ]

            if nbr_to_ev_qs:
                qs_eval = global_scalar_list(to_ev_qs, comm)

            cls_vlty.run(tht)

            nder = global_scalar(nDJ, comm, np.sqrt)

            if rank == 0:
                lset_steps, lset_end = lsearch.get(nder)
            else:
                lset_steps, lset_end = None, None
            lset_steps = comm.bcast(lset_steps, root=0)
            lset_end = comm.bcast(lset_end, root=0)

            xdmf.write_function(phi, iter)
            if degree_space == 1:
                for f in ste_fcs:
                    xdmf.write_function(f, iter)
                for f in adj_fcs:
                    xdmf.write_function(f, iter)
            else:
                ste_fcsP1 = interpolate(ste_fcs, spaceP1, name="u")
                adj_fcsP1 = interpolate(adj_fcs, spaceP1, name="p")
                for f in ste_fcsP1:
                    xdmf.write_function(f, iter)
                for f in adj_fcsP1:
                    xdmf.write_function(f, iter)

            if nbr_to_ev_fs > 0:
                for f in fcs_to_eval:
                    xdmf.write_function(f, iter)

            xdmf.write_function(tht, iter)

            if rank == 0:
                if nbr_ctr > 0:
                    print0(iter, cost, ctrs, nder, lset_steps, cls_meth.see())
                else:
                    print1(iter, cost, nder, lset_steps)
                tosave.add(cost, nder)
                if nbr_to_ev_qs > 0:
                    tosave.add_quantities(qs_eval)

                if iter > start_to_check:
                    if nbr_ctr > 0:
                        ctrn_errs = [c - 1.0 for c in ctrs]
                        lgrn_last = cls_meth.list_Lg[-1]
                        lgrn_errs = [l - lgrn_last for l in cls_meth.list_Lg[-prev:-1]]
                        cond1 = Norm(ctrn_errs, np.inf) < ctrn_tol
                        cond2 = Norm(lgrn_errs, np.inf) < lgrn_tol * abs(lgrn_last)
                        stop_flag = cond1 and cond2
                    else:
                        cost_last = tosave.cost[-1]
                        cost_diff = [j - cost_last for j in tosave.cost[-prev:-1]]
                        stop_flag = Norm(cost_diff, np.inf) < cost_tol * abs(cost_last)

            stop_flag = comm.bcast(stop_flag, root=0)

            if stop_flag:
                if rank == 0:
                    print("> Stopping condition reached!")
                break

    local_solve = MPI.Wtime() - start_solve
    max_solve = comm.allreduce(local_solve, op=MPI.MAX)

    if rank == 0:
        if nbr_ctr > 0:
            tosave.add_ppl(cls_meth)
        tosave.add_times(max_assembly, max_solve)
        tosave.save(model.path)
        # save_info_solver.save(model.path)
        print(f"> Assembly time = {max_assembly} s")
        print(f"> Resolution time = {max_solve} s")

    return phi


def runTP(
    model: Model,
    niter: int,
    reinit_step: int | bool,
    reinit_pars: Tuple[int, float],
    dfactor: float,
    lv_time: Tuple[float, float],
    lv_iter: Tuple[int, int],
    smooth: bool,
    start_to_check: int,
    ctrn_tol: float,
    lgrn_tol: float,
    cost_tol: float,
    prev: int,
    random_pars: Tuple[int, float],
) -> Function:
    """
    Implements Task Parallelism.
    """

    start_assembly = MPI.Wtime()

    rein_steps, rein_end = reinit_pars
    min_time, max_time = lv_time
    min_iter, max_iter = lv_iter
    seed, noise = random_pars

    # Constants ===========================
    filename = model.path / f"{res_name}.xdmf"
    stop_flag = False

    dim = model.dim
    domain = model.domain

    if rank == 0:
        vol = volume(domain, MPI.COMM_SELF)
        nfems = nbr_fems(domain, dim, MPI.COMM_SELF)
        diam2 = get_diam2(dim, vol, nfems)
        np.random.seed(seed)
        lsearch = AdapTime((min_time, max_time), (min_iter, max_iter), dfactor, noise)
        tosave = Save()
    else:
        vol = None
        nfems = None
        diam2 = None

    vol = comm.bcast(vol, root=0)
    nfems = comm.bcast(nfems, root=0)
    diam2 = comm.bcast(diam2, root=0)

    # Level set function
    sp_lset = create_space(domain, "CG", 1)
    phi = Function(sp_lset)
    phi.name = "phi"
    # Velocity field
    sp_vlty = create_space(domain, "CG", dim)
    tht = Function(sp_vlty)
    tht.name = "tht"

    # State equations/functions/problems
    ste_eqs = model.pde(phi)
    nbr_ste = len(ste_eqs)
    ste_fcs = [Function(model.space) for _ in range(nbr_ste)]
    for i in range(nbr_ste):
        ste_fcs[i].name = "u" + str(i)

    ste_problem = ste_eqs[rank]
    if len(ste_problem) == 2:
        weak_form, bcs = ste_problem
        bi, li = system(weak_form)
        ste_pb = create_solver(form(bi), form(li), bcs, ste_fcs[rank])
    elif len(ste_problem) == 5:
        weak_form, bcs, jacobian, seudo_state, ini_func = ste_problem
        true_weak_form = replace(weak_form, {seudo_state: ste_fcs[rank]})
        true_jacobian = replace(jacobian, {seudo_state: ste_fcs[rank]})
        ste_pb = create_nonlinear_solver(
            comm_self, true_weak_form, bcs, true_jacobian, ste_fcs[rank], ini_func
        )
    else:
        pass

    # Adjoint equations/functions/problems
    adj_eqs = model.adjoint(phi, ste_fcs)
    nbr_adj = len(adj_eqs)
    if nbr_adj > 0:
        adj_fcs = [Function(model.space) for _ in range(nbr_adj)]
        for i in range(nbr_adj):
            adj_fcs[i].name = "p" + str(i)

        adj_problem = adj_eqs[rank]
        weak_form, bcs = adj_problem
        bi, li = system(weak_form)
        adj_pb = create_solver(form(bi), form(li), bcs, adj_fcs[rank])
    else:
        adj_fcs = []

    if rank == 0:
        # Cost functional
        J = form(model.cost(phi, ste_fcs))
        # Derivative components
        S0_cts, S1_cts = model.derivative(phi, ste_fcs, adj_fcs)
        S0 = S0_cts[0]
        S1 = S1_cts[0]
        # Equality constraints
        eq_ctrs = model.constraint(phi, ste_fcs)
        nbr_ctr = len(eq_ctrs)
        if nbr_ctr > 0:
            # Compilation of the constraints
            C = [form(c) for c in eq_ctrs]
            # Lagrange multipliers
            L = [const(domain, 0) for _ in range(nbr_ctr)]
            # Creation of the derivatives components
            for i in range(nbr_ctr):
                S0 += L[i] * S0_cts[1][i]
                S1 += L[i] * S1_cts[1][i]
        # Derivative norm
        nDJ = form((model.bilinear_form(tht, tht))[0])
        # To calculate the velocity field
        cls_vlty = Velocity(dim, domain, sp_vlty, model.bilinear_form, S0, S1)
        # To calculate the level set function
        cls_lset = Level(domain, sp_lset, phi, tht, diam2, smooth)
        # Reinitialization
        if reinit_step:
            cls_rein = Reinit(domain, sp_lset, phi, diam2)

    local_assembly = MPI.Wtime() - start_assembly
    max_assembly = comm.allreduce(local_assembly, op=MPI.MAX)

    # Iteration i = 0 =======
    start_solve = MPI.Wtime()

    # --------------------------------------------
    if rank == 0:
        phi.interpolate(model._get_initial_level())
        phi_vls = phi.x.array[:]
    else:
        phi_vls = None
    phi.x.array[:] = comm.bcast(phi_vls, root=0)
    # ---------------------------------------------

    ste_pb.solve()
    comm.barrier()

    # ---------------------------------------------------
    # Alternative form, without ste_vls -----------------
    # ---------------------------------------------------
    # for i in range(size):
    # 	if rank == i:
    # 		u_i = ste_fcs[i].x.array[:]
    # 	else:
    # 		u_i = None
    # 	ste_fcs[i].x.array[:] = comm.bcast(u_i, root = i)
    # ---------------------------------------------------

    # ------------------------------------------------
    ste_vls = comm.allgather(ste_fcs[rank].x.array[:])
    for i in range(size):
        if i == rank:
            continue
        ste_fcs[i].x.array[:] = ste_vls[i]
    # ------------------------------------------------

    comm.barrier()

    if rank == 0:
        cost = global_scalar(J, MPI.COMM_SELF)
        if nbr_ctr > 0:
            ctrs = global_scalar_list(C, MPI.COMM_SELF)
            cls_meth = PPL(nbr_ctr, cost, ctrs)

    if nbr_adj > 0:
        adj_pb.solve()
        comm.barrier()

        # ---------------------------------------------------
        # Alternative form, without adj_vls -----------------
        # ---------------------------------------------------
        # for i in range(size):
        # 	if rank == i:
        # 		p_i = adj_fcs[i].x.array[:]
        # 	else:
        # 		p_i = None
        # 	adj_fcs[i].x.array[:] = comm.bcast(p_i, root = i)
        # ---------------------------------------------------

        # ------------------------------------------------
        adj_vls = comm.allgather(adj_fcs[rank].x.array[:])
        for i in range(size):
            if i == rank:
                continue
            adj_fcs[i].x.array[:] = adj_vls[i]
        # ------------------------------------------------
        comm.barrier()

    if rank == 0:
        cls_vlty.run(tht)
        nder = global_scalar(nDJ, MPI.COMM_SELF, np.sqrt)
        lset_steps, lset_end = lsearch.get(nder)

    # print ==============================================
    if rank == 0:
        print("> Iterations:")
        if nbr_ctr > 0:
            print0(0, cost, ctrs, nder, 0, cls_meth.see())
        else:
            print1(0, cost, nder, 0)
        tosave.add(cost, nder)
    # ====================================================

    if rank == 0:
        xdmf = XDMFFile(MPI.COMM_SELF, filename, "w")
        xdmf.write_mesh(domain)
        xdmf.write_function(phi, 0)
        for f in ste_fcs:
            xdmf.write_function(f, 0)
        for f in adj_fcs:
            xdmf.write_function(f, 0)
        xdmf.write_function(tht, 0)

    for iter in range(1, niter + 1):
        # -------------------------------------------------
        if rank == 0:
            cls_lset.run(phi, lset_steps, lset_end)
            # Reinitialization
            if reinit_step and iter > start_to_check:
                if iter % reinit_step == 0:
                    cls_rein.run(phi, rein_steps, rein_end)
            phi_vls = phi.x.array[:]
        else:
            phi_vls = None
        phi.x.array[:] = comm.bcast(phi_vls, root=0)
        # -------------------------------------------------

        ste_pb.solve()
        comm.barrier()

        # ------------------------------------------------
        ste_vls = comm.allgather(ste_fcs[rank].x.array[:])
        for i in range(size):
            if i == rank:
                continue
            ste_fcs[i].x.array[:] = ste_vls[i]
        # ------------------------------------------------

        comm.barrier()

        if rank == 0:
            cost = global_scalar(J, MPI.COMM_SELF)
            if nbr_ctr > 0:
                ctrs = global_scalar_list(C, MPI.COMM_SELF)
                lm = cls_meth.run(cost, ctrs)
                for i in range(nbr_ctr):
                    L[i].value = lm[i]

        if nbr_adj > 0:
            adj_pb.solve()
            comm.barrier()

            # ------------------------------------------------
            adj_vls = comm.allgather(adj_fcs[rank].x.array[:])
            for i in range(size):
                if i == rank:
                    continue
                adj_fcs[i].x.array[:] = adj_vls[i]
            # ------------------------------------------------
            comm.barrier()

        if rank == 0:
            cls_vlty.run(tht)
            nder = global_scalar(nDJ, MPI.COMM_SELF, np.sqrt)
            lset_steps, lset_end = lsearch.get(nder)

        if rank == 0:
            xdmf.write_function(phi, iter)
            for f in ste_fcs:
                xdmf.write_function(f, iter)
            for f in adj_fcs:
                xdmf.write_function(f, iter)
            xdmf.write_function(tht, iter)

        if rank == 0:
            if nbr_ctr > 0:
                print0(iter, cost, ctrs, nder, lset_steps, cls_meth.see())
            else:
                print1(iter, cost, nder, lset_steps)
            tosave.add(cost, nder)

            if iter > start_to_check:
                if nbr_ctr > 0:
                    ctrn_errs = [c - 1.0 for c in ctrs]
                    lgrn_last = cls_meth.list_Lg[-1]
                    lgrn_errs = [l - lgrn_last for l in cls_meth.list_Lg[-prev:-1]]
                    cond1 = Norm(ctrn_errs, np.inf) < ctrn_tol
                    cond2 = Norm(lgrn_errs, np.inf) < lgrn_tol * abs(lgrn_last)
                    stop_flag = cond1 and cond2
                else:
                    cost_last = tosave.cost[-1]
                    cost_diff = [j - cost_last for j in tosave.cost[-prev:-1]]
                    stop_flag = Norm(cost_diff, np.inf) < cost_tol * abs(cost_last)

        stop_flag = comm.bcast(stop_flag, root=0)

        if stop_flag:
            if rank == 0:
                print("> Stopping condition reached!")
            break

    if rank == 0:
        xdmf.close()

    local_solve = MPI.Wtime() - start_solve
    max_solve = comm.allreduce(local_solve, op=MPI.MAX)

    if rank == 0:
        if nbr_ctr > 0:
            tosave.add_ppl(cls_meth)
        tosave.add_times(max_assembly, max_solve)
        tosave.save(model.path)
        print(f"> Assembly time = {max_assembly} s")
        print(f"> Resolution time = {max_solve} s")

    return phi


def runMP(
    sub_comm: MPI.Comm,
    model: Model,
    niter: int,
    reinit_step: int | bool,
    reinit_pars: Tuple[int, float],
    dfactor: float,
    lv_time: Tuple[float, float],
    lv_iter: Tuple[int, int],
    smooth: bool,
    start_to_check: int,
    ctrn_tol: float,
    lgrn_tol: float,
    cost_tol: float,
    prev: int,
    random_pars: Tuple[int, float],
) -> Function:
    """
    Implements Mix Parallelism.
    """

    start_assembly = MPI.Wtime()

    rein_steps, rein_end = reinit_pars
    min_time, max_time = lv_time
    min_iter, max_iter = lv_iter
    seed, noise = random_pars

    group_size = sub_comm.size
    nbr_groups = size // group_size
    color = rank * nbr_groups // size
    sub_rank = sub_comm.rank

    # Constants ===========================
    filename = model.path / f"{res_name}.xdmf"
    stop_flag = False

    dim = model.dim
    domain = model.domain

    if color == 0:
        vol = volume(domain, sub_comm)
        nfems = nbr_fems(domain, dim, sub_comm)

    if rank == 0:
        diam2 = get_diam2(dim, vol, nfems)
        np.random.seed(seed)
        lsearch = AdapTime((min_time, max_time), (min_iter, max_iter), dfactor, noise)
        tosave = Save()
    else:
        diam2 = None

    diam2 = comm.bcast(diam2, root=0)

    # Level set function
    sp_lset = create_space(domain, "CG", 1)
    phi = Function(sp_lset)
    phi.name = "phi"
    # Velocity field
    sp_vlty = create_space(domain, "CG", dim)
    tht = Function(sp_vlty)
    tht.name = "tht"

    # State equations/functions/problems
    ste_eqs = model.pde(phi)
    nbr_ste = len(ste_eqs)
    ste_fcs = [Function(model.space) for _ in range(nbr_ste)]
    for i in range(nbr_ste):
        ste_fcs[i].name = "u" + str(i)

    ste_problem = ste_eqs[color]
    if len(ste_problem) == 2:
        weak_form, bcs = ste_problem
        bi, li = system(weak_form)
        ste_pb = create_solver(form(bi), form(li), bcs, ste_fcs[color])
    elif len(ste_problem) == 5:
        weak_form, bcs, jacobian, seudo_state, ini_func = ste_problem
        true_weak_form = replace(weak_form, {seudo_state: ste_fcs[color]})
        true_jacobian = replace(jacobian, {seudo_state: ste_fcs[color]})
        ste_pb = create_nonlinear_solver(
            sub_comm, true_weak_form, bcs, true_jacobian, ste_fcs[color], ini_func
        )
    else:
        pass

    # Adjoint equations/functions/problems
    adj_eqs = model.adjoint(phi, ste_fcs)
    nbr_adj = len(adj_eqs)
    if nbr_adj > 0:
        adj_fcs = [Function(model.space) for _ in range(nbr_adj)]
        for i in range(nbr_adj):
            adj_fcs[i].name = "p" + str(i)

        adj_problem = adj_eqs[color]
        weak_form, bcs = adj_problem
        bi, li = system(weak_form)
        adj_pb = create_solver(form(bi), form(li), bcs, adj_fcs[color])
    else:
        adj_fcs = []

    if color == 0:
        # Cost functional
        J = form(model.cost(phi, ste_fcs))
        # Derivative components
        S0_cts, S1_cts = model.derivative(phi, ste_fcs, adj_fcs)
        S0 = S0_cts[0]
        S1 = S1_cts[0]
        # Equality constraints
        eq_ctrs = model.constraint(phi, ste_fcs)
        nbr_ctr = len(eq_ctrs)
        if nbr_ctr > 0:
            # Compilation of the constraints
            C = [form(c) for c in eq_ctrs]
            # Lagrange multipliers
            L = [const(domain, 0) for _ in range(nbr_ctr)]
            # Creation of the derivatives components
            for i in range(nbr_ctr):
                S0 += L[i] * S0_cts[1][i]
                S1 += L[i] * S1_cts[1][i]
        # Derivative norm
        nDJ = form((model.bilinear_form(tht, tht))[0])
        # To calculate the velocity field
        cls_vlty = Velocity(dim, domain, sp_vlty, model.bilinear_form, S0, S1)
        # To calculate the level set function
        cls_lset = Level(domain, sp_lset, phi, tht, diam2, smooth)
        # Reinicialization
        if reinit_step:
            cls_rein = Reinit(domain, sp_lset, phi, diam2)

    local_assembly = MPI.Wtime() - start_assembly
    max_assembly = comm.allreduce(local_assembly, op=MPI.MAX)

    # Iteration i = 0 =======
    start_solve = MPI.Wtime()

    # ------------------------------------
    if color == 0:
        phi.interpolate(model._get_initial_level())
        phi_vls_loc = phi.x.array[:]
    else:
        # Colect None if color is not 0
        phi_vls_loc = None
    phi_vls = comm.allgather(phi_vls_loc)
    if color > 0:
        phi.x.array[:] = phi_vls[sub_rank]
    # ------------------------------------

    comm.barrier()
    ste_pb.solve()
    comm.barrier()

    # ----------------------------------------------------------
    ste_vls = comm.allgather(ste_fcs[color].x.array[:])
    for i in range(nbr_groups):
        if i == color:
            continue
        ste_fcs[i].x.array[:] = ste_vls[i * group_size + sub_rank]
    # ----------------------------------------------------------

    comm.barrier()

    if color == 0:
        cost = global_scalar(J, sub_comm)
        if nbr_ctr > 0:
            ctrs = global_scalar_list(C, sub_comm)
            if rank == 0:
                cls_meth = PPL(nbr_ctr, cost, ctrs)

    if nbr_adj > 0:
        adj_pb.solve()
        comm.barrier()

        # ----------------------------------------------------------
        adj_vls = comm.allgather(adj_fcs[color].x.array[:])
        for i in range(nbr_groups):
            if i == color:
                continue
            adj_fcs[i].x.array[:] = adj_vls[i * group_size + sub_rank]
        # ----------------------------------------------------------
        comm.barrier()

    if color == 0:
        cls_vlty.run(tht)
        nder = global_scalar(nDJ, sub_comm, np.sqrt)

        if rank == 0:
            lset_steps, lset_end = lsearch.get(nder)
        else:
            lset_steps, lset_end = None, None
        lset_steps = sub_comm.bcast(lset_steps, root=0)
        lset_end = sub_comm.bcast(lset_end, root=0)

    # print ==============================================
    if rank == 0:
        print("> Iterations:")
        if nbr_ctr > 0:
            print0(0, cost, ctrs, nder, 0, cls_meth.see())
        else:
            print1(0, cost, nder, 0)
        tosave.add(cost, nder)
    # ====================================================

    if color == 0:
        xdmf = XDMFFile(sub_comm, filename, "w")
        xdmf.write_mesh(domain)
        xdmf.write_function(phi, 0)
        for f in ste_fcs:
            xdmf.write_function(f, 0)
        for f in adj_fcs:
            xdmf.write_function(f, 0)
        xdmf.write_function(tht, 0)

    for iter in range(1, niter + 1):
        # -------------------------------------------------
        if color == 0:
            cls_lset.run(phi, lset_steps, lset_end)
            # Reinitialization
            if reinit_step and iter > start_to_check:
                if iter % reinit_step == 0:
                    cls_rein.run(phi, rein_steps, rein_end)
            phi_vls_loc = phi.x.array[:]
        else:
            phi_vls_loc = None
        phi_vls = comm.allgather(phi_vls_loc)
        if color > 0:
            phi.x.array[:] = phi_vls[sub_rank]
        # -------------------------------------------------

        comm.barrier()
        ste_pb.solve()
        comm.barrier()

        # ----------------------------------------------------------
        ste_vls = comm.allgather(ste_fcs[color].x.array[:])
        for i in range(nbr_groups):
            if i == color:
                continue
            ste_fcs[i].x.array[:] = ste_vls[i * group_size + sub_rank]
        # ----------------------------------------------------------

        comm.barrier()

        if color == 0:
            cost = global_scalar(J, sub_comm)
            if nbr_ctr > 0:
                ctrs = global_scalar_list(C, sub_comm)
                if rank == 0:
                    lm = cls_meth.run(cost, ctrs)
                else:
                    lm = None
                lm = sub_comm.bcast(lm, root=0)
                for i in range(nbr_ctr):
                    L[i].value = lm[i]

        if nbr_adj > 0:
            adj_pb.solve()
            comm.barrier()

            # ----------------------------------------------------------
            adj_vls = comm.allgather(adj_fcs[color].x.array[:])
            for i in range(nbr_groups):
                if i == color:
                    continue
                adj_fcs[i].x.array[:] = adj_vls[i * group_size + sub_rank]
            # ----------------------------------------------------------
            comm.barrier()

        if color == 0:
            cls_vlty.run(tht)
            nder = global_scalar(nDJ, sub_comm, np.sqrt)

            if rank == 0:
                lset_steps, lset_end = lsearch.get(nder)
            else:
                lset_steps, lset_end = None, None
            lset_steps = sub_comm.bcast(lset_steps, root=0)
            lset_end = sub_comm.bcast(lset_end, root=0)

        if color == 0:
            xdmf.write_function(phi, iter)
            for f in ste_fcs:
                xdmf.write_function(f, iter)
            for f in adj_fcs:
                xdmf.write_function(f, iter)
            xdmf.write_function(tht, iter)

        if rank == 0:
            if nbr_ctr > 0:
                print0(iter, cost, ctrs, nder, lset_steps, cls_meth.see())
            else:
                print1(iter, cost, nder, lset_steps)
            tosave.add(cost, nder)

            if iter > start_to_check:
                if nbr_ctr > 0:
                    ctrn_errs = [c - 1.0 for c in ctrs]
                    lgrn_last = cls_meth.list_Lg[-1]
                    lgrn_errs = [l - lgrn_last for l in cls_meth.list_Lg[-prev:-1]]
                    cond1 = Norm(ctrn_errs, np.inf) < ctrn_tol
                    cond2 = Norm(lgrn_errs, np.inf) < lgrn_tol * abs(lgrn_last)
                    stop_flag = cond1 and cond2
                else:
                    cost_last = tosave.cost[-1]
                    cost_diff = [j - cost_last for j in tosave.cost[-prev:-1]]
                    stop_flag = Norm(cost_diff, np.inf) < cost_tol * abs(cost_last)

        stop_flag = comm.bcast(stop_flag, root=0)

        if stop_flag:
            if rank == 0:
                print("> Stopping condition reached!")
            break

    if color == 0:
        xdmf.close()

    local_solve = MPI.Wtime() - start_solve
    max_solve = comm.allreduce(local_solve, op=MPI.MAX)

    if rank == 0:
        if nbr_ctr > 0:
            tosave.add_ppl(cls_meth)
        tosave.add_times(max_assembly, max_solve)
        tosave.save(model.path)
        print(f"> Assembly time = {max_assembly} s")
        print(f"> Resolution time = {max_solve} s")

    return phi
