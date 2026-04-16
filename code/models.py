from formopt import Model

from ufl import (
    TrialFunction,
    TestFunction,
    FacetNormal,
    Identity,
    Measure,
    SpatialCoordinate,
    Coefficient,
    conditional,
    indices,
    as_vector,
    inner,
    outer,
    grad,
    sym,
    dot,
    lt,
    pi,
    cos,
    sqrt,
    nabla_div,
    Constant,
    det,
    inv,
    tr,
    ln,
)


"""
Models:
    Compliance
    CompliancePlus
    InverseElasticity
    Heat
    HeatPlus
    Logistic
    Mechanism
    Gripping
    SVK
"""


class Compliance(Model):
    """
    Models the compliance minimization
    problem for linear elasticity.
    """

    def __init__(self, dim, domain, space, g, ds_g, dir_bcs, vol, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)
        self.g = as_vector(g)
        self.ds_g = ds_g
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []

        E, nu = 1.0, 0.3
        lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        mu = E / 2.0 / (1.0 + nu)

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda w: sym(grad(w))
        self.sigma = lambda w: (
            lmbda * nabla_div(w) * self.Id + 2.0 * mu * self.epsilon(w)
        )
        self.A = lambda w: (conditional(lt(w, 0.0), 1.0, 1e-4))
        self.chi = lambda w: (conditional(lt(w, 0.0), 1.0, 0.0))
        self.Coef = 0.1

    def updateA(self, eps):
        self.A = lambda w: (conditional(lt(w, 0.0), 1.0, eps))

    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        su = self.sigma(u)
        ev = self.epsilon(v)

        W = self.A(phi) * inner(su, ev) * self.dx
        W -= dot(self.g, v) * self.ds_g

        return [(W, self.bc)]

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):

        u = U[0]
        su = self.sigma(u)
        eu = self.epsilon(u)

        J = self.A(phi) * (inner(su, eu)) * self.dx

        return J

    def constraint(self, phi, U):

        C = (1.0 / self.vol) * self.chi(phi) * self.dx

        return [C]

    def derivative(self, phi, U, P):

        u = U[0]
        su = self.sigma(u)
        eu = self.epsilon(u)

        S0_J = self.zero_vec
        S1_J = 2.0 * grad(u).T * su
        S1_J -= inner(su, eu) * self.Id
        S1_J *= self.A(phi)

        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        S0 = (S0_J, [S0_C])
        S1 = (S1_J, [S1_C])

        return S0, S1

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)

        B = self.Coef * dot(th, xi) * self.dx
        B += inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False


class CompliancePlus(Model):
    """
    Models a compliance optimization
    problem with multiple load cases
    """

    def __init__(self, dim, domain, space, g, ds_g, dir_bcs, vol, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)
        self.g = [as_vector(_) for _ in g]
        self.ds_g = ds_g
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []

        E, nu = 1.0, 0.3
        lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        mu = E / 2.0 / (1.0 + nu)

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda v: sym(grad(v))
        self.sigma = lambda v: (
            lmbda * nabla_div(v) * self.Id + 2.0 * mu * self.epsilon(v)
        )
        self.A = lambda v: (conditional(lt(v, 0.0), 1.0, 1e-4))
        self.chi = lambda w: (conditional(lt(w, 0.0), 1.0, 0.0))

    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        su = self.sigma(u)
        ev = self.epsilon(v)

        pdes = []
        for g, ds_g in zip(self.g, self.ds_g):
            W = self.A(phi) * inner(su, ev) * self.dx
            W -= dot(g, v) * ds_g
            pdes.append((W, self.bc))

        return pdes

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):

        J = []
        for u in U:
            su = self.sigma(u)
            eu = self.epsilon(u)
            J.append(self.A(phi) * inner(su, eu) * self.dx)

        return sum(J)

    def constraint(self, phi, U):

        C = (1.0 / self.vol) * self.chi(phi) * self.dx

        return [C]

    def derivative(self, phi, U, P):

        S0_J = self.zero_vec

        S1_J = []
        for u in U:
            su = self.sigma(u)
            eu = self.epsilon(u)
            s1 = 2.0 * grad(u).T * su
            s1 -= inner(su, eu) * self.Id
            S1_J.append(self.A(phi) * s1)

        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        S0 = (S0_J, [S0_C])
        S1 = (sum(S1_J), [S1_C])

        return S0, S1

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)

        B = 0.1 * dot(th, xi) * self.dx
        B += inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e5 * sb * dot(th, xi) * self.dx

        return B, False


class InverseElasticity(Model):

    def __init__(
        self,
        dim,
        domain,
        space,
        forces,
        ds_forces,
        ds1,
        dirbc_partial,
        dirbc_total,
        path,
    ):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.fs = [as_vector(f) for f in forces]
        self.dfs = ds_forces
        self.bcF = dirbc_partial
        self.bcG = dirbc_total
        self.ds1 = ds1

        self.gs = []
        self.N = len(self.fs)
        self.dx = Measure("dx", domain=domain)

        E, nu = 1.0, 0.3
        lm = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        mu = E / 2.0 / (1.0 + nu)
        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda w: sym(grad(w))
        self.sigma = lambda w: lm * nabla_div(w) * self.Id + 2.0 * mu * self.epsilon(w)
        self.A = lambda w: conditional(lt(w, 0.0), 10.0, 1.0)
        self.alpha = 1.0
        self.beta = 1.0
        self.phi = None

    def f_prob(self, u, w, phi, f, df):
        # f-problem

        su = self.sigma(u)
        ew = self.epsilon(w)

        W = self.A(phi) * inner(su, ew) * self.dx
        W -= dot(f, w) * df

        return (W, self.bcF)

    def g_prob(self, v, w, phi, g):
        # g-problem

        sv = self.sigma(v)
        sg = self.sigma(g)
        ew = self.epsilon(w)

        W = self.A(phi) * inner(sv, ew) * self.dx
        W += self.A(phi) * inner(sg, ew) * self.dx

        return (W, self.bcG)

    def adj_f_prob(self, p, r, phi, u, v, g):
        # adjoint of f-problem

        sp = self.sigma(p)
        er = self.epsilon(r)
        a_g = u - v - g
        b_g = u - g

        W = self.A(phi) * inner(sp, er) * self.dx
        W += self.alpha * dot(a_g, r) * self.dx
        W += self.beta * dot(b_g, r) * self.ds1

        return (W, self.bcF)

    def adj_g_prob(self, q, r, phi, u, v, g):
        # adjoint of g-problem

        sq = self.sigma(q)
        er = self.epsilon(r)
        a_g = u - v - g

        W = self.A(phi) * inner(sq, er) * self.dx
        W -= self.alpha * dot(a_g, r) * self.dx

        return (W, self.bcG)

    def pde0(self, phi):

        a = TrialFunction(self.space)
        b = TestFunction(self.space)

        F = [self.f_prob(a, b, phi, f, df) for f, df in zip(self.fs, self.dfs)]

        return F

    def pde(self, phi):
        # State problems
        a = TrialFunction(self.space)
        b = TestFunction(self.space)

        F = [self.f_prob(a, b, phi, f, df) for f, df in zip(self.fs, self.dfs)]
        G = [self.g_prob(a, b, phi, g) for g in self.gs]

        return F + G

    def adjoint(self, phi, U):
        # Adjoint problems
        a = TrialFunction(self.space)
        b = TestFunction(self.space)

        AF = [
            self.adj_f_prob(a, b, phi, u, v, g)
            for u, v, g in zip(U[: self.N], U[self.N :], self.gs)
        ]
        AG = [
            self.adj_g_prob(a, b, phi, u, v, g)
            for u, v, g in zip(U[: self.N], U[self.N :], self.gs)
        ]

        return AF + AG

    def cost(self, phi, U):
        # Cost functional
        uvg = zip(U[: self.N], U[self.N :], self.gs)
        ug = zip(U[: self.N], self.gs)

        Ja = [dot(u - v - g, u - v - g) * self.dx for u, v, g in uvg]

        Jb = [dot(u - g, u - g) * self.ds1 for u, g in ug]

        return (self.alpha / 2.0) * sum(Ja) + (self.beta / 2.0) * sum(Jb)

    def constraint(self, phi, U):
        # Constraint (empty list)
        return []

    def S0(self, u, v, q, g, phi):
        # Derivative component S0
        i, j, k = indices(3)
        sq = self.sigma(q)
        eg = self.epsilon(g)

        s0a = grad(g).T * (u - v - g)
        s0b = as_vector(sq[i, j] * (grad(eg))[i, j, k], (k))

        return -self.alpha * s0a + self.A(phi) * s0b

    def S1(self, u, v, p, q, g, phi):
        # Derivative component S1
        su = self.sigma(u)
        sp = self.sigma(p)
        sq = self.sigma(q)
        svg = self.sigma(v + g)
        ep = self.epsilon(p)
        eq = self.epsilon(q)
        uvg = u - v - g

        s1i = inner(su, ep) + inner(svg, eq)
        s1i = self.alpha * dot(uvg, uvg) / 2.0 + self.A(phi) * s1i

        s1j = grad(u).T * sp + grad(p).T * su
        s1j += grad(v).T * sq + grad(q).T * svg

        return s1i * self.Id - self.A(phi) * s1j

    def derivative(self, phi, U, P):

        fu = U[: self.N]
        gv = U[self.N :]
        fp = P[: self.N]
        gq = P[self.N :]

        uvqg = zip(fu, gv, gq, self.gs)
        uvpqg = zip(fu, gv, fp, gq, self.gs)

        S0 = [self.S0(u, v, q, g, phi) for u, v, q, g in uvqg]
        S1 = [self.S1(u, v, p, q, g, phi) for u, v, p, q, g in uvpqg]

        return (sum(S0), []), (sum(S1), [])

    def bilinear_form(self, th, xi):
        # Bilinear form to compute the velocity,
        # with Homogeneous Dirichlet boundary condition

        biform = 0.1 * dot(th, xi) * self.dx
        biform += inner(grad(th), grad(xi)) * self.dx

        return biform, True


class Heat(Model):

    def __init__(self, dim, domain, space, dir_bcs, vol, path, sc_type="Uniform"):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)

        self.A = lambda w: conditional(lt(w, 0.0), 1.0, 1e-3)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)

        self.sc_type = sc_type
        self.f = self.source()

    def source(self):
        if self.sc_type == "Uniform":
            return 1.0
        elif self.sc_type == "1Load":
            x0, y0 = 0.5, 0.5
            max_value, epsilon = 50.0, 0.1
            x = SpatialCoordinate(self.domain)
            r = sqrt((x[0] - x0) ** 2 + (x[1] - y0) ** 2)
            delta_expr = conditional(
                lt(r, epsilon), max_value * (1.0 + cos(pi * r / epsilon)) / 2.0, 0.0
            )
            return delta_expr
        elif self.sc_type == "4Loads":
            x1, y1 = 0.5, 0.25
            x2, y2 = 0.75, 0.5
            x3, y3 = 0.5, 0.75
            x4, y4 = 0.25, 0.5
            max_value, epsilon = 50.0, 0.05
            x = SpatialCoordinate(self.domain)
            r1 = sqrt((x[0] - x1) ** 2 + (x[1] - y1) ** 2)
            r2 = sqrt((x[0] - x2) ** 2 + (x[1] - y2) ** 2)
            r3 = sqrt((x[0] - x3) ** 2 + (x[1] - y3) ** 2)
            r4 = sqrt((x[0] - x4) ** 2 + (x[1] - y4) ** 2)
            delta1 = conditional(
                lt(r1, epsilon), max_value * (1.0 + cos(pi * r1 / epsilon)) / 2.0, 0.0
            )
            delta2 = conditional(
                lt(r2, epsilon), max_value * (1.0 + cos(pi * r2 / epsilon)) / 2.0, 0.0
            )
            delta3 = conditional(
                lt(r3, epsilon), max_value * (1.0 + cos(pi * r3 / epsilon)) / 2.0, 0.0
            )
            delta4 = conditional(
                lt(r4, epsilon), max_value * (1.0 + cos(pi * r4 / epsilon)) / 2.0, 0.0
            )
            return delta1 + delta2 + delta3 + delta4

    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)

        W = self.A(phi) * dot(grad(u), grad(v)) * self.dx
        W -= self.f * v * self.dx

        return [(W, self.bc)]

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):

        u = U[0]
        J = self.f * u * self.dx

        return J

    def constraint(self, phi, U):

        C = (1.0 / self.vol) * self.chi(phi) * self.dx

        return [C]

    def derivative(self, phi, U, P):

        u = U[0]

        if self.sc_type == "Uniform":
            S0_J = self.zero_vec
        elif self.sc_type == "1Load" or self.sc_type == "4Loads":
            S0_J = 2.0 * u * grad(self.f)

        S1_J = (2.0 * u * self.f - self.A(phi) * dot(grad(u), grad(u))) * self.Id
        S1_J += 2.0 * self.A(phi) * outer(grad(u), grad(u))

        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        S0 = (S0_J, [S0_C])
        S1 = (S1_J, [S1_C])

        return S0, S1

    def bilinear_form(self, th, xi):

        B = inner(grad(th), grad(xi)) * self.dx
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        bc = True

        if self.sc_type == "1Load":
            nv = FacetNormal(self.domain)
            B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
            bc = False

        return B, bc


class HeatPlus(Model):

    def __init__(self, dim, domain, space, dir_bcs, vol, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.N = len(dir_bcs)
        self.dx = Measure("dx", domain=domain)
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []
        self.wt = self.N * [1.0]  # weights

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.f = self.N * [1.0]  # sources

    @staticmethod
    def A(w):
        return conditional(lt(w, 0.0), 1.0, 1e-3)

    @staticmethod
    def chi(w):
        return conditional(lt(w, 0.0), 1.0, 0.0)

    def source(self, x0, y0, max_value=50.0, epsilon=0.1):
        x = SpatialCoordinate(self.domain)
        r = sqrt((x[0] - x0) ** 2 + (x[1] - y0) ** 2)
        scfun = conditional(
            lt(r, epsilon), max_value * (1.0 + cos(pi * r / epsilon)) / 2.0, 0.0
        )
        return scfun

    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)

        P = []
        for f, bc in zip(self.f, self.bc):
            P.append(
                [self.A(phi) * dot(grad(u), grad(v)) * self.dx - f * v * self.dx, bc]
            )

        return P

    def adjoint(self, phi, U):
        return []

    def cost(self, phi, U):

        J = []
        for wt, u, f in zip(self.wt, U, self.f):
            J.append(wt * f * u * self.dx)

        return sum(J)

    def constraint(self, phi, U):

        C = (1.0 / self.vol) * self.chi(phi) * self.dx

        return [C]

    def derivative(self, phi, U, P):

        S0_J = []
        for wt, u, f in zip(self.wt, U, self.f):
            if isinstance(f, (int, float)):
                S0_J.append(self.zero_vec)
            else:
                S0_J.append(wt * 2.0 * u * grad(f))

        S1_J = []
        for wt, u, f in zip(self.wt, U, self.f):
            s1 = (wt * 2.0 * u * f - self.A(phi) * dot(grad(u), grad(u))) * self.Id
            s1 += 2.0 * self.A(phi) * outer(grad(u), grad(u))
            S1_J.append(s1)

        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        S0 = [sum(S0_J), [S0_C]]
        S1 = [sum(S1_J), [S1_C]]

        return S0, S1

    def bilinear_form(self, th, xi):

        B = inner(grad(th), grad(xi)) * self.dx
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, True


class Logistic(Model):

    def __init__(self, dim, domain, space, vol, r, u0, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)

        self.r = r
        self.vol = vol
        self.u0 = u0

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)

        self.chi = lambda w: (conditional(lt(w, 0.0), 1.0, 0.0))
        self.K = lambda w: (conditional(lt(w, 0.0), 1.0, 1e-2))

    def L(self, phi, u):
        return self.r * (1.0 - u / self.K(phi)) * u

    def DL(self, phi, u, du):
        return self.r * (1.0 - 2.0 * u / self.K(phi)) * du

    def pde(self, phi):

        u = Coefficient(self.space)
        v = TestFunction(self.space)
        du = TrialFunction(self.space)

        F = dot(grad(u), grad(v)) * self.dx
        F -= self.L(phi, u) * v * self.dx

        J = dot(grad(du), grad(v)) * self.dx
        J -= self.DL(phi, u, du) * v * self.dx

        return [(F, [], J, u, self.u0)]

    def adjoint(self, phi, U):

        u = U[0]

        p = TrialFunction(self.space)
        q = TestFunction(self.space)

        W = dot(grad(p), grad(q)) * self.dx
        W -= self.DL(phi, u, q) * p * self.dx
        W -= q * self.dx

        return [(W, [])]

    def cost(self, phi, U):
        u = U[0]
        J = -u * self.dx
        return J

    def constraint(self, phi, U):

        C = (1.0 / self.vol) * self.chi(phi) * self.dx

        return [C]

    def derivative(self, phi, U, P):

        u = U[0]
        p = P[0]

        S0_J = self.zero_vec
        S1_J = (dot(grad(u), grad(p)) - u) * self.Id
        S1_J -= self.L(phi, u) * p * self.Id
        S1_J -= outer(grad(u), grad(p)) + outer(grad(p), grad(u))

        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        S0 = (S0_J, [S0_C])
        S1 = (S1_J, [S1_C])

        return S0, S1

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)
        B = inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds

        return B, False


class Mechanism(Model):
    """
    Models a compliant mechanism
    problem for linear elasticity.
    (in progress - Antoine - need to finish this)
    """

    def __init__(self, dim, domain, space, g, ds_g, dir_bcs, vol, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)
        self.g = as_vector(g)
        self.ds_g = ds_g
        self.bc = dir_bcs
        self.vol = vol
        self.sub = []

        E, nu = 20.0, 0.3
        lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        mu = E / 2.0 / (1.0 + nu)
        ks = 0.01  # Robin condition parameter
        eta_in = 2.0
        eta_out = 1.0

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda w: sym(grad(w))
        self.sigma = lambda w: (
            lmbda * nabla_div(w) * self.Id + 2.0 * mu * self.epsilon(w)
        )
        self.A = lambda w: (conditional(lt(w, 0.0), 1.0, 1e-2))
        self.chi = lambda w: (conditional(lt(w, 0.0), 1.0, 0.0))
        self.robin = lambda w: (ks * w)

    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        su = self.sigma(u)
        ev = self.epsilon(v)

        W = self.A(phi) * inner(su, ev) * self.dx
        W -= dot(self.g, v) * self.ds_g[1]  # Neumann condition on input boundary
        W += dot(self.robin(u), v) * self.ds_g[0]  # Robin condition on output boundary

        return [(W, self.bc)]

    def adjoint(self, phi, U):

        p = TrialFunction(self.space)
        r = TestFunction(self.space)
        sp = self.sigma(p)
        er = self.epsilon(r)

        W = self.A(phi) * inner(sp, er) * self.dx
        W += (
            2.0 * r[0] * self.ds_g[1] + r[0] * self.ds_g[0]
        )  # rhs for adjoint, on both input and output
        W += dot(self.robin(p), r) * self.ds_g[0]  # Robin condition on output boundary

        return [(W, self.bc)]

    def cost(self, phi, U):

        u = U[0]

        J = 2.0 * u[0] * self.ds_g[1] + u[0] * self.ds_g[0]

        return J

    def constraint(self, phi, U):

        C = (1.0 / self.vol) * self.chi(phi) * self.dx

        return [C]

    def derivative(self, phi, U, P):

        u = U[0]
        p = P[0]
        su = self.sigma(u)
        eu = self.epsilon(u)
        sp = self.sigma(p)
        ep = self.epsilon(p)

        S0_J = self.zero_vec
        S1_J = -grad(u).T * sp - grad(p).T * su
        S1_J += inner(su, ep) * self.Id
        S1_J *= self.A(phi)

        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        S0 = (S0_J, [S0_C])
        S1 = (S1_J, [S1_C])

        return S0, S1

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)

        B = 0.1 * dot(th, xi) * self.dx
        B += inner(grad(th), grad(xi)) * self.dx
        B += 1e5 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        # Also need to modify a bit the bilinear form, so that the input and output regions do not move ... see the paper

        return B, False


class Gripping(Model):

    def __init__(self, dim, domain, space, g, ds_g, k, dir_bcs, bc_theta, vol, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)
        self.g = [as_vector(gi) for gi in g]
        self.ds_g = ds_g
        self.k = [as_vector(ki) for ki in k]
        self.bc = dir_bcs
        self.bc_theta = bc_theta
        self.vol = vol
        self.sub = []

        E, nu = 200.0, 0.3
        lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        mu = E / 2.0 / (1.0 + nu)

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)
        self.epsilon = lambda w: sym(grad(w))
        self.sigma = lambda w: (
            lmbda * nabla_div(w) * self.Id + 2.0 * mu * self.epsilon(w)
        )
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)
        self.A = lambda w: conditional(lt(w, 0.0), 1.0, 1e-3)

    def pde(self, phi):

        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        su = self.sigma(u)
        ev = self.epsilon(v)

        W = self.A(phi) * inner(su, ev) * self.dx
        for gi, dgi in zip(self.g, self.ds_g):
            W -= dot(gi, v) * dgi

        return [(W, self.bc)]

    def adjoint(self, phi, U):

        p = TrialFunction(self.space)
        r = TestFunction(self.space)
        sp = self.sigma(p)
        er = self.epsilon(r)
        su = self.sigma(U[0])

        W = self.A(phi) * inner(sp, er) * self.dx
        for ki, dgi in zip(self.k, self.ds_g):
            W += dot(ki, r) * dgi

        return [(W, self.bc)]

    def cost(self, phi, U):

        u = U[0]
        J = sum([dot(ki, u) * dgi for ki, dgi in zip(self.k, self.ds_g)])

        return J

    def constraint(self, phi, U):

        C = (1.0 / self.vol) * self.chi(phi) * self.dx

        return [C]

    def derivative(self, phi, U, P):

        u = U[0]
        p = P[0]
        su = self.sigma(u)
        eu = self.epsilon(u)
        sp = self.sigma(p)
        ep = self.epsilon(p)

        S0_J = self.zero_vec
        S1_J = self.A(phi) * (
            -grad(u).T * sp
            - grad(p).T * su
            + 0.5 * inner(su, ep) * self.Id
            + 0.5 * inner(eu, sp) * self.Id
        )

        S0_C = self.zero_vec
        S1_C = (1.0 / self.vol) * self.chi(phi) * self.Id

        S0 = (S0_J, [S0_C])
        S1 = (S1_J, [S1_C])

        return S0, S1

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)

        B = 0.1 * dot(th, xi) * self.dx
        B = inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds

        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, self.bc_theta


class SVK(Model):
    """
    Saint Venant-Kirchhoff nonlinear elasticity model

    Run `test_37` for cantilever example.
    """

    def __init__(self, dim, domain, space, g, ds_g, dir_bcs, alpha, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)
        self.g = as_vector(g)
        self.ds_g = ds_g
        self.bc = dir_bcs
        self.alpha = alpha
        self.u0 = lambda x: 0.0 * x[:dim]

        E, nu = 200.0, 0.3
        lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        mu = E / 2.0 / (1.0 + nu)

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)

        self.E = lambda M: 0.5 * (M.T * M - self.Id)
        self.S = lambda M: lmbda * inner(M, self.Id) * self.Id + 2.0 * mu * M

        self.A = lambda w: conditional(lt(w, 0.0), 1.0, 1e-3)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)

    def pde(self, phi):

        u = Coefficient(self.space)
        v = TestFunction(self.space)
        du = TrialFunction(self.space)
        l = Constant(self.domain)

        F = self.Id + grad(u)
        S = self.S(self.E(F))

        Eq = self.A(phi) * inner(F * S, grad(v)) * self.dx
        Eq -= l * dot(self.g, v) * self.ds_g

        arg = grad(du) * S + F * self.S(sym(F.T * grad(du)))
        Jac = self.A(phi) * inner(arg, grad(v)) * self.dx

        return [(Eq, self.bc, Jac, u, self.u0, l, (0.1, 4))]

    def adjoint(self, phi, U):

        u = U[0]
        p = TrialFunction(self.space)
        q = TestFunction(self.space)

        F = self.Id + grad(u)
        S = self.S(self.E(F))

        arg = grad(p) * S + F * self.S(sym(F.T * grad(p))) + 2.0 * F * S
        W = self.A(phi) * inner(arg, grad(q)) * self.dx

        return [(W, self.bc)]

    def cost(self, phi, U):

        u = U[0]

        F = self.Id + grad(u)
        E = self.E(F)
        S = self.S(E)

        J = self.A(phi) * inner(S, E) * self.dx
        J += self.alpha * self.chi(phi) * self.dx

        return J

    def constraint(self, phi, U):

        return []

    def derivative(self, phi, U, P):

        u = U[0]
        p = P[0]

        F = self.Id + grad(u)
        E = self.E(F)
        S = self.S(E)

        S1 = inner(S, E) * self.Id - 2.0 * grad(u).T * F * S
        S1 += inner(F * S, grad(p)) * self.Id
        S1 -= grad(u).T * grad(p) * S
        S1 -= grad(p).T * F * S
        S1 -= grad(u).T * F * self.S(sym(F.T * grad(p)))

        S0_J = self.zero_vec
        S1_J = self.A(phi) * S1 + self.alpha * self.chi(phi) * self.Id

        return (S0_J, []), (S1_J, [])

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)

        B = dot(th, xi) * self.dx
        B += 0.1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds

        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False


class NHK(Model):

    def __init__(self, dim, domain, space, g, ds_g, dir_bcs, alpha, path):

        self.dim = dim
        self.domain = domain
        self.space = space
        self.path = path

        self.dx = Measure("dx", domain=domain)
        self.ds = Measure("ds", domain=domain)
        self.g = as_vector(g)
        self.ds_g = ds_g
        self.bc = dir_bcs
        self.alpha = alpha
        self.ini_func = None
        self.sub = []
        self.u0 = lambda x: 0.0 * x[:dim]

        # Material parameters
        E, nu = 200.0, 0.3
        self.lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
        self.mu = E / 2.0 / (1.0 + nu)

        self.zero_vec = as_vector(dim * [0.0])
        self.Id = Identity(dim)

        # SVK strain and stress, other kinematic helpers
        self.C = lambda M: M.T * M
        self.J = lambda M: det(M)

        self.A = lambda w: conditional(lt(w, 0.0), 1.0, 1e-2)
        self.chi = lambda w: conditional(lt(w, 0.0), 1.0, 0.0)

    def S(self, F):

        J = det(F)
        a = self.lmbda * 0.5 * (J**2 - 1) - self.mu

        S = a * inv(self.C(F))
        S += self.mu * self.Id

        return S

    def dS(self, F, dF):

        J = det(F)
        a = self.lmbda * 0.5 * (J**2 - 1) - self.mu

        dS = a * -1.0 * inv(self.C(F)) * (F.T * dF + dF.T * F) * inv(self.C(F))
        dS += self.lmbda * J * J * tr(inv(F) * dF) * inv(self.C(F))

        return dS

    def E(self, F):

        E = 0.5 * (F.T * F - self.Id)

        return E

    def dE(self, F, dF):

        dE = 0.5 * (F.T * dF + dF.T * F)

        return dE

    def dW(self, F, dF):

        return inner(F * self.S(F), dF)

    def d2W(self, F, dF1, dF2):

        return inner((dF2 * self.S(F) + F * self.dS(F, dF2)), dF1)

    def pde(self, phi):

        u = Coefficient(self.space)
        v = TestFunction(self.space)
        du = TrialFunction(self.space)
        l = Constant(self.domain)

        F = self.Id + grad(u)
        S = self.S(F)

        Eq = self.A(phi) * inner(F * S, grad(v)) * self.dx
        Eq -= l * dot(self.g, v) * self.ds_g

        arg = grad(du) * S + F * self.dS(F, grad(du))
        Jac = self.A(phi) * inner(arg, grad(v)) * self.dx

        return [(Eq, self.bc, Jac, u, self.u0, l, (0.1, 4))]

    def adjoint(self, phi, U):

        u = U[0]
        p = TrialFunction(self.space)
        q = TestFunction(self.space)

        F = self.Id + grad(u)
        J = det(F)
        E = self.E(F)
        S = self.S(F)
        Cinv = inv(self.C(F))
        Finv = inv(F)

        # Some kinematic helpers
        a = self.mu - self.lmbda * 0.5 * (J**2 - 1)
        M = Cinv * E * Cinv

        # Variational
        arg = grad(p) * S + F * self.dS(F, grad(p))

        # Cost functional
        arg += F * S
        arg += 2.0 * a * F * M
        arg += self.lmbda * J**2 * inner(Cinv, E) * Finv.T

        W = self.A(phi) * inner(arg, grad(q)) * self.dx

        return [(W, self.bc)]

    def cost(self, phi, U):

        u = U[0]

        F = self.Id + grad(u)
        E = self.E(F)
        S = self.S(F)

        J = self.A(phi) * inner(S, E) * self.dx
        J += self.alpha * self.chi(phi) * self.dx

        return J

    def constraint(self, phi, U):

        return []

    def derivative(self, phi, U, P):

        u = U[0]
        p = P[0]
        F = self.Id + grad(u)
        J = det(F)
        E = self.E(F)
        S = self.S(F)
        Cinv = inv(self.C(F))
        Finv = inv(F)

        # Some kinematic helpers
        a = self.mu - self.lmbda * 0.5 * (J**2 - 1)
        M = Cinv * E * Cinv

        # Cost functional
        S1 = inner(S, E) * self.Id - 1.0 * grad(u).T * F * S
        S1 -= grad(u).T * 2.0 * a * F * M
        S1 -= grad(u).T * self.lmbda * J**2 * inner(Cinv, E) * Finv.T

        # Variational
        S1 += inner(F * S, grad(p)) * self.Id
        S1 -= grad(p).T * F * S
        S1 -= grad(u).T * grad(p) * S
        S1 -= grad(u).T * F * self.dS(F, grad(p))

        S0_J = self.zero_vec
        S1_J = self.A(phi) * S1 + self.alpha * self.chi(phi) * self.Id

        return (S0_J, []), (S1_J, [])

    def bilinear_form(self, th, xi):

        nv = FacetNormal(self.domain)
        B = dot(th, xi) * self.dx
        B += 0.1 * inner(grad(th), grad(xi)) * self.dx
        B += 1e4 * dot(th, nv) * dot(xi, nv) * self.ds
        for sb in self.sub:
            B += 1e4 * sb * dot(th, xi) * self.dx

        return B, False
