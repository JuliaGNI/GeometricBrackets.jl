```@meta
CurrentModule = GeometricBrackets
```

# Burgers

The transport bracket

```math
\{A,B\} = \int_\Omega u \left(
    \frac{\delta A}{\delta u} \partial_x \frac{\delta B}{\delta u}
  - \frac{\delta B}{\delta u} \partial_x \frac{\delta A}{\delta u} \right) dx ,
\qquad H = \tfrac{1}{2}\int_\Omega u^2 \, dx
```

gives ``u_t = 3uu_x``.

## Making the bracket constant

Under ``\bar{u} = \sqrt{u}`` the bracket becomes *independent of the dynamical variables*,

```math
\{A, B\} = \tfrac{1}{4} \int_\Omega \left(
    \frac{\delta A}{\delta \bar{u}} \partial_x \frac{\delta B}{\delta \bar{u}}
  - \frac{\delta B}{\delta \bar{u}} \partial_x \frac{\delta A}{\delta \bar{u}} \right) dx ,
```

the term that would spoil it being symmetric in ``A`` and ``B`` and so cancelling under
antisymmetrisation. Discretising **there**, where the bracket is constant and therefore
automatically Poisson, and undoing the transformation afterwards, gives

```math
\mathbb{J}_{ij}(u) = \sqrt{u_i} \; \mathbb{K}_{ij} \; \sqrt{u_j} ,
\qquad \mathbb{K} = \mathbb{M}^{-1}\mathbb{D}\mathbb{M}^{-1} ,
```

an instance of [`GaugedBracket`](@ref) and hence Poisson by construction on ``u > 0``.

This cancellation is a particular feature of the one-dimensional transport bracket and does
**not** happen for general Lie-Poisson brackets.

## The theorem behind it

``\mathbb{J}_{ij} = g_i(u_i) \, \mathbb{K}_{ij} \, g_j(u_j)`` is a Poisson bracket for any
**constant** antisymmetric ``\mathbb{K}`` and any smooth ``g_i`` of the **single** variable
``u_i``. Both hypotheses are sharp, and the test suite exhibits both failures rather than
merely asserting the theorem.

Its Casimirs are ``C_n = \sum_i n_i G_i(u_i)`` with ``G' = 1/g`` and ``n \in \ker\mathbb{K}``.
For the square root, ``G(u) = 2\sqrt{u}``, giving the *exact* discrete Casimir

```math
C(u) = 2 \sum_i \left( \int_\Omega \phi_i \, dx \right) \sqrt{u_i} .
```

## Why a nodal basis

Reversing the transformation replaces the coefficients ``\bar{u}_i`` by ``\sqrt{u_i}``, which
is only justified when the coefficients *are* nodal values — hence [`LagrangeSpace`](@ref)
rather than a spline space.

## The even-rank trap

``\mathbb{K}`` is antisymmetric, and an antisymmetric matrix has **even** rank. For an odd
number of degrees of freedom its corank is one and the single kernel vector
``\mathbb{M}\mathbf{1} = \int\phi_i`` gives the physical Casimir. For an **even** number
there is one further kernel vector — the sawtooth mode ``(-1)^i`` — and hence a *spurious*
discrete Casimir with no continuum counterpart. Choose ``N = p\,n_e`` **odd**.

## Initial conditions

A smooth, strictly positive periodic profile is required, since the bracket takes square
roots. The standard one is

```math
u_0 (x) = 2 + \sin x + 0.3 \cos 2x ,
```

on ``[0, 2\pi)`` with ``P_1`` elements on 25 cells or ``P_2`` on 13 — both giving an odd
``N``.

```@example burgers
using GeometricBrackets
sys = BurgersSystem(2, 13)                     # N = 26 ... even, see the trap above
u₀ = [2 + sin(x) + 0.3cos(2x) for x in nodes(sys.space)]
traj = integrate(sys, Integrator(sys.flow, ImplicitMidpoint(), 1e-3), u₀, 200; stride = 20)
(H = drift(traj, :H), C = drift(traj, :C))
```

Integrating in ``\bar{u} = \sqrt{u}``, where the bracket is the constant ``\mathbb{K}/4``,
makes the Casimir *linear* and therefore exactly conserved by any symplectic method at any
step size. See [`to_sqrt_variables`](@ref).

## Reference

```@autodocs
Modules = [GeometricBrackets]
Pages = ["equations/burgers.jl"]
```
