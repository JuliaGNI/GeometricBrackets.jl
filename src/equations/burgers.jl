
@doc raw"""
    burgers_bracket(space)

The discrete Burgers bracket,

```math
\mathbb{J}_{ij} (u) = \sqrt{u_i} \; \mathbb{K}_{ij} \; \sqrt{u_j} ,
\qquad
\mathbb{K} = \mathbb{M}^{-1} \mathbb{D} \mathbb{M}^{-1} ,
\qquad
\mathbb{D}_{kl} = \int_\Omega
    \left( \phi_k \phi_l' - \phi_l \phi_k' \right) dx ,
```

a [`GaugedBracket`](@ref), hence Poisson by construction on ``u > 0``.

# Where it comes from

The continuous bracket

```math
\{A, B\} = \int_\Omega u \left(
    \frac{\delta A}{\delta u} \partial_x \frac{\delta B}{\delta u}
  - \frac{\delta B}{\delta u} \partial_x \frac{\delta A}{\delta u} \right) dx
```

gives ``u_t = 3 u u_x`` with ``H = \tfrac{1}{2}\int u^2``. Under ``\bar{u} = \sqrt{u}`` it
becomes *independent of the dynamical variables*,

```math
\{A, B\} = \tfrac{1}{4} \int_\Omega \left(
    \frac{\delta A}{\delta \bar{u}} \partial_x \frac{\delta B}{\delta \bar{u}}
  - \frac{\delta B}{\delta \bar{u}} \partial_x \frac{\delta A}{\delta \bar{u}} \right) dx ,
```

the term that would spoil it being symmetric in ``A`` and ``B`` and so cancelling under
antisymmetrisation. The discretisation is carried out *there*, where the bracket is
constant and its discretisation is automatically Poisson, and the transformation is then
undone. This cancellation is a particular feature of the one-dimensional transport bracket
and does not happen for general Lie-Poisson brackets.

# Why a nodal basis

Reversing the transformation replaces the coefficients ``\bar{u}_i`` by ``\sqrt{u_i}``,
which is only justified when the coefficients *are* nodal values — hence
[`LagrangeSpace`](@ref) rather than a spline space. In practice the construction is used
more widely than that argument licenses; what it does genuinely require is ``u > 0``, so
that the square roots exist.

# The even-rank trap

``\mathbb{K}`` is antisymmetric, and an antisymmetric matrix has **even** rank. For an odd
number of degrees of freedom its corank is one, and the single kernel vector
``\mathbb{M}\mathbf{1} = \int \phi_i`` gives the physical Casimir. For an **even** number
there is one further kernel vector — the sawtooth mode ``(-1)^i`` — and hence a *spurious*
discrete Casimir with no continuum counterpart. Choose ``N = p \, n_e`` odd.
"""
function burgers_bracket(s::DiscreteSpace)
    Minv = inverse_mass_matrix(s)
    S = derivative_matrix(s)
    K = Minv * (S - S') * Minv
    GaugedBracket(K, sqrt, u -> 1 / (2 * sqrt(u)))
end

@doc raw"""
    BurgersHamiltonian(space)

``H = \tfrac{1}{2} \int_\Omega u^2 \, dx``, discretised as
``\tfrac{1}{2} \hat{u}^T \mathbb{M} \hat{u}``.

Contracted with [`burgers_bracket`](@ref) it reproduces the nodal values of ``3 u u_x`` to
second order.
"""
BurgersHamiltonian(s::DiscreteSpace) = QuadraticHamiltonian(mass_matrix(s))

@doc raw"""
    burgers_casimir(space)

The exact discrete Casimir of [`burgers_bracket`](@ref),

```math
C(u) = 2 \sum_i \left( \int_\Omega \phi_i \, dx \right) \sqrt{u_i} ,
```

with gradient ``\partial C / \partial u_i = \left(\int \phi_i\right) / \sqrt{u_i}``.

Exact, not approximate: ``\mathbb{J} \, \partial C/\partial u = 0`` to round-off, because
``\int \phi_i`` spans the kernel of ``\mathbb{K}`` and the two gauge factors cancel. It is
at the same time a consistent quadrature of the continuous Casimir
``2 \int_\Omega \sqrt{u} \, dx``.
"""
burgers_casimir(s::DiscreteSpace{T}) where {T} = GaugedCasimir{
    T, typeof(_2sqrt), typeof(_invsqrt)}(basis_integrals(s), _2sqrt, _invsqrt)

_2sqrt(u) = 2 * sqrt(u)
_invsqrt(u) = 1 / sqrt(u)

@doc raw"""
    to_sqrt_variables(û)
    from_sqrt_variables(ū)

The transformation ``\bar{u}_i = \sqrt{u_i}`` and its inverse ``u_i = \bar{u}_i^2``.

In ``\bar{u}`` the Burgers bracket is the **constant** tensor ``\mathbb{K}/4``, so the
system there is a Poisson system with a constant structure matrix and every symplectic
Runge-Kutta method is a Poisson integrator for it. The Casimir is *linear* in these
variables, ``C = 2 \left(\int \phi_i\right) \cdot \bar{u}``, and is therefore conserved
exactly by any such method, at any step size.

Integrating in ``\bar{u}`` and transforming back is the practical form of the construction.

!!! note "The factor and the transformation move together"
    ``\bar{u} = \sqrt{u}`` goes with ``\mathbb{K}/4`` and ``\partial H / \partial
    \bar{u}_i = 2 \sqrt{u_i} \, (\mathbb{M} u)_i``; the other common choice,
    ``\bar{u} = 2\sqrt{u}``, goes with ``\mathbb{K}`` and no factor. Pairing
    ``2\sqrt{u}`` with ``\mathbb{K}/4`` makes the flow four times too slow, which
    no conservation test can see: a constant rescaling of a Poisson vector field preserves
    every Casimir, every energy and every convergence *rate*. The convention here is the
    manuscript's, eq. (variable-transformation).
"""
to_sqrt_variables(û::AbstractVector) = sqrt.(û)
from_sqrt_variables(ū::AbstractVector) = ū .^ 2

@doc raw"""
    BurgersSystem(space)
    BurgersSystem(p, ne; L = 2π)

The Burgers system: the gauged bracket, the quadratic Hamiltonian, the exact Casimir, and
the flow they generate.

```jldoctest
julia> sys = BurgersSystem(2, 13);            # N = 26 is even -- see the note below

julia> nbasis(sys.space)
26
```

!!! note "Use an odd number of degrees of freedom"
    ``N = p \, n_e`` should be odd, or the antisymmetric ``\mathbb{K}`` has corank two and
    the extra sawtooth kernel vector gives a spurious Casimir. See
    [`burgers_bracket`](@ref).
"""
struct BurgersSystem{T, ST <: DiscreteSpace{T}, BT, HT, CT, FT}
    space::ST
    bracket::BT
    H::HT
    C::CT
    flow::FT
end

function BurgersSystem(s::DiscreteSpace{T}) where {T}
    b = burgers_bracket(s)
    h = BurgersHamiltonian(s)
    c = burgers_casimir(s)
    f = HamiltonianFlow(s, b, h)
    BurgersSystem{T, typeof(s), typeof(b), typeof(h), typeof(c), typeof(f)}(s, b, h, c, f)
end

function BurgersSystem(p::Integer, ne::Integer; L = 2π, kwargs...)
    BurgersSystem(LagrangeSpace(p, ne; L = L, kwargs...))
end

Base.eltype(::BurgersSystem{T}) where {T} = T
nbasis(sys::BurgersSystem) = nbasis(sys.space)
