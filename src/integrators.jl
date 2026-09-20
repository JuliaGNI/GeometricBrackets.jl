
@doc raw"""
    IntegratorMethod

A one-step method for an [`AbstractFlow`](@ref).

The methods here are chosen to expose a trade-off rather than to cover the space of
integrators. Three properties are in play — whether the map is Poisson, whether it conserves
the generating Hamiltonian exactly, and whether it conserves the mass — and no method has
all three:

| method | Poisson map | generating `H` | mass |
|:--|:--|:--|:--|
| [`ImplicitMidpoint`](@ref) | yes, for a constant bracket | only if `H` is quadratic | yes |
| [`AverageVectorField`](@ref) | no | yes, for a constant bracket | yes |
| [`Gonzalez`](@ref), [`GonzalezMass`](@ref) | no | yes, for any `H` | yes |
| [`ExplicitEuler`](@ref), [`RungeKutta4`](@ref) | no | no | yes |
| [`ProjectionMethod`](@ref) | no | yes, by construction | only if asked for |

That no method here is both a Poisson map and exactly energy-preserving is the Ge-Marsden
theorem, not a gap in the list: a Poisson integrator that also conserved the Hamiltonian
exactly would reproduce the exact flow up to a reparametrisation of time. The theorem carries
non-degeneracy hypotheses on the system that are not checked here, so read the statement as
the reason to expect the trade-off rather than as a proof that no such method can exist for
these particular brackets — one of which is not even Poisson.

The mass, by contrast, comes free to *every* method here, explicit Euler included, because
its gradient spans the kernel of the first bracket and every increment above lies in that
bracket's range. See [`MassCasimir`](@ref).
"""
abstract type IntegratorMethod end

"""
    isexplicit(method)

Whether the method requires no nonlinear solve.
"""
isexplicit(::IntegratorMethod) = false

"""
    ExplicitEuler()

The explicit Euler method, ``\\hat{u}^{n+1} = \\hat{u}^n + \\Delta t \\, f(\\hat{u}^n)``.

Present as a negative control. It is not a Poisson map, does not conserve energy and is not
even stable here — but it *does* conserve the mass exactly, which is the point: that
conservation is a property of the bracket's kernel and not of the integrator at all.
"""
struct ExplicitEuler <: IntegratorMethod end
isexplicit(::ExplicitEuler) = true

@doc raw"""
    RungeKutta4()

The classical explicit fourth-order Runge-Kutta method — the non-geometric reference.

Neither a Poisson map nor energy-preserving, and only conditionally stable. Both linearised
fields here have essentially imaginary spectra, so the relevant limit is the imaginary-axis
one, ``|\lambda| \Delta t \le 2\sqrt{2}``, and the spectral radius grows like ``h^{-3}``
through the third derivative; see [`stability_limit`](@ref). Inside that limit the method is
very slightly dissipative, ``|R(iy)| < 1``, so the invariants do not merely wander, they
decay.
"""
struct RungeKutta4 <: IntegratorMethod end
isexplicit(::RungeKutta4) = true

@doc raw"""
    ImplicitMidpoint()

The implicit midpoint rule,
``\hat{u}^{n+1} = \hat{u}^n + \Delta t \, f\!\left(\tfrac{1}{2}(\hat{u}^n +
\hat{u}^{n+1})\right)``.

A **Poisson integrator** whenever the structure matrix is constant. Differentiating the step
implicitly gives the Cayley transform

```math
D\Phi_{\Delta t} = \left( \mathbb{I} - \tfrac{1}{2} \Delta t \, \mathbb{P} H'' \right)^{-1}
                   \left( \mathbb{I} + \tfrac{1}{2} \Delta t \, \mathbb{P} H'' \right) ,
```

and a Cayley transform of ``\mathbb{P}`` times a symmetric matrix preserves ``\mathbb{P}``
exactly, so ``D\Phi \, \mathbb{P} \, D\Phi^T = \mathbb{P}`` to round-off. See
[`tangent_map`](@ref).

It conserves *quadratic* invariants of the differential equation, being symplectic — so it
holds the quadratic ``H_2`` exactly along the second KdV flow. It does not hold the cubic
``H_1`` along the first.
"""
struct ImplicitMidpoint <: IntegratorMethod end

@doc raw"""
    AverageVectorField()

The average vector field method with a two-point Gauß rule along the segment,

```math
\hat{u}^{n+1} = \hat{u}^n + \Delta t \int_0^1
    f\!\left( \hat{u}^n + \tau \, \Delta \hat{u} \right) d\tau ,
```

exact here because both vector fields are quadratic in ``\hat{u}``.

For a **constant** structure matrix, averaging the vector field and averaging only the
gradient are the same thing, the averaged gradient is a discrete gradient, and
``\bar{g}^T \mathbb{P} \bar{g} = 0`` by antisymmetry — so the method conserves the
generating Hamiltonian exactly, cubic or not.

!!! warning "Not energy-preserving for a variable structure matrix"
    Once ``\mathbb{P}`` depends on ``\hat{u}`` the two readings part company and the
    identity fails: what the method contracts the averaged gradient against is
    ``\int_0^1 \mathbb{P}(\xi) \, \partial H/\partial \hat{u} (\xi) \, d\tau``, and the
    average of a product is not the product of the averages. Along the second KdV flow it
    loses ``H_2``. The remedy is to keep the discrete-gradient form instead —
    [`Gonzalez`](@ref) — which evaluates ``\mathbb{P}`` at a single point.
"""
struct AverageVectorField <: IntegratorMethod end

# The two-point Gauß-Legendre nodes on [0,1], as fractions of the segment.
const AVF_NODES = (0.5 - sqrt(3.0) / 6.0, 0.5 + sqrt(3.0) / 6.0)

@doc raw"""
    Gonzalez()
    GonzalezMass()

The midpoint discrete gradient method of Gonzalez,

```math
\hat{u}^{n+1} = \hat{u}^n + \Delta t \, \mathbb{P}(\bar{u}) \, \bar{g} ,
\qquad
\bar{g} = \frac{\partial H}{\partial \hat{u}} (\bar{u})
        + \frac{\Delta H - \left\langle \partial H/\partial \hat{u} (\bar{u}),
                \Delta \hat{u} \right\rangle}
               {\left\langle \Delta \hat{u}, c \right\rangle} \, c ,
```

with ``\bar{u} = \tfrac{1}{2}(\hat{u}^n + \hat{u}^{n+1})`` and
``\Delta \hat{u} = \hat{u}^{n+1} - \hat{u}^n``.

The rank-one correction is what makes ``\bar{g}`` a *discrete gradient*, i.e. what makes
``\bar{g} \cdot \Delta\hat{u} = \Delta H`` hold exactly; energy conservation then follows
from the antisymmetry of ``\mathbb{P}`` alone, for any Hamiltonian. For a **quadratic**
Hamiltonian the correction vanishes identically and the method reduces to the plain midpoint
rule — which is where the name comes from.

The two variants differ in the direction ``c`` of the correction:

  - `Gonzalez` takes ``c = \Delta\hat{u}``, the Euclidean choice;
  - `GonzalezMass` takes ``c = \mathbb{M} \Delta\hat{u}``, the mass metric.

The discrete-gradient property fixes only the component of the correction along
``\Delta\hat{u}``, so any ``c`` with ``\langle \Delta\hat{u}, c\rangle \ne 0`` will do. The
Euclidean choice is equivariant only under Euclidean-orthogonal changes of variables,
whereas the symmetry group of the problem is ``\mathbb{M}``-orthogonal — which is the
argument for the second variant. In practice the correction it changes is ``O(\Delta t^2)``
either way and the two agree to a few per cent; both are kept because the comparison is the
point.
"""
struct Gonzalez <: IntegratorMethod end

@doc (@doc Gonzalez)
struct GonzalezMass <: IntegratorMethod end

"""
    DiscreteGradient

Either discrete-gradient method, [`Gonzalez`](@ref) or [`GonzalezMass`](@ref).

The two differ only in the metric their rank-one correction is taken in, so every dispatch
that cares about the *property* — exact conservation of the generating Hamiltonian for any
`H`, at the cost of the Poisson-map property — is written against this union rather than
against either of them.
"""
const DiscreteGradient = Union{Gonzalez, GonzalezMass}

@doc raw"""
    ProjectionMethod(base, invariants)

A `base` method followed by a projection onto the level sets of `invariants`,

```math
\hat{u}^{n+1} = \tilde{u} + \sum_k \lambda_k \, \frac{\partial I_k}{\partial \hat{u}} ,
```

with the multipliers ``\lambda`` solving ``I_k(\hat{u}^{n+1}) = I_k(\hat{u}^0)`` for every
`k`.

This is how to hold *both* Hamiltonians of a bi-Hamiltonian pair at once, and it is worth
being clear about what it costs and what it means.

  - The correction direction is not mass-neutral, so projecting onto two Hamiltonians alone
    **loses** the Casimir. Pass the mass as a third invariant and all three are held at once.
  - The projected map is no longer Poisson.
  - More fundamentally, pinning the second Hamiltonian to its initial value enforces
    something the *semi-discrete* flow itself violates, at exactly the size the projection
    removes. Exact conservation of both is therefore available, but as a label attached to
    the trajectory rather than as a property inherited from the continuous bi-Hamiltonian
    structure.
"""
struct ProjectionMethod{MT <: IntegratorMethod, IT} <: IntegratorMethod
    base::MT
    invariants::IT
end

isexplicit(m::ProjectionMethod) = isexplicit(m.base)

@doc raw"""
    default_f_abstol(T, ndofs, û₀ = nothing)

The residual tolerance for the Newton iteration,
``4 \, \max(8, N) \, \varepsilon \, \lVert \hat{u}_0 \rVert_\infty``.

The tolerance is **absolute**, so it has to sit above the round-off floor of the residual —
and that floor scales with both the number of degrees of freedom, which is how many terms
accumulate error, and the *amplitude* of the field. Getting either wrong is expensive in a
different direction, and both were measured on implicit midpoint on the second KdV flow:

| tolerance | ``\lVert u \rVert \approx 4`` | ``\lVert u \rVert \approx 1`` |
|:--|:--|:--|
| ``\max(8,N)\varepsilon`` | 40 iterations, never converges | 40 iterations, never converges |
| **this** | **3.0 iterations** | **2.5 iterations** |
| ``16 \times`` this | 2.9 iterations, ``H_2`` error ``10\times`` worse | 2.0 iterations |

Below the floor Newton reaches its limit in three iterations and then spins to the iteration
cap making no progress, reporting non-convergence on a step it has in fact solved. Above it,
Newton stops early and the conservation laws pay: at ``10^{-11}`` the drift in ``H_2`` is
``170\times`` larger. Neither is a tolerable trade in a package whose subject is exact
conservation, and the window between them is not wide — hence a formula rather than a
constant.

Pass `û₀` to [`Integrator`](@ref) so the amplitude is known; without it the factor is one,
which is right for a field of order one and too tight for anything much larger.
"""
default_f_abstol(::Type{T}, ndofs::Integer, û₀ = nothing) where {T} = 4 * max(8, ndofs) *
                                                                      eps(T) *
                                                                      (û₀ === nothing ?
                                                                       one(T) :
                                                                       max(one(T),
                                                                          convert(T, maximum(abs, û₀))))

@doc raw"""
    Integrator(flow, method, Δt; kwargs...)

A one-step integrator for `flow`, with its nonlinear solver built **once** and reused across
every step.

```jldoctest
julia> sys = KdVSystem(SplineSpace(16, 3));

julia> integ = Integrator(sys.flow1, ImplicitMidpoint(), 1e-3);

julia> û = project(sys.space, sin); integrate_step!(û, integ); length(û)
16
```

The solver is a `SimpleSolvers.NewtonSolver` with an analytic Jacobian. Its `refactorize`
option is the quasi-Newton scheme these problems want: the Jacobian is factorised once and
reused, and refactorised only when the residual stops decreasing. That matters because an
assembly costs ``O(N^2 Q)``, and at the resolutions and step counts used here it is the
whole run time.

# Keyword arguments

  - `refactorize = 5`: Newton iterations between Jacobian factorisations.
  - `linesearch = Static(T)`: no line search, i.e. a full Newton step, which is what the
    problem wants at these step sizes and is measurably faster than the default
    `Backtracking`.
  - `fallback_linesearch = Backtracking(T)`: used to *redo* a step whose first attempt did
    not converge. See [`_step!`](@ref).
  - `linear_solver_method`: left to SimpleSolvers, which picks `LapackLU` for the dense
    formulation and `SparspakLU` for the mixed one. Pass it explicitly to override — e.g.
    `SimpleSolvers.LU()` for the portable scalar factorisation, which is the only choice for
    element types LAPACK does not provide, and which accounted for 74 % of an implicit step at
    `N = 384` before `LapackLU` replaced it.
  - `formulation = :dense`: see below.
  - `f_abstol = 1e-13`, `max_iterations = 40`: passed through to the solver's options.
  - any other `SimpleSolvers.Options` keyword.

# The mixed two-field formulation

`formulation = :mixed` solves the step as `2N` equations in `(y, v)` rather than `N` in `y`:

```
M (y - uⁿ) - Δt K(ū) v = 0        M v - ∂H/∂u(ū) = 0        ū = (uⁿ + y)/2
```

Eliminating `v` gives back the dense residual, so it is the same step — but it never forms
`Minv * K * Minv`, which is what makes the dense Newton matrix dense. All four blocks of the
mixed Jacobian are banded with circular bandwidth `p`, and the pattern is fixed for the whole
run, so one ordering and symbolic factorisation serve every step. See [`mixed_residual!`](@ref)
and [`kernel_operator`](@ref).

It is not free: `2N` unknowns against `N`, so the dense form wins while `N` is small.
Measured on implicit midpoint on the second KdV flow at `p = 3`, one step in milliseconds, with
both formulations converging in two Newton iterations and agreeing to 12–14 digits:

| N | dense | mixed | speedup |
|---:|---:|---:|---:|
| 64 | 0.192 | 0.259 | 0.74 |
| 128 | 0.518 | 0.485 | 1.07 |
| 384 | 3.34 | 1.56 | **2.1** |
| 1024 | 22.9 | 4.22 | **5.4** |
| 1536 | 58.2 | 6.29 | **9.3** |

So the crossover is near `N = 125`, and `:dense` stays the default.

!!! warning "Restrictions"
    Only [`ImplicitMidpoint`](@ref), and only an [`AffineBracket`](@ref) — the KdV and
    Camassa-Holm second brackets. [`AverageVectorField`](@ref) would need one auxiliary per
    quadrature node and the discrete-gradient methods carry a rank-one term in `∂ḡ/∂y` that
    would have to be bordered; the other brackets store the sandwiched `Minv*K*Minv` rather
    than `K`, so the weak-form block cannot be recovered. Each of these is an `ArgumentError`
    rather than a silently different method.

!!! warning "The linear solver is a correctness constraint here"
    The mixed system is block-structured, and UMFPACK mis-handles it: from `N = 768` upward
    `SimpleSolvers.UmfpackLU` returns a solution wrong by a factor of 150 while reporting
    success, which shows up as Newton diverging. `SparspakLU` and dense LAPACK both solve the
    same matrices to the accuracy their condition number allows, so `SparspakLU` is the default
    for this formulation. Do not override it without checking the residual.
"""
struct Integrator{T, FT <: AbstractFlow{T}, MT <: IntegratorMethod, ST, CT, BT, BCT}
    flow::FT
    method::MT
    Δt::T
    solver::ST
    state::CT
    # the same problem with a line search; a separate type parameter because the line
    # search is part of the solver's type
    fallback::BT
    fallbackstate::BCT
    cache::Vector{Vector{T}}
    targets::Vector{T}          # level-set values held by a ProjectionMethod
    formulation::Symbol         # :dense (the state alone) or :mixed (state + auxiliary)
end

function Integrator(f::AbstractFlow{T}, method::IntegratorMethod, Δt::Real;
        refactorize::Integer = 5,
        û₀ = nothing,
        f_abstol = default_f_abstol(T, nbasis(space(f)), û₀),
        min_iterations::Integer = 1,
        max_iterations::Integer = 40, verbosity::Integer = 0,
        linesearch = Static(T), fallback_linesearch = Backtracking(T),
        linear_solver_method = missing, formulation::Symbol = :dense,
        kwargs...) where {T}
    N = nbasis(space(f))
    dt = convert(T, Δt)
    formulation in (:dense, :mixed) || throw(ArgumentError(
        "formulation must be :dense or :mixed, but got :$(formulation)"))

    if isexplicit(method)
        # The explicit methods evaluate `vectorfield` and nothing else, so the formulation is
        # not a choice they have: `poisson_apply` is already matrix-free.
        return Integrator{T, typeof(f), typeof(method), Nothing, Nothing, Nothing, Nothing}(
            f, method, dt, nothing, nothing, nothing, nothing, [zeros(T, N)], T[], :dense)
    end

    # `params` carries the state at the beginning of the step; SimpleSolvers passes it
    # through to both callbacks, so the solver itself can be built once, outside the loop.
    if formulation === :mixed
        n = mixed_dimension(method, N)
        proto = mixed_jacobian_prototype(f, method)

        # The first residual block carries a factor of the mass matrix and the second does
        # not, so the equation is scaled rather than the tolerance -- see `mixed_row_scale`.
        # `f_abstol` therefore means the same thing here as in the dense formulation.
        σ = mixed_row_scale(space(f))

        # `SparspakLU` rather than the `UmfpackLU` that SimpleSolvers would otherwise pick for
        # a sparse Float64 Jacobian. This is a correctness requirement, not a preference:
        # UMFPACK's ordering and threshold pivoting mis-handle this particular *block*
        # structure, and from N = 768 upward it returns a solution wrong by a factor of 150
        # while reporting `issuccess`. Sparspak solves the same matrices to the accuracy their
        # condition number allows, and dense LAPACK agrees with it. See the notes on
        # `formulation` in the `Integrator` docstring.
        lsm = linear_solver_method === missing ? SparspakLU() : linear_solver_method

        Fm!(r, z, params) = mixed_residual!(r, method, f, params.un, z, dt, σ)
        Jm!(j, z, params) = mixed_jacobian!(j, method, f, params.un, z, dt, σ)

        newton_mixed(ls) = NewtonSolver(zeros(T, n), zeros(T, n);
            F = Fm!, (DF!) = Jm!, refactorize = refactorize,
            jacobian_prototype = copy(proto),
            linesearch = ls,
            linear_solver_method = lsm,
            f_abstol = f_abstol,
            min_iterations = min_iterations,
            max_iterations = max_iterations,
            verbosity = verbosity, kwargs...)
        solver = newton_mixed(linesearch)
        fallback = newton_mixed(fallback_linesearch)
        cache = [zeros(T, N), zeros(T, n)]
    else
        F!(y, x, params) = residual!(y, method, f, params.un, x, dt)
        J!(j, x, params) = residual_jacobian!(j, method, f, params.un, x, dt)

        newton_dense(ls) = NewtonSolver(zeros(T, N), zeros(T, N);
            F = F!, (DF!) = J!, refactorize = refactorize,
            linesearch = ls,
            linear_solver_method = linear_solver_method,
            f_abstol = f_abstol,
            min_iterations = min_iterations,
            max_iterations = max_iterations,
            verbosity = verbosity, kwargs...)
        solver = newton_dense(linesearch)
        fallback = newton_dense(fallback_linesearch)
        cache = [zeros(T, N)]
    end

    state = SolverState(solver)
    fbstate = SolverState(fallback)

    Integrator{T, typeof(f), typeof(method), typeof(solver), typeof(state),
        typeof(fallback), typeof(fbstate)}(
        f, method, dt, solver, state, fallback, fbstate, cache, T[], formulation)
end

Base.eltype(::Integrator{T}) where {T} = T
timestep(integ::Integrator) = integ.Δt

## Residuals of the implicit methods
#
# Each method contributes a residual r(y) = y - uⁿ - Δt Φ(uⁿ, y) and its exact Jacobian
# ∂r/∂y = I - Δt ∂Φ/∂y. Writing the Jacobians out rather than differencing them is what
# lets the Poisson-map test below measure the method instead of its own truncation error.

function residual!(r, ::ImplicitMidpoint, f::AbstractFlow, un, y, Δt)
    ū = (un .+ y) ./ 2
    r .= y .- un .- Δt .* vectorfield(f, ū)
end

function residual_jacobian!(j, ::ImplicitMidpoint, f::AbstractFlow, un, y, Δt)
    ū = (un .+ y) ./ 2
    j .= -(Δt / 2) .* jacobian(f, ū)
    @inbounds for i in axes(j, 1)
        j[i, i] += 1
    end
    return j
end

function residual!(r, ::AverageVectorField, f::AbstractFlow, un, y, Δt)
    r .= y .- un
    for τ in AVF_NODES
        r .-= (Δt / 2) .* vectorfield(f, un .+ τ .* (y .- un))
    end
    return r
end

function residual_jacobian!(j, ::AverageVectorField, f::AbstractFlow, un, y, Δt)
    fill!(j, 0)
    for τ in AVF_NODES
        j .-= (Δt * τ / 2) .* jacobian(f, un .+ τ .* (y .- un))
    end
    @inbounds for i in axes(j, 1)
        j[i, i] += 1
    end
    return j
end

function residual!(r, m::DiscreteGradient, f::HamiltonianFlow, un, y, Δt)
    ū = (un .+ y) ./ 2
    ḡ = discrete_gradient(m, f, un, y)
    r .= y .- un .- Δt .* poisson_apply(f.bracket, ū, ḡ)
end

function residual_jacobian!(j, m::DiscreteGradient, f::HamiltonianFlow, un, y, Δt)
    ū = (un .+ y) ./ 2
    ḡ = discrete_gradient(m, f, un, y)
    dḡ = discrete_gradient_jacobian(m, f, un, y)
    j .= -Δt .* poisson_matrix(f.bracket, ū) * dḡ
    add_bracket_term!(j, f.bracket, ū, (-Δt / 2) .* ḡ)
    @inbounds for i in axes(j, 1)
        j[i, i] += 1
    end
    return j
end

"""
    correction_direction(method, space, Δ)

The direction ``c`` of the rank-one correction of the discrete gradient: ``\\Delta\\hat{u}``
for [`Gonzalez`](@ref), ``\\mathbb{M}\\Delta\\hat{u}`` for [`GonzalezMass`](@ref).
"""
correction_direction(::Gonzalez, s::DiscreteSpace, Δ::AbstractVector) = Δ
function correction_direction(::GonzalezMass, s::DiscreteSpace, Δ::AbstractVector)
    mass_matrix(s) * Δ
end

@doc raw"""
    discrete_gradient(method, flow, x, y)

The Gonzalez midpoint discrete gradient of the flow's Hamiltonian between `x` and `y`.

Falls back to the plain gradient at the midpoint when ``\Delta\hat{u}`` is too small for the
correction to be formed — which is not a special case but the limit of the formula, the
correction being ``O(|\Delta\hat{u}|)``.
"""
function discrete_gradient(m::DiscreteGradient, f::HamiltonianFlow, x, y)
    Δ = y .- x
    ū = (x .+ y) ./ 2
    g = gradient(f, ū)
    c = correction_direction(m, f.space, Δ)
    nn = dot(Δ, c)
    abs(nn) < eps(eltype(f))^2 && return g
    α = (hamiltonian(f, y) - hamiltonian(f, x) - dot(g, Δ)) / nn
    g .+ α .* c
end

@doc raw"""
    discrete_gradient_jacobian(method, flow, x, y)

The derivative of [`discrete_gradient`](@ref) with respect to `y`.

With ``\alpha = (\Delta H - \langle g(\bar{u}), \Delta \rangle) / \langle \Delta, c \rangle``
the gradient is ``\bar{g} = g(\bar{u}) + \alpha c``, so

```math
\frac{\partial \bar{g}}{\partial y} = \tfrac{1}{2} H''(\bar{u})
    + \alpha \frac{\partial c}{\partial y}
    + c \left( \frac{\partial \alpha}{\partial y} \right)^T ,
\qquad
\frac{\partial \alpha}{\partial y} =
    \frac{g(y) - \tfrac{1}{2} H''(\bar{u}) \Delta - g(\bar{u})}{\langle \Delta, c\rangle}
  - \frac{2 \alpha \, c}{\langle \Delta, c \rangle} ,
```

with ``\partial c/\partial y`` the identity for [`Gonzalez`](@ref) and ``\mathbb{M}`` for
[`GonzalezMass`](@ref).
"""
function discrete_gradient_jacobian(m::DiscreteGradient, f::HamiltonianFlow, x, y)
    Δ = y .- x
    ū = (x .+ y) ./ 2
    H = hessian(f, ū)
    g = gradient(f, ū)
    c = correction_direction(m, f.space, Δ)
    dc = m isa Gonzalez ? I : mass_matrix(f.space)
    nn = dot(Δ, c)
    abs(nn) < eps(eltype(f))^2 && return H ./ 2
    α = (hamiltonian(f, y) - hamiltonian(f, x) - dot(g, Δ)) / nn
    dα = (gradient(f, y) .- (H * Δ) ./ 2 .- g) ./ nn .- (2α / nn) .* c
    H ./ 2 + α * dc + c * dα'
end

## Stepping

"""
    integrate_step!(û, integrator)

Advance `û` by one step, in place. Returns `û`.
"""
function integrate_step!(û::AbstractVector, integ::Integrator)
    _step!(û, integ.method, integ)
end

function _step!(û, ::ExplicitEuler, integ::Integrator)
    û .+= integ.Δt .* vectorfield(integ.flow, û)
end

function _step!(û, ::RungeKutta4, integ::Integrator)
    f, Δt = integ.flow, integ.Δt
    k1 = vectorfield(f, û)
    k2 = vectorfield(f, û .+ (Δt / 2) .* k1)
    k3 = vectorfield(f, û .+ (Δt / 2) .* k2)
    k4 = vectorfield(f, û .+ Δt .* k3)
    û .+= (Δt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

@doc raw"""
    _step!(û, method, integrator)

One implicit step: a full Newton step first, and a line search only if that fails.

The fast path is `SimpleSolvers.Static` — no line search — because at these
step sizes the residual is a small perturbation of the identity and Newton converges in two
or three iterations from the previous state, so a backtracking search spends several extra
residual evaluations per iteration probing a step length that is accepted at one.

It is not always enough. On the large-amplitude Miura initial condition, implicit midpoint on
the second flow fails to reach the tolerance on about 1 % of its steps. Measured, those steps
land on the same state to every digit that matters — a line search changes the final field by
less than ``10^{-4}`` relative over four hundred steps — so the failures are consequence-free
here. That is a statement about one problem, though, not a guarantee, and a run that silently
accepts unconverged steps has no way to tell the two apart.

So the step is *redone* from ``\hat{u}^n`` with `fallback_linesearch` when the first attempt
does not converge, and only a failure of both is warned about. The cost is paid on the
1 % of steps that need it rather than on all of them.
"""
function _step!(û, method::IntegratorMethod, integ::Integrator)
    integ.formulation === :mixed && return _step_mixed!(û, method, integ)

    un = integ.cache[1]
    un .= û
    params = (un = un,)

    # A singular Newton matrix is a statement about the step size, not a bug. SimpleSolvers'
    # `ldiv!` reports it as a bare `SingularException(k)` naming the zero pivot, which says
    # nothing about what to do; it is caught here and given the context the caller needs.
    _solve_or_explain!(û, integ.solver, integ.state, params)
    converged(integ.solver, integ.state) && return û

    û .= un
    _solve_or_explain!(û, integ.fallback, integ.fallbackstate, params)
    converged(integ.fallback, integ.fallbackstate) || _warn_not_converged()
    return û
end

converged(s, st) = SimpleSolvers.isconverged(SimpleSolvers.status(s, st))

@noinline _warn_not_converged() = @warn("the nonlinear solver did not converge in this step, with or without a line " *
                                        "search; the step size is probably too large for the stiffness of this " *
                                        "discretisation",
    maxlog = 3)

"""
    _step_mixed!(û, method, integ)

One step of the mixed two-field formulation: solve the `2N` system in `(y, v)` and keep `y`.

The auxiliary is re-seeded by solving `M v = ∂H/∂u(uⁿ)` at the start of each step and each retry
(see [`mixed_initial_guess!`](@ref)), so the solve starts from a consistent pair rather than
from whatever the previous step left behind.
"""
function _step_mixed!(û, ::IntegratorMethod, integ::Integrator)
    un = integ.cache[1]
    z = integ.cache[2]
    un .= û
    params = (un = un,)

    mixed_initial_guess!(z, integ.flow, un)
    _solve_or_explain!(z, integ.solver, integ.state, params)
    if converged(integ.solver, integ.state)
        û .= @view z[eachindex(û)]
        return û
    end

    mixed_initial_guess!(z, integ.flow, un)
    _solve_or_explain!(z, integ.fallback, integ.fallbackstate, params)
    converged(integ.fallback, integ.fallbackstate) || _warn_not_converged()
    û .= @view z[eachindex(û)]
    return û
end

"""
    _solve_or_explain!(û, solver, state, params)

`solve!`, with a singular Newton matrix reported as the step-size statement it is.
"""
function _solve_or_explain!(û, solver, state, params)
    try
        solve!(û, solver, state, params)
    catch err
        err isa LinearAlgebra.SingularException || rethrow()
        error("the Newton matrix is singular at pivot $(err.info); the step size is " *
              "probably too large for the stiffness of this discretisation")
    end
    return û
end

function residual!(r, m::ProjectionMethod, f::AbstractFlow, un, y, Δt)
    residual!(r, m.base, f, un, y, Δt)
end

function residual_jacobian!(j, m::ProjectionMethod, f::AbstractFlow, un, y, Δt)
    residual_jacobian!(j, m.base, f, un, y, Δt)
end

function _step!(û, method::ProjectionMethod, integ::Integrator)
    s = space(integ.flow)

    # the level sets to hold are those of the initial state, recorded on the first step
    if isempty(integ.targets)
        append!(integ.targets, (hamiltonian(I, s, û) for I in method.invariants))
    end

    # dispatch on the base method already separates the explicit step from the solver one
    _step!(û, method.base, integ)

    project_invariants!(û, s, method.invariants, integ.targets)
end

@doc raw"""
    tangent_map(integrator, û)

The Jacobian ``D\Phi_{\Delta t}(\hat{u})`` of the numerical map, obtained by differentiating
the step **implicitly** rather than by finite differences.

For an implicit method with residual ``r(y; \hat{u}) = 0`` this is
``-(\partial r/\partial y)^{-1} (\partial r/\partial \hat{u})``, and for the implicit
midpoint rule it works out to the Cayley transform of the docstring of
[`ImplicitMidpoint`](@ref).

This is what the Poisson-map test needs. A central-difference Jacobian has a floor around
``4 \times 10^{-10}`` on these problems, while the defect of a genuine Poisson integrator is
at round-off — so a differenced tangent map would report its own truncation error as a
violation of the Poisson property, three orders too large.
"""
function tangent_map(integ::Integrator, û::AbstractVector)
    _tangent_map(integ.method, integ, û)
end

function _tangent_map(::ExplicitEuler, integ::Integrator, û)
    I + integ.Δt .* jacobian(integ.flow, û)
end

function _tangent_map(::RungeKutta4, integ::Integrator, û)
    f, Δt = integ.flow, integ.Δt
    k1 = vectorfield(f, û)
    J1 = jacobian(f, û)
    k2 = vectorfield(f, û .+ (Δt/2) .* k1)
    J2 = jacobian(f, û .+ (Δt/2) .* k1)
    k3 = vectorfield(f, û .+ (Δt/2) .* k2)
    J3 = jacobian(f, û .+ (Δt/2) .* k2)
    J4 = jacobian(f, û .+ Δt .* k3)
    D1 = J1
    D2 = J2 * (I + (Δt/2) .* D1)
    D3 = J3 * (I + (Δt/2) .* D2)
    D4 = J4 * (I + Δt .* D3)
    I + (Δt/6) .* (D1 .+ 2 .* D2 .+ 2 .* D3 .+ D4)
end

function _tangent_map(method::IntegratorMethod, integ::Integrator, û)
    N = length(û)
    Δt = integ.Δt
    y = copy(û)
    integrate_step!(y, integ)

    # ∂r/∂y from the method's own Jacobian; ∂r/∂uⁿ by the same formulae with the roles of
    # the two arguments exchanged, obtained here by a directional identity rather than by
    # differencing: r is symmetric in how it depends on uⁿ and y for these methods, so
    # ∂r/∂uⁿ = -(I - (∂r/∂y - I) reflected). Rather than rely on that, differentiate the
    # residual in uⁿ explicitly.
    Ry = zeros(eltype(û), N, N)
    residual_jacobian!(Ry, method, integ.flow, û, y, Δt)
    Ru = residual_jacobian_initial(method, integ.flow, û, y, Δt)
    -(Ry \ Ru)
end

"""
    residual_jacobian_initial(method, flow, uⁿ, y, Δt)

The derivative of the step's residual with respect to the *initial* state ``\\hat{u}^n``,
the second half of what [`tangent_map`](@ref) needs.
"""
function residual_jacobian_initial(::ImplicitMidpoint, f::AbstractFlow, un, y, Δt)
    ū = (un .+ y) ./ 2
    J = -(Δt / 2) .* jacobian(f, ū)
    J - I
end

function residual_jacobian_initial(::AverageVectorField, f::AbstractFlow, un, y, Δt)
    N = length(un)
    J = zeros(eltype(un), N, N)
    for τ in AVF_NODES
        J .-= (Δt * (1 - τ) / 2) .* jacobian(f, un .+ τ .* (y .- un))
    end
    J - I
end

function residual_jacobian_initial(m::DiscreteGradient, f::HamiltonianFlow, un, y, Δt)
    # the discrete gradient is symmetric under x ↔ y up to the sign of Δ, so its derivative
    # in the first argument is obtained from the same expression with the roles exchanged
    ū = (un .+ y) ./ 2
    ḡ = discrete_gradient(m, f, un, y)
    dḡ = discrete_gradient_jacobian_initial(m, f, un, y)
    J = -Δt .* poisson_matrix(f.bracket, ū) * dḡ
    add_bracket_term!(J, f.bracket, ū, (-Δt / 2) .* ḡ)
    J - I
end

function discrete_gradient_jacobian_initial(m::DiscreteGradient, f::HamiltonianFlow, x, y)
    Δ = y .- x
    ū = (x .+ y) ./ 2
    H = hessian(f, ū)
    g = gradient(f, ū)
    c = correction_direction(m, f.space, Δ)
    dc = m isa Gonzalez ? I : mass_matrix(f.space)
    nn = dot(Δ, c)
    abs(nn) < eps(eltype(f))^2 && return H ./ 2
    α = (hamiltonian(f, y) - hamiltonian(f, x) - dot(g, Δ)) / nn
    dα = (-gradient(f, x) .- (H * Δ) ./ 2 .+ g) ./ nn .+ (2α / nn) .* c
    H ./ 2 - α * dc + c * dα'
end

@doc raw"""
    poisson_defect(integrator, û)

The relative defect of the Poisson-map property,

```math
\frac{\left\| D\Phi \, \mathbb{P} \, D\Phi^T - \mathbb{P} \right\|}{\left\| \mathbb{P} \right\|} ,
```

with ``D\Phi`` the analytic [`tangent_map`](@ref).

For the implicit midpoint rule on a constant bracket this is at round-off; for the average
vector field method it is around ``10^{-5}``, and for explicit Euler it is of order one.

A [`MetriplecticFlow`](@ref) is rejected rather than measured, and the error says which of the
two reasons applies. In the four-argument form there is no ``\mathbb{P}`` to preserve; with a
Poisson half there is one, but the metric half breaks
``D\Phi \, \mathbb{P} \, D\Phi^T = \mathbb{P}`` *by construction*, so the number would measure
the dissipation rather than the method — which is worse than refusing, because it looks like
an answer.
"""
function poisson_defect(integ::Integrator, û::AbstractVector)
    P = poisson_matrix(_poisson_bracket(integ.flow), û)
    DΦ = tangent_map(integ, û)
    maximum(abs, DΦ * P * DΦ' - P) / maximum(abs, P)
end

"""
    _poisson_bracket(flow)

The bracket [`poisson_defect`](@ref) measures against, or an error naming why there is none.

Routed through the `bracket` accessor rather than `f.bracket`, so that a flow carrying no
Poisson half reports the accessor it is missing instead of a `FieldError` about a field the
[`AbstractFlow`](@ref) interface never asked it to have.
"""
_poisson_bracket(f::AbstractFlow) = bracket(f)

function _poisson_bracket(f::MetriplecticFlow)
    throw(ArgumentError(
        "poisson_defect measures how far the numerical map is from preserving 𝔓, and a " *
        "MetriplecticFlow is not a candidate for it: " *
        (f.bracket === nothing ?
         "this one carries no Poisson half at all, having been built with the " *
         "four-argument form" :
         "its metric half breaks the Poisson property by construction, so the result " *
         "would measure the dissipation and not the method") *
        ". Build the Poisson half on its own as a HamiltonianFlow to measure the method"))
end

@doc raw"""
    stability_limit(flow, û)

The explicit Runge-Kutta stability limit ``2\sqrt{2} / \rho`` with ``\rho`` the spectral
radius of the linearised flow.

``2\sqrt{2}`` is the **imaginary-axis** crossing of the classical fourth-order Runge-Kutta
stability region, so this is the right constant for a flow whose spectrum lies on that axis.
A [`HamiltonianFlow`](@ref) is such a flow: ``\mathbb{P}`` is antisymmetric, so
``\mathbb{P} H''`` is similar to a skew matrix and ``\rho`` is essentially the largest
imaginary part. ``\rho`` grows like ``h^{-3}`` through the third derivative, which is why an
explicit method needs so many more steps than an implicit one at the same resolution.

A [`MetriplecticFlow`](@ref) is **not** such a flow, and the number returned here is then an
estimate rather than the limit. The metric half contributes a spectrum with a negative real
part — that is what dissipation is — and the region's crossing on the negative real axis is
``2.7853``, not ``2\sqrt{2} = 2.8284``. For a spectrum spread through the left half-plane the
governing crossing is neither of the two, so read this as an order of magnitude on a
dissipative field and not as a threshold.
"""
stability_limit(f::AbstractFlow, û::AbstractVector) = 2 * sqrt(2) /
                                                      maximum(abs, eigvals(jacobian(f, û)))

@doc raw"""
    project_invariants!(û, space, invariants, targets; maxiter = 20, tol = 1e-14)

Project `û` onto the joint level set ``I_k(\hat{u}) = \mathrm{targets}[k]`` along the span
of the invariants' gradients,

```math
\hat{u} \leftarrow \tilde{u} + \sum_k \lambda_k \frac{\partial I_k}{\partial \hat{u}} ,
```

solving the small nonlinear system for ``\lambda`` by Newton's method.

Pass the mass among the invariants if it is to be kept: the correction direction of two
Hamiltonians is not mass-neutral, and projecting onto them alone loses the Casimir.
"""
function project_invariants!(û::AbstractVector, s::DiscreteSpace, invariants, targets;
        maxiter::Integer = 20, tol = 1e-14)
    k = length(invariants)
    G = hcat((gradient(I, s, û) for I in invariants)...)
    λ = zeros(eltype(û), k)
    ũ = copy(û)
    for _ in 1:maxiter
        û .= ũ .+ G * λ
        r = [hamiltonian(invariants[i], s, û) - targets[i] for i in 1:k]
        maximum(abs, r) < tol && break
        J = [dot(gradient(invariants[i], s, û), G[:, j]) for i in 1:k, j in 1:k]
        λ .-= J \ r
    end
    return û
end
