
@doc raw"""
    invariants(system, û)

The conserved quantities of `system` at `û`, as a `NamedTuple`.

Assembled from a **single** evaluation of the field and its derivative rather than by
calling each Hamiltonian in turn. This is called once per time step by [`integrate`](@ref),
and at the step counts these runs reach it is a third of the total time.
"""
function invariants end

"""
    invariant_names(system)

The names of the quantities [`invariants`](@ref) returns, in order.
"""
function invariant_names end

function invariants(sys::KdVSystem, û::AbstractVector)
    s = sys.space
    w = quadrature_weights(s)
    uh = field(s, û, 0)
    ux = field(s, û, 1)
    wu = w .* uh
    # H₁ = ½∫(u_x² - 2u³). This DUPLICATES `hamiltonian(KdVHamiltonian1(), …)` on purpose --
    # see the docstring -- so the two have to be changed together; a sign convention that
    # moved in one and not the other would show up as a spurious drift and nothing else.
    (H1 = (dot(w, ux .^ 2) - 2 * dot(wu, uh .^ 2)) / 2,
        H2 = dot(wu, uh) / 2,
        C0 = sum(wu))
end

invariant_names(::KdVSystem) = (:H1, :H2, :C0)

@doc raw"""
    invariants(sys::MiuraSystem, v̂)

The invariants of a [`MiuraSystem`](@ref), **reported in ``u``**.

The state is ``\hat{v}``, so the map through [`miura_map`](@ref) comes first and everything
after it is the ordinary KdV computation. This is what lets a `MiuraSystem` be handed to
[`integrate`](@ref) and to the plotting routines alongside the ``u``-chart systems and
compared with them directly.

Note that ``C_{0,d} = \int u_h = -\int v_h^2`` is negative by construction here, and is *not*
a Casimir in this chart — the Casimir of ``\mathbb{P}^1`` in ``\hat{v}`` is
[`miura_casimir`](@ref), ``\int v_h``, which corresponds to nothing in ``u``.
"""
function invariants(sys::MiuraSystem, v̂::AbstractVector)
    s = sys.space
    w = quadrature_weights(s)
    û = miura_map(s, v̂, sys.H.λ)
    uh = field(s, û, 0)
    ux = field(s, û, 1)
    wu = w .* uh
    # H₁ = ½∫(u_x² - 2u³). This DUPLICATES `hamiltonian(KdVHamiltonian1(), …)` on purpose --
    # see the docstring -- so the two have to be changed together; a sign convention that
    # moved in one and not the other would show up as a spurious drift and nothing else.
    (H1 = (dot(w, ux .^ 2) - 2 * dot(wu, uh .^ 2)) / 2,
        H2 = dot(wu, uh) / 2,
        C0 = sum(wu))
end

invariant_names(::MiuraSystem) = (:H1, :H2, :C0)

function invariants(sys::CamassaHolmSystem, m̂::AbstractVector)
    s = sys.space
    w = quadrature_weights(s)
    û = velocity(sys, m̂)               # one Helmholtz solve for all three
    uh = field(s, û, 0)
    ux = field(s, û, 1)
    (H1 = (dot(w, uh .^ 2) + dot(w, ux .^ 2)) / 2,
        H2 = (dot(w, uh .^ 3) + dot(w .* uh, ux .^ 2)) / 2,
        C0 = dot(w, field(s, m̂, 0)))
end

invariant_names(::CamassaHolmSystem) = (:H1, :H2, :C0)

function invariants(sys::BurgersSystem, û::AbstractVector)
    s = sys.space
    (H = hamiltonian(sys.H, s, û),
        C = hamiltonian(sys.C, s, û))
end

invariant_names(::BurgersSystem) = (:H, :C)

@doc raw"""
    Trajectory

The record of a run: the output times, the initial values of the invariants, and the
**windowed maxima** of their deviations.

`deviation[k, j]` is the largest ``|I_k - I_k(0)|`` reached *anywhere* inside output window
`j`, not the value at its end.

# Why windowed maxima and not samples

The error of a geometric integrator oscillates with a period of tens of steps. Sampling it
every `stride` steps over a run of millions aliases that oscillation into noise, and the
noise can be an order of magnitude below the true envelope — so a plot built from samples
shows a method conserving an invariant far better than it does. The invariants are therefore
evaluated at *every* step and only the reporting is windowed.
"""
struct Trajectory{T, NT}
    t::Vector{T}
    reference::NT
    deviation::Matrix{T}
    names::NTuple
    final::Vector{T}
end

Base.length(traj::Trajectory) = length(traj.t)

"""
    deviation(traj, name)

The windowed deviation series of the invariant called `name`.
"""
function deviation(traj::Trajectory, name::Symbol)
    k = findfirst(==(name), traj.names)
    k === nothing && throw(ArgumentError(
        "no invariant called $(name); this trajectory has $(traj.names)"))
    traj.deviation[k, :]
end

@doc raw"""
    integrate(system, integrator, û₀, nsteps; stride = 1)

Run `nsteps` steps of `integrator` from `û₀`, recording the invariants at every step and
reporting them as windowed maxima over `nsteps ÷ stride` windows.

Returns a [`Trajectory`](@ref). The initial state is not modified.

```jldoctest
julia> sys = KdVSystem(SplineSpace(16, 3));

julia> integ = Integrator(sys.flow1, ImplicitMidpoint(), 1e-3);

julia> traj = integrate(sys, integ, project(sys.space, sin), 20; stride = 5);

julia> length(traj)
4
```
"""
function integrate(sys, integ::Integrator{T}, û₀::AbstractVector, nsteps::Integer;
        stride::Integer = 1) where {T}
    stride ≥ 1 || throw(ArgumentError("the output stride must be positive, got $(stride)"))
    nsteps ≥ stride || throw(ArgumentError(
        "the run is $(nsteps) steps but the output stride is $(stride); there would be no " *
        "complete output window"))

    û = collect(T, û₀)
    ref = invariants(sys, û)
    names = invariant_names(sys)
    nk = length(names)
    # Ceiling, and a final short window, so that all `nsteps` steps are taken even when the
    # stride does not divide them. Rounding down instead would silently drop up to
    # `stride - 1` steps -- 381 of the 766781 of the explicit cosine run -- while the tables
    # went on reporting `nsteps`, and would end the run before the final time.
    nout = cld(nsteps, stride)

    t = Vector{T}(undef, nout)
    dev = zeros(T, nk, nout)
    ref0 = T[getproperty(ref, n) for n in names]

    done = 0
    for j in 1:nout
        peak = zeros(T, nk)
        for _ in 1:min(stride, nsteps - done)
            integrate_step!(û, integ)
            done += 1
            cur = invariants(sys, û)
            for k in 1:nk
                d = abs(getproperty(cur, names[k]) - ref0[k])
                d > peak[k] && (peak[k] = d)
            end
        end
        t[j] = done * integ.Δt
        dev[:, j] .= peak
    end

    Trajectory{T, typeof(ref)}(t, ref, dev, names, û)
end

@doc raw"""
    drift(traj, name)

The largest **relative** deviation of the invariant `name` over the whole run,
``\max_j |I - I_0| / |I_0|``.

Use this for the Hamiltonians. For an invariant whose reference value is zero — the mass of
the cosine initial condition, for one — use [`absolute_drift`](@ref) instead; dividing by
zero there would report `Inf` for a quantity that is in fact conserved to round-off.
"""
function drift(traj::Trajectory{T}, name::Symbol; atol = sqrt(eps(T))) where {T}
    ref = abs(getproperty(traj.reference, name))
    d = maximum(deviation(traj, name))
    # The guard is on `atol` rather than on exact zero because the reference value of a
    # quantity that vanishes mathematically does not vanish numerically: the mass of the
    # cosine initial condition comes out around 1e-17, and dividing a 1e-15 deviation by it
    # would report a 100-fold "relative drift" of a quantity conserved to round-off.
    ref > atol || throw(ArgumentError(
        "the reference value of $(name) is $(ref), which is below atol = $(atol), so a " *
        "relative drift is not meaningful; use absolute_drift"))
    d / ref
end

@doc raw"""
    entropy_production(flow, û)

The rate at which the metric half of a [`MetriplecticFlow`](@ref) dissipates the entropy,

```math
(S, S)_d = \left( \frac{\partial S}{\partial \hat{u}} \right)^T \mathbb{G} (\hat{u})
           \left( \frac{\partial S}{\partial \hat{u}} \right) \ge 0 ,
```

non-negative for the same reason [`ispositive_semidefinite`](@ref) is true, and equal to
``- \dot{S}`` when the Poisson half is absent.

# Why this and not the drift in `S`

Because they fail differently, and only one of the two failures shows up as a drift. A metric
bracket assembled with a sign error still dissipates *something*: it produces entropy of the
wrong sign, and ``S`` then grows, which is a backward heat equation and not a small error. A
bracket that has lost its degeneracy instead leaks ``H``. Measuring only ``|S - S_0|`` — which
is what [`deviation`](@ref) records — reports both as "the quantity moved", and by the same
amount whichever way it moved.

So this is the quantity to assert a **sign** on, and the ``S`` series the one to assert
**monotonicity** on. Neither is implied by the other: the Poisson half contributes
``\{S, H\}`` to ``\dot{S}``, which is not sign-definite, so a flow with a bracket can produce
entropy at every step and still have ``S`` rise.
"""
function entropy_production(f::MetriplecticFlow, û::AbstractVector)
    g = entropy_gradient(f, û)
    dot(g, metric_apply(f.metric, û, g))
end

"""
    absolute_drift(traj, name)

The largest absolute deviation of the invariant `name` over the whole run.
"""
absolute_drift(traj::Trajectory, name::Symbol) = maximum(deviation(traj, name))

@doc raw"""
    growth(traj, name; fraction = 1//10)

The ratio of the error envelope over the last `fraction` of the run to that over the first.

Near one means a bounded oscillation; large means a secular drift. This is the number that
separates "the invariant departs by ``10^{-6}`` and stays there" from "the invariant is
being lost", and the two look identical in a single end-of-run figure.
"""
function growth(traj::Trajectory, name::Symbol; fraction = 1//10)
    d = deviation(traj, name)
    n = length(d)
    m = max(1, round(Int, n * fraction))
    first_env = maximum(@view d[1:m])
    last_env = maximum(@view d[(end - m + 1):end])
    iszero(first_env) ? one(eltype(d)) : last_env / first_env
end

@doc raw"""
    energyplot(trajectories, labels; name = :H1, kwargs...)
    energyplot(trajectories, labels, names; kwargs...)
    stateplot(system, states, labels; xleft = 0, kwargs...)

Plot the error of **one** invariant against time, and the initial and final states.

Defined in a package extension: load `CairoMakie` to make them available.

```julia
using GeometricBrackets, CairoMakie
fig = energyplot(trajs, labels; name = :H2)
```

One invariant per figure, and one *family* of vector fields per figure. Putting several
invariants side by side forces a shared axis on quantities whose errors sit orders of
magnitude apart, and putting every run on one axis makes eight curves of four colours where
the comparison that matters is within a family.

The encoding is deliberate and worth keeping if these are adapted: **colour carries the
integrator and line style carries the vector field**, so no series is identified by colour
alone and the figure survives being printed in grey. The palette is Okabe-Ito.

The three-argument method draws a row of panels, one per invariant, each with its own y
axis. It exists for the manuscript's own figure layout; the single-panel method remains the
default for the reason just given.

If the reference value of the invariant vanishes — the mass of the cosine initial condition
does, to round-off — the panel reports the absolute error instead of the relative one, and
says so on the axis.
"""
function energyplot end

@doc (@doc energyplot)
function stateplot end

@doc raw"""
    sweepplot(dts, series; kwargs...)

The step-size sweep of the backward error analysis, as two stacked panels: the departure of
the respective other Hamiltonian against ``\Delta t``, and its **excess over the
``\Delta t \to 0`` plateau**.

`series` is a vector of `label => values` pairs, one per run, with `values` aligned to `dts`.

Both panels are needed and neither alone will do. On the upper one the curves sit within a
factor of eight of each other and nothing is visible; the dependence on ``\Delta t`` only
appears once each run's own plateau — its value at the smallest step size — is subtracted.
What is then left is the ``O(\Delta t^2)`` of the energy-preserving methods, and *nothing at
all* for the midpoint rule on the first flow, which is the whole point: its error is spatial,
and the time discretisation adds none of its own.

A dotted ``O(\Delta t^2)`` guide is drawn on the lower panel, anchored to the first series
that has a nonzero excess.

Defined in a package extension: load `CairoMakie` to make it available.
"""
function sweepplot end

@doc raw"""
    convergenceplot(xs, series; kwargs...)

A refinement study on log-log axes: `series` is a vector of `label => errors` pairs against
the resolutions `xs`, with fitted orders shown in the legend.

Written for the claims that are about a rate *not* being achieved as much as for the ones
where it is. A residual that is flat under refinement — the ``\approx 0.42`` Jacobiator of
the second KdV bracket, or the ``\approx 0.73`` structure-constant violation of the nodal
finite elements — reads as a horizontal line against the sloped ones, which is the honest way
to show that no amount of refinement will fix it.

Pass `guide = q` to draw a dotted ``O(h^q)`` reference line.

Defined in a package extension: load `CairoMakie` to make it available.
"""
function convergenceplot end

"""
    INTEGRATOR_COLORS

The Okabe-Ito colour assigned to each integrator by [`energyplot`](@ref). Colour carries
the *integrator*; line style carries the vector field.
"""
const INTEGRATOR_COLORS = Dict(
    "midpoint" => "#0072B2",
    "discrete gradient" => "#D55E00",
    "discrete gradient (mass)" => "#F0A070",
    "AVF" => "#009E73",
    "RK4" => "#E69F00",
    "Euler" => "#CC79A7"
)

"""
    FLOW_STYLES

The line style assigned to each vector field by [`energyplot`](@ref): solid for the first
flow, dashed for the second, dash-dotted for the Miura flow, which is a third vector field
and needs a third pattern.
"""
const FLOW_STYLES = Dict(1 => :solid, 2 => :dash, :M => :dashdot)

"""
    ERROR_FLOOR

The floor of the logarithmic axis in [`energyplot`](@ref). Deviations below round-off are
clamped to it rather than dropped, so that a method conserving an invariant exactly shows
as a flat line at the floor instead of a gap.
"""
const ERROR_FLOOR = 1e-16
