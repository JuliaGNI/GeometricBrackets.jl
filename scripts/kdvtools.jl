#
# Shared assembly for the KdV diagnostics.
#
#     include(joinpath(@__DIR__, "kdvtools.jl"))
#     using .KdVTools
#
# The Python prototypes carried these in four or five copies apiece -- `nq_for` in four files,
# a `Sys` class in three, the K2 blocks in two -- and `docs/src/verification.md` records that
# the duplication cost real errors when a sign convention changed and only some copies were
# updated. One definition each, here.
#
# Beware the broadcast orientation throughout: Φ is (nb × nq) and the weights are (nq,), so
# numpy's `W * P[d]`, which scales the QUADRATURE axis, is `Φ .* transpose(W)` in Julia.

module KdVTools

using GeometricBrackets
using LinearAlgebra
using Printf
using SimpleSplines: UniformMesh, GradedMesh, RandomMesh

export L, U0, nq_for, setup, fieldat, gram, skew_S,
       grad_H1, grad_H2, grad_H3, hess_H1, mass_grad,
       K2_blocks, K2_blocks_unsym, K2apply,
       rates, o2, e3, newton, fd_newton,
       KdVSys, sysH1, sysH2, sysC0, gH1, gH2, hessH1, f1, Df1, K2, f2, Df2,
       SG, midpoint, avf, dgrad_H1, dgrad, dgrad_H1_mass, dgrad_mass, euler,
       integrate_windowed, env_drift, env_growth

const L = 2π

"A generic smooth initial field, exciting more than one mode on purpose."
U0(x) = sin(x) + 0.4 * cos(2x)

"Gauss points per cell that integrate degree 3p-1 exactly."
nq_for(p) = ceil(Int, 1.5p)

"""
    setup(n; p, uniform, graded, nq, f) -> (space, coefficients)

The periodic B-spline space of degree `p` on `n` cells, and the projection of `f` onto it.
`graded` takes precedence over `uniform`, matching the Python's keyword handling.
"""
function setup(n; p = 3, uniform = true, graded = false, nq = nothing, f = U0)
    nqv = max(nq === nothing ? nq_for(p) : nq, 2)
    mesh = graded ? GradedMesh(n, L) : (uniform ? UniformMesh(n, L) : RandomMesh(n, L))
    s = SplineSpace(mesh, p; nq = nqv)
    return s, project(s, f)
end

"The d-th derivative of the field of `û`, sampled at the quadrature nodes.\nNamed `fieldat` rather than `field`/`fld`: the latter is `Base.fld`, floor division."
fieldat(s, û, d) = transpose(basis_values(s, d)) * û

"The weighted outer product ∫ φ^(da)_k φ^(db)_l."
function gram(s, da, db)
    basis_values(s, da) *
    transpose(basis_values(s, db) .* transpose(quadrature_weights(s)))
end

"The skew part of S = ∫ φ_k φ_l', which is what the first bracket is built from."
skew_S(s) = (S = gram(s, 0, 1); (S - transpose(S)) / 2)

grad_H1(s, û) = gradient(KdVHamiltonian1(), s, û)
grad_H2(s, û) = Matrix(mass_matrix(s)) * û
grad_H3(s, û) = gradient(KdVHamiltonian3(), s, û)
hess_H1(s, û) = hessian(KdVHamiltonian1(), s, û)

"dC0/du_i = ∫ φ_i, the gradient of the mass."
mass_grad(s) = basis_values(s, 0) * quadrature_weights(s)

@doc raw"""
    K2_blocks(s) -> (Ku, K0)

The u-linear tensor and constant block of the SKEW-symmetrised second bracket,

    Ku[m,k,l] = -2 ∫ φ_m ( φ_k φ_l' - φ_l φ_k' ) ,
    K0[k,l]   = ½ ∫ ( φ_k' φ_l'' - φ_l' φ_k'' ) .

Assembled directly rather than recovered from `poisson_tensor` as `M C[m] M`: both blocks are
antisymmetric in (k,l) BY CONSTRUCTION at any quadrature and any mesh, and the quadrature
sweeps report `|K2 + K2'|` as a check of exactly that. Going through the mass matrix would
turn a structural zero into 1e-14.
"""
function K2_blocks(s)
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    nb = nbasis(s)
    T1 = [sum(Φ0[m, q] * Φ0[k, q] * Φ1[l, q] * W[q] for q in eachindex(W))
          for m in 1:nb, k in 1:nb, l in 1:nb]
    G = gram(s, 1, 2)
    return -2.0 .* (T1 .- permutedims(T1, (1, 3, 2))), (G - transpose(G)) / 2
end

@doc raw"""
    K2_blocks_unsym(s) -> (Ku, K0)

The UNSYMMETRISED blocks, as first written in the notes,

    Ku[m,k,l] = -( 4 ∫ φ_m φ_k φ_l' + 2 ∫ φ_m' φ_k φ_l ) ,   K0[k,l] = ∫ φ_k' φ_l'' .

Antisymmetric only once the quadrature integrates the total derivatives exactly, i.e. to
degree 3p-1. Kept for the equivalence check against [`K2_blocks`](@ref); the brackets
themselves use the skew form.
"""
function K2_blocks_unsym(s)
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    nb = nbasis(s)
    T1 = [sum(Φ0[m, q] * Φ0[k, q] * Φ1[l, q] * W[q] for q in eachindex(W))
          for m in 1:nb, k in 1:nb, l in 1:nb]
    T3 = [sum(Φ1[m, q] * Φ0[k, q] * Φ0[l, q] * W[q] for q in eachindex(W))
          for m in 1:nb, k in 1:nb, l in 1:nb]
    return -(4.0 .* T1 .+ 2.0 .* T3), gram(s, 1, 2)
end

"K2(u) c for the skew-symmetrised bracket, matrix-free -- O(NQ) rather than O(N^3)."
function K2apply(s, û, c)
    W = quadrature_weights(s)
    Φ0, Φ1, Φ2 = basis_values(s, 0), basis_values(s, 1), basis_values(s, 2)
    uh = fieldat(s, û, 0)
    ch, cx, cxx = fieldat(s, c, 0), fieldat(s, c, 1), fieldat(s, c, 2)
    -2.0 .* (Φ0 * (W .* uh .* cx) - Φ1 * (W .* uh .* ch)) .+
    0.5 .* (Φ1 * (W .* cxx) - Φ2 * (W .* cx))
end

"Observed convergence orders from a sequence of errors at resolutions `ns`."
function rates(errs, ns)
    [log(errs[i - 1] / errs[i]) / log(ns[i] / ns[i - 1]) for i in 2:length(errs)]
end

"Format a vector of orders like the Python's list-of-strings repr, for diffing."
o2(v) = "[" * join(["'" * (@sprintf("%.2f", x)) * "'" for x in v], ", ") * "]"
e3(v) = join([@sprintf("%.3e", x) for x in v], "  ")

"Newton with an analytic Jacobian; the systems here are small and dense."
function newton(res, jac, x0; tol = 1e-13, itmax = 40)
    x = copy(x0)
    for _ in 1:itmax
        r = res(x)
        maximum(abs, r) < tol && break
        x = x - jac(x) \ r
    end
    x
end

"Newton with a finite-difference Jacobian; the systems here are small and dense."
function fd_newton(res, x0, n; tol = 1e-13, itmax = 60, eps = 1e-7)
    y = copy(x0)
    for _ in 1:itmax
        r = res(y)
        maximum(abs, r) < tol && break
        J = zeros(n, n)
        for j in 1:n
            yp = copy(y)
            yp[j] += eps
            J[:, j] = (res(yp) - r) ./ eps
        end
        y = y - J \ r
    end
    y
end

# --------------------------------------------------------------------------
# the two semi-discrete flows, their exact Jacobians, and the integrators
#
# One `KdVSys`, where the Python had three near-identical copies -- `kdvsim.KdVSys`,
# `verify_kdv_timedisc.KdVSys` and `verify_kdv_miura.Chart` -- whose drift when a sign
# convention changed is recorded in `docs/src/verification.md`.
# --------------------------------------------------------------------------

"""
    KdVSys(n; p, uniform, graded, nq, u0)

Assembly plus the two semi-discrete vector fields, their Jacobians and the invariants.

Named `KdVSys` and not `Sys`: `Base.Sys` is the system-information module, so a bare `Sys`
exported from here is ambiguous at every call site that also does `using Base`.
"""
struct KdVSys
    s::Any
    X::Any
    W::Any
    M::Any
    Minv::Any
    d0::Any
    S::Any
    K1::Any
    P1::Any
    Ku::Any
    K0::Any
    g::Any
    n::Any
    Lx::Any
end

function KdVSys(n; p = 3, uniform = true, graded = false, nq = nothing, u0 = U0)
    s, d0 = setup(n; p, uniform, graded, nq, f = u0)
    M = Matrix(mass_matrix(s))
    Minv = inv(M)
    S = skew_S(s)
    Ku, K0 = K2_blocks(s)
    KdVSys(s, quadrature_nodes(s), quadrature_weights(s), M, Minv, d0, S, gram(s, 1, 1),
        Minv * S * Minv, Ku, K0, mass_grad(s), nbasis(s), domainlength(s))
end

sysH1(y::KdVSys, d) = dot(y.W, fieldat(y.s, d, 1) .^ 2 - 2 .* fieldat(y.s, d, 0) .^ 3) / 2
sysH2(y::KdVSys, d) = dot(y.W, fieldat(y.s, d, 0) .^ 2) / 2
sysC0(y::KdVSys, d) = dot(y.W, fieldat(y.s, d, 0))
gH1(y::KdVSys, d) = grad_H1(y.s, d)
gH2(y::KdVSys, d) = y.M * d
hessH1(y::KdVSys, d) = hess_H1(y.s, d)

f1(y::KdVSys, d) = y.P1 * gH1(y, d)
Df1(y::KdVSys, d) = y.P1 * hessH1(y, d)
K2(y::KdVSys, d) = y.K0 + sum(d[m] .* y.Ku[m, :, :] for m in axes(y.Ku, 1))
f2(y::KdVSys, d) = y.Minv * (K2(y, d) * d)
function Df2(y::KdVSys, d)
    y.Minv * (K2(y, d) + [sum(y.Ku[j, i, l] * d[l] for l in eachindex(d))
      for i in eachindex(d), j in eachindex(d)])
end

"The two-point Gauss nodes of the average-vector-field method."
const SG = (0.5 - sqrt(3.0) / 6.0, 0.5 + sqrt(3.0) / 6.0)

"Implicit midpoint on flow `which`. Returns (y, Dy) with Dy exact, by implicit differentiation."
function midpoint(y::KdVSys, d, dt, which = 1)
    f, Df = which == 1 ? (f1, Df1) : (f2, Df2)
    Id = Matrix{Float64}(I, y.n, y.n)
    z = newton(w -> w - d - dt .* f(y, 0.5 .* (d + w)),
        w -> Id - 0.5dt .* Df(y, 0.5 .* (d + w)), d)
    A = Df(y, 0.5 .* (d + z))
    return z, (Id - 0.5dt .* A) \ (Id + 0.5dt .* A)
end

"Two-point-Gauss average vector field method: average the whole field."
function avf(y::KdVSys, d, dt, which = 1)
    f, Df = which == 1 ? (f1, Df1) : (f2, Df2)
    fbar(w) = sum(f(y, d + t .* (w - d)) for t in SG) / 2
    dfy(w) = sum(t .* Df(y, d + t .* (w - d)) for t in SG) / 2
    dfd(w) = sum((1 - t) .* Df(y, d + t .* (w - d)) for t in SG) / 2
    Id = Matrix{Float64}(I, y.n, y.n)
    z = newton(w -> w - d - dt .* fbar(w), w -> Id - dt .* dfy(w), d)
    return z, (Id - dt .* dfy(z)) \ (Id + dt .* dfd(z))
end

"""
    dgrad_H1(y, x, z; metric = I)

Gonzalez midpoint discrete gradient of H1: `grad H1(x̄)` corrected by a rank-one term so the
discrete-gradient property holds exactly. For a QUADRATIC Hamiltonian the correction vanishes
identically and this reduces to the plain midpoint gradient -- hence the name. `metric = :mass`
measures the increment in the mass metric instead of the Euclidean one.
"""
function dgrad_H1(y::KdVSys, x, z; metric = :euclidean)
    xbar, dx = 0.5 .* (x + z), z - x
    g = gH1(y, xbar)
    w = metric === :mass ? y.M * dx : dx
    nn = dot(dx, w)
    abs(nn) < 1e-30 && return g
    g + ((sysH1(y, z) - sysH1(y, x)) - dot(g, dx)) / nn .* w
end

dgrad_H1_mass(y::KdVSys, x, z) = dgrad_H1(y, x, z; metric = :mass)

"Discrete-gradient method on flow 1; Jacobian numerical."
function dgrad(y::KdVSys, d, dt; metric = :euclidean)
    fd_newton(w -> w - d - dt .* (y.P1 * dgrad_H1(y, d, w; metric)), d, y.n)
end

dgrad_mass(y::KdVSys, d, dt) = dgrad(y, d, dt; metric = :mass)

function euler(y::KdVSys, d, dt, which = 1)
    f, Df = which == 1 ? (f1, Df1) : (f2, Df2)
    d + dt .* f(y, d), Matrix{Float64}(I, y.n, y.n) + dt .* Df(y, d)
end

"""
    integrate_windowed(y, step, nsteps; stride)

Integrate, recording the WINDOWED MAXIMUM of |invariant - reference| over each block of
`stride` steps. `step` is a function of the state alone. Returns `(final, reference,
deviation)` with `deviation[q]` the envelope of invariant `q` (1 = H1, 2 = H2, 3 = C0).
"""
function integrate_windowed(y::KdVSys, step, nsteps; stride = 1)
    d = copy(y.d0)
    ref = [sysH1(y, d), sysH2(y, d), sysC0(y, d)]
    dev = [Float64[], Float64[], Float64[]]
    run = zeros(3)
    for i in 1:nsteps
        d = step(d)
        cur = [sysH1(y, d), sysH2(y, d), sysC0(y, d)]
        run = max.(run, abs.(cur - ref))
        if i % stride == 0 || i == nsteps
            for q in 1:3
                push!(dev[q], run[q])
            end
            run = zeros(3)
        end
    end
    return d, ref, dev
end

"""
    env_drift(env, ref)

Maximum relative deviation of an envelope from its reference value.

Named apart from `GeometricBrackets.drift`, which takes a `Trajectory` and an invariant
name; a bare `drift` exported from here would be ambiguous with it. Same for
`env_growth` against `GeometricBrackets.growth`.
"""
env_drift(env, ref) = maximum(env) / max(abs(ref), 1e-30)

"""
    env_growth(env)

Envelope over the last tenth of a run against the first. Near one for a bounded oscillation,
large for a drift.
"""
function env_growth(env)
    m = max(1, length(env) ÷ 10)
    a, b = maximum(env[2:min(m + 1, end)]), maximum(env[max(end - m + 1, 1):end])
    a > 0 ? b / a : Inf
end

end # module
