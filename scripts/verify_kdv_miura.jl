#!/usr/bin/env julia
#
# The Miura chart: where the second KdV bracket becomes exactly Poisson, and what that costs.
#
#     julia --project=scripts scripts/verify_kdv_miura.jl
#
#   1.  the Miura factorisation (2v + d_x) d_x (2v - d_x) = 4u d_x + 2u_x - d_x^3, symbolically;
#   2.  where the discrete map is a local diffeomorphism: DM is singular exactly on int v = 0;
#   3.  P^2_Miura is exactly Poisson at EVERY N, against the flat 0.42 of the Galerkin P^2;
#   4.  the closed form of the notes, and the price -- the two interior M^-1 make it DENSE;
#   5.  consistency: the two discretisations agree on smooth data but differ entrywise at
#       order one, in the same under-resolved modes as the Jacobi violation;
#   6.  where the chart lives: C_0,d = -int v_h^2 exactly, so the image is a half-space, and
#       none of the manuscript's examples has a preimage at lambda = 0;
#   6b. the spectral parameter puts all of them back in the chart;
#   7.  the three-field semi-discrete system, and why the pairings must be skew-symmetrised;
#   8.  what the flow conserves -- and the one law it gives up, the KdV mass;
#   9.  the two charts are NOT the same numerical method: they differ at O(dt^3) per step,
#       because only affine maps commute with a Runge-Kutta method. But the Poisson-map
#       property IS chart-free, so midpoint in v pushes forward to a genuine Poisson
#       integrator of the second structure.
#
# The chart machinery comes from the package -- `miura_map`, `miura_invert`, `hill_lambda0`,
# `miura_lambda` and `kdv_miura_bracket` are what `test/miura.jl` exercises.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random
using SymPyPythonCall

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary
include(joinpath(@__DIR__, "kdvtools.jl"));
using .KdVTools

const TOL = 1e-9

# --------------------------------------------------------------------------
header("1. the Miura factorisation, symbolically")

let NJ = 8
    vj = [Sym("v$i") for i in 0:(NJ - 1)]
    wj = [Sym("w$i") for i in 0:(NJ - 1)]
    FAM = (vj, wj)
    function D(f, n = 1)
        for _ in 1:n
            f = expand(sum(fam[k + 2] * diff(f, fam[k + 1]) for fam in FAM
            for k in 0:(NJ - 2)))
        end
        f
    end
    v(k) = vj[k + 1]
    w(k) = wj[k + 1]

    u = v(0)^2 + v(1)                     # u = v^2 + v_x
    ux = D(u)
    # M' d_x M'^dagger applied to w:  (2v + d_x) d_x (2v - d_x) w
    inner = D(2 * v(0) * w(0) - D(w(0)))
    lhs = expand(2 * v(0) * inner + D(inner))
    rhs = expand(4 * u * D(w(0)) + 2 * ux * w(0) - D(w(0), 3))
    check("(2v + d_x) d_x (2v - d_x) = 4u d_x + 2u_x - d_x^3  for  u = v^2 + v_x",
        simplify(lhs - rhs) == 0, "difference $(simplify(lhs - rhs))")

    # the induced flow in v is mKdV: dH~/dv = M'^dagger dH2/du = (2v - d_x) u, v_t = d_x of that
    dHdv = expand(2 * v(0) * u - D(u))
    vt = expand(D(dHdv))
    check(
        "v_t = d_x ( 2vu - u_x ) = 6 v^2 v_x - v_xxx, i.e. mKdV under the constant structure",
        simplify(vt - (6 * v(0)^2 * v(1) - v(3))) == 0, "got $vt")
end

# --------------------------------------------------------------------------
# discrete assembly
# --------------------------------------------------------------------------

"P^2_Miura and its derivative with respect to the u-degrees of freedom."
function miura_bracket_data(n; mean = 1.0, p = 3)
    s = SplineSpace(n, p)
    M = Matrix(mass_matrix(s))
    Minv = inv(M)
    W = quadrature_weights(s)
    Φ0 = basis_values(s, 0)
    S = gram(s, 0, 1)
    P1 = Minv * ((S - transpose(S)) / 2) * Minv        # constant, exactly Poisson
    nb = nbasis(s)
    T = [sum(Φ0[j, q] * Φ0[k, q] * Φ0[l, q] * W[q] for q in eachindex(W))
         for j in 1:nb, k in 1:nb, l in 1:nb]          # ∫ φ_j φ_k φ_l
    v̂ = project(s, x -> mean + sin(x) + 0.3 * cos(2x))
    vh = fieldat(s, v̂, 0)
    # (vh .* Φ0') .* W is already (nq × nb); Φ0 is (nb × nq)
    B = Φ0 * ((vh .* transpose(Φ0)) .* W)              # ∫ v_h φ_k φ_l
    DM = -(Minv * (2B + S))                            # du/dv
    Pm = DM * P1 * transpose(DM)
    # dP/dv_j, then the chain rule to dP/du_m
    dDdv = [-2 * sum(Minv[a, b] * T[j, b, c] for b in 1:nb)
            for j in 1:nb, a in 1:nb, c in 1:nb]
    dPdv = [sum(dDdv[j, a, c] * P1[c, d] * DM[b, d] + DM[a, c] * P1[c, d] * dDdv[j, b, d]
            for c in 1:nb, d in 1:nb) for j in 1:nb, a in 1:nb, b in 1:nb]
    DMinv = inv(DM)
    dPdu = [sum(DMinv[j, m] * dPdv[j, a, b] for j in 1:nb)
            for m in 1:nb, a in 1:nb, b in 1:nb]
    û = -(Minv * (Φ0 * (W .* (vh .^ 2 .+ fieldat(s, v̂, 1)))))
    return (; s, M, Minv, v̂, û, DM, Pm, dPdu, B, S, W, Φ0)
end

header("2. where the discrete Miura map is a local diffeomorphism")
println("      ker(2v + d_x) is spanned by exp(-2 int v), periodic iff int v = 0:")
@printf("      %8s %12s\n", "int v", "cond(DM)")
conds = Dict{Float64, Float64}()
for mean in (0.0, 0.25, 1.0, 2.0)
    r = miura_bracket_data(24; mean)
    conds[mean] = cond(r.DM)
    @printf("      %8.2f %12.2e\n", mean * 2π, conds[mean])
end
check("DM is near-singular exactly at int v = 0, and well conditioned away from it",
    conds[0.0] > 1e6 && maximum(conds[m] for m in (0.25, 1.0, 2.0)) < 1e4,
    @sprintf("cond at int v = 0 is %.1e, elsewhere <= %.1e", conds[0.0],
        maximum(conds[m] for m in (0.25, 1.0, 2.0))))

header("3. P^2_Miura is exactly Poisson")
@printf("      %4s %10s %10s %19s\n", "N", "cond(DM)", "antisym", "normalised Jacobi")
let resid = Float64[]
    for n in (12, 16, 24, 32)
        r = miura_bracket_data(n)
        a = maximum(abs, r.Pm + transpose(r.Pm)) / maximum(abs, r.Pm)
        res, sc = jacobi_residual(r.Pm, r.dPdu; normalised = false)
        push!(resid, res / sc)
        @printf("      %4d %10.2e %10.1e %19.2e\n", n, cond(r.DM), a, res / sc)
        check("N = $n: antisymmetric", a < TOL, @sprintf("%.1e", a))
    end
    check(
        "the Jacobi residual is at machine precision at every N, against 0.42 -- flat -- " *
        "for the Galerkin P^2",
        maximum(resid) < 1e-11,
        @sprintf("worst %.1e", maximum(resid)))
end

header("4. the closed form written out in the notes")
let n = 24, r = miura_bracket_data(n)
    Minv, B, S, M = r.Minv, r.B, r.S, r.M
    check("S is antisymmetric, so P^1 = Minv S Minv (the 1/2 and the S - S^T cancel)",
        maximum(abs, (S - transpose(S)) / 2 - S) / maximum(abs, S) < TOL)
    check("B is symmetric", maximum(abs, B - transpose(B)) / maximum(abs, B) < TOL)
    Lop = -(Minv * (2B + S))
    check("DM_h = -Minv (2B + S)", maximum(abs, Lop - r.DM) / maximum(abs, Lop) < TOL)
    closed = Minv * (2B + S) * Minv * S * Minv * (2B - S) * Minv
    e = maximum(abs, closed - r.Pm) / maximum(abs, r.Pm)
    check("P^2_M = Minv (2B + S) Minv S Minv (2B - S) Minv, as written in the notes",
        e < TOL, @sprintf("rel. error %.2e", e))
    check("antisymmetry follows algebraically from B^T = B and S^T = -S",
        maximum(abs, closed + transpose(closed)) / maximum(abs, closed) < TOL)

    # The two interior inverse mass matrices are what make it nonlocal. On a periodic mesh the
    # matrices are circulant, so bandwidth must be measured with the CIRCULAR distance.
    function band(A)
        m = size(A, 1)
        thr = 1e-10 * maximum(abs, A)
        maximum(min(abs(i - j), m - abs(i - j))
        for i in 1:m, j in 1:m if abs(A[i, j]) > thr)
    end
    Ku, K0 = K2_blocks(r.s)
    ud = Minv * (r.Φ0 * (r.W .* (fieldat(r.s, r.v̂, 0) .^ 2 .+ fieldat(r.s, r.v̂, 1))))
    Kgal = K0 + sum(ud[m] .* Ku[m, :, :] for m in axes(Ku, 1))     # Galerkin assembly
    KM = (2B + S) * Minv * S * Minv * (2B - S)                     # the Miura one
    @printf("      circular bandwidth, Galerkin K:  %3d   (local assembly, p = 3)\n",
        band(Kgal))
    @printf("      circular bandwidth, Miura K_M:   %3d   (n = %d, so n/2 = dense)\n",
        band(KM), n)
    check(
        "K_M is dense while the Galerkin assembly is local: the two interior Minv are " *
        "the nonlocality that carries the Jacobi identity",
        band(KM) ≥ n ÷ 2 && band(Kgal) ≤ 4, "$(band(Kgal)) vs $(band(KM))")
end

header("5. consistency: P^2_Miura approximates the same operator")
println("      applied to the gradient of a fixed smooth functional, the two")
println("      discretisations of D2 agree, and converge to one another:")
@printf("      %4s %16s %7s\n", "N", "rel. difference", "order")
let prev = nothing, prev_n = 0, orders = Float64[]
    for n in (12, 16, 24, 32, 48)
        r = miura_bracket_data(n)
        Pgal = poisson_matrix(kdv_bracket_2(r.s), r.û)
        X = quadrature_nodes(r.s)
        g = r.Φ0 * (r.W .* U0.(X))
        d = maximum(abs, (r.Pm - Pgal) * g) / maximum(abs, Pgal * g)
        prev === nothing || push!(orders, log(prev / d) / log(n / prev_n))
        @printf("      %4d %16.4e%s\n", n, d,
            isempty(orders) ? "" : @sprintf(" %7.1f", orders[end]))
        prev, prev_n = d, n
    end
    check("on smooth data the two discretisations converge to one another",
        minimum(orders) > 0.8,
        "observed orders [" *
        join(["'" * (@sprintf("%.1f", o)) * "'" for o in orders], ", ") * "]")
end

println("\n      entrywise, however, they do NOT converge -- the difference lives in")
println("      the under-resolved modes, where both are O(h^-3) and differ at O(1):")
@printf("      %4s %16s\n", "N", "rel. difference")
let ent = Float64[]
    for n in (12, 24, 48)
        r = miura_bracket_data(n)
        Pgal = poisson_matrix(kdv_bracket_2(r.s), r.û)
        e = maximum(abs, r.Pm - Pgal) / maximum(abs, Pgal)
        push!(ent, e)
        @printf("      %4d %16.4e\n", n, e)
    end
    check(
        "the entrywise difference stays of order one -- the same signature as the " *
        "Jacobi violation, and in the same modes",
        minimum(ent) > 0.1,
        "[" * join(["'" * (@sprintf("%.2f", e)) * "'" for e in ent], ", ") * "]")
end

println("\n      the induced flow reproduces 6uu_x - u_xxx:")
@printf("      %4s %14s %7s\n", "N", "rel. error", "order")
let prev = nothing, prev_n = 0, orders = Float64[], efin = 0.0
    for n in (12, 16, 24, 32, 48)
        r = miura_bracket_data(n)
        udot = r.Pm * (r.M * r.û)                       # dH2/du = M u
        uh, uhx, uhxxx = fieldat(r.s, r.û, 0), fieldat(r.s, r.û, 1), fieldat(r.s, r.û, 3)
        exact = r.M \ (r.Φ0 * (r.W .* (-6 .* uh .* uhx .- uhxxx)))
        e = maximum(abs, udot - exact) / maximum(abs, exact)
        prev === nothing || push!(orders, log(prev / e) / log(n / prev_n))
        @printf("      %4d %14.4e%s\n", n, e,
            isempty(orders) ? "" : @sprintf(" %7.1f", orders[end]))
        prev, prev_n, efin = e, n, e
    end
    check("the semi-discrete flow converges rapidly to the KdV right-hand side",
        minimum(orders) > 0.8 && efin < 1e-6,
        "observed orders [" *
        join(["'" * (@sprintf("%.1f", o)) * "'" for o in orders], ", ") * "]")
end

# --------------------------------------------------------------------------
# the v chart: assemblies, the pushed-forward Hamiltonian, and the flow
# --------------------------------------------------------------------------

"Assembly plus the Miura flow, its Jacobian, and the invariants."
struct Chart
    s::Any
    n::Any
    X::Any
    W::Any
    M::Any
    Minv::Any
    Sraw::Any
    S::Any
    P1::Any
    g::Any
    v̂::Any
end

function Chart(n; p = 3, nq = nothing, uniform = true, graded = false, v0 = nothing)
    s, _ = setup(n; p, nq, uniform, graded, f = U0)
    M = Matrix(mass_matrix(s))
    Minv = inv(M)
    Sraw = gram(s, 0, 1)
    S = (Sraw - transpose(Sraw)) / 2
    Chart(s, nbasis(s), quadrature_nodes(s), quadrature_weights(s), M, Minv, Sraw, S,
        Minv * S * Minv, mass_grad(s), v0 === nothing ? nothing : project(s, v0))
end

function Bmat(c::Chart, v̂)
    basis_values(c.s, 0) *
    ((fieldat(c.s, v̂, 0) .* transpose(basis_values(c.s, 0))) .* c.W)
end
Lop(c::Chart, v̂) = -(c.Minv * (2 .* Bmat(c, v̂) + c.S))
function Mmap(c::Chart, v̂)
    -(c.Minv * (basis_values(c.s, 0) *
       (c.W .* (fieldat(c.s, v̂, 0) .^ 2 .+ fieldat(c.s, v̂, 1)))))
end
P2M(c::Chart, v̂) = (Lm = Lop(c, v̂); Lm * c.P1 * transpose(Lm))

cH2(c::Chart, û) = dot(c.W, fieldat(c.s, û, 0) .^ 2) / 2
cH1(c::Chart, û) = dot(c.W, fieldat(c.s, û, 1) .^ 2 - 2 .* fieldat(c.s, û, 0) .^ 3) / 2
cC0(c::Chart, û) = dot(c.W, fieldat(c.s, û, 0))
Ht(c::Chart, v̂) = cH2(c, Mmap(c, v̂))
gHt(c::Chart, v̂) = -((2 .* Bmat(c, v̂) - c.S) * Mmap(c, v̂))
function hess_Ht(c::Chart, v̂)
    Lm = Lop(c, v̂)
    uh = fieldat(c.s, Mmap(c, v̂), 0)
    transpose(Lm) * c.M * Lm -
    2 .* (basis_values(c.s, 0) * ((uh .* transpose(basis_values(c.s, 0))) .* c.W))
end
cf(c::Chart, v̂) = c.P1 * gHt(c, v̂)
cDf(c::Chart, v̂) = c.P1 * hess_Ht(c, v̂)

"""
    three_field(c, v̂; skew)

vdot from the three sparse solves of eq. (semidiscrete-miura). With `skew` the two derivative
pairings are skew-symmetrised, as everywhere else in the notes, and the result is the bracket
flow for ANY quadrature; without it the two are the same operator only when the quadrature
integrates d_x(phi_k phi_l) exactly.
"""
function three_field(c::Chart, v̂; skew = true)
    Φ0 = basis_values(c.s, 0)
    Sop = skew ? c.S : c.Sraw
    vh = fieldat(c.s, v̂, 0)
    û = -(c.M \ (Φ0 * (c.W .* (vh .^ 2 .+ fieldat(c.s, v̂, 1)))))
    uh = fieldat(c.s, û, 0)
    ẑ = -(c.M \ (2 .* (Φ0 * (c.W .* vh .* uh)) - Sop * û))
    return c.M \ (Sop * ẑ), û, ẑ
end

"Implicit midpoint in the v chart; Dy is the exact Cayley transform."
function midpoint_v(c::Chart, v̂, dt)
    Id = Matrix{Float64}(I, c.n, c.n)
    y = newton(w -> w - v̂ - dt .* cf(c, 0.5 .* (v̂ + w)),
        w -> Id - 0.5dt .* cDf(c, 0.5 .* (v̂ + w)), v̂; itmax = 60)
    A = cDf(c, 0.5 .* (v̂ + y))
    y, (Id - 0.5dt .* A) \ (Id + 0.5dt .* A)
end

"Gonzalez midpoint discrete gradient of Htilde."
function dgrad_Ht(c::Chart, x, y)
    xbar, dx = 0.5 .* (x + y), y - x
    g = gHt(c, xbar)
    nn = dot(dx, dx)
    nn < 1e-30 && return g
    g + ((Ht(c, y) - Ht(c, x)) - dot(g, dx)) / nn .* dx
end

"Newton with a finite-difference Jacobian; the systems here are small."
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

function dgrad_v(c::Chart, v̂, dt)
    (fd_newton(w -> w - v̂ - dt .* (c.P1 * dgrad_Ht(c, v̂, w)), v̂, c.n), nothing)
end

# --------------------------------------------------------------------------
header("6. where the Miura chart lives")

# Taking the trace of u = Pi(v^2 + v_x) against the partition of unity: 1 is in V_h, so
# int Pi f = int f, and int v_h,x = 0. Hence C_0,d = int u_h = -int v_h^2, an EXACT identity.
let c = Chart(20; p = 3, v0 = x -> 1.0 + sin(x))
    û = Mmap(c, c.v̂)
    lhs, rhs = cC0(c, û), -dot(c.W, fieldat(c.s, c.v̂, 0) .^ 2)
    check(
        "C_0,d = int u_h = -int v_h^2, exactly -- so the image of M_h lies in " *
        "C_0,d <= 0",
        abs(lhs - rhs) / abs(lhs) < 1e-14,
        @sprintf("%.15f vs %.15f", lhs, rhs))
end

# The sharp condition is spectral: v = psi'/psi turns u = v^2 + v_x into psi'' = u psi, and a
# real periodic v exists iff that has a nodeless solution, i.e. iff 0 lies below the spectrum
# of the Hill operator -d_x^2 + u_h.
println("\n      u = c + sin x + 0.4 cos 2x, N = 20, p = 3:")
@printf("      %5s %9s %12s %26s\n", "c", "C_0,d", "lam_0(Hill)", "preimages, int v")
const s20 = SplineSpace(20, 3; nq = 6)
const ubase(cc) = project(s20, x -> cc + sin(x) + 0.4 * cos(2x))
let found = Dict{Float64, Vector{Float64}}()
    for cc in (0.0, 0.2, 0.4, 0.5, 1.0, 2.0)
        u = ubase(cc)
        br = sort(unique(round(dot(quadrature_weights(s20), fieldat(s20, v, 0)); digits = 3)
        for v in (miura_invert(s20, u; seed = sd)
            for sd in range(-4.0, 4.0; length = 17)) if v !== nothing))
        found[cc] = br
        λ = hill_lambda0(s20, u)
        @printf("      %5.1f %9.3f %12.4f %26s\n", cc,
            dot(quadrature_weights(s20), fieldat(s20, u, 0)), λ,
            isempty(br) ? "none" : string(br))
    end
    check("a preimage exists exactly when the discrete Hill operator is positive",
        all(!isempty(found[cc]) == (hill_lambda0(s20, ubase(cc)) > 0)
        for cc in keys(found)),
        string(Dict(k => !isempty(v) for (k, v) in found)))
    check(
        "and where it exists there are two, the two Floquet solutions, with " *
        "opposite int v -- so M_h is not injective either",
        all(length(v) == 2 && abs(v[1] + v[2]) < 1e-6
        for v in values(found) if !isempty(v)),
        string(found))
end

# the threshold predicted by the Hill eigenvalue, against a bisection
let base(x) = sin(x) + 0.4 * cos(2x)
    λ0 = hill_lambda0(s20, project(s20, base))
    # λ0(c) = λ0(0) - c exactly, a constant in the potential shifting every eigenvalue, so the
    # ADMISSIBLE side is c below the threshold and the bisection brackets it from below.
    lo, hi = -1.0, 0.0
    for _ in 1:40
        mid = 0.5 * (lo + hi)
        u = project(s20, x -> mid + base(x))
        if any(miura_invert(s20, u; seed = sd) !== nothing
        for sd in range(-4.0, 4.0; length = 17))
            lo = mid
        else
            hi = mid
        end
    end
    # At the threshold psi acquires a zero and v = psi'/psi is unbounded, so it is the Newton
    # iteration and not the criterion that limits the measurement.
    check(@sprintf("the existence threshold is lam_0 = %.6f, measured %.6f", λ0, lo),
        abs(lo - λ0) < 1e-3, @sprintf("predicted %.6f, measured %.6f", λ0, lo))
end

println("\n      the examples of Sections 8 and 9 are outside the image:")
const EXAMPLES = (("u0 = cos x", cos, 2π, 20),
    ("u0 = sin x + 0.4 cos 2x", U0, 2π, 20),
    ("single soliton, [0, 40)",
        x -> 2.0 / cosh(clamp(x - 20.0, -350, 350))^2, 40.0, 64))
for (lab, f, Lx, n) in EXAMPLES
    se = SplineSpace(n, 3; L = Lx, nq = 6)
    u = project(se, f)
    ok = any(miura_invert(se, u; seed = sd) !== nothing
    for sd in range(-4.0, 4.0; length = 17))
    @printf("      %-26s C_0,d = %8.3f   preimage: %s\n", lab,
        dot(quadrature_weights(se), fieldat(se, u, 0)), ok ? "yes" : "NO")
    check("$lab has no Miura preimage, its mass being non-negative", !ok)
end
println("      so the Miura runs of Section 8 CANNOT be seeded in u at lam = 0.")

# --------------------------------------------------------------------------
header("6b. the spectral parameter puts all of them back in the chart")

# What lifts the obstruction is the spectral parameter: u = -(v^2 + v_x) - lam turns the
# Riccati substitution into (-d_x^2 - u) psi = lam psi, so lam is an eigenvalue parameter of
# the very Hill operator that decides invertibility, and a preimage exists exactly when
# lam < lam_0. That can always be arranged.
println("\n      the same examples, at lam = miura_lambda(u):")
@printf("      %-26s %9s %9s %9s %9s %11s\n",
    "example", "C_0,d", "lam_0", "lam", "preimage", "round trip")
for (lab, f, Lx, n) in EXAMPLES
    se = SplineSpace(n, 3; L = Lx, nq = 6)
    u = project(se, f)
    λ0 = hill_lambda0(se, u)
    λ = miura_lambda(se, u)
    v = miura_invert(se, u; λ)
    rt = v === nothing ? NaN :
         maximum(abs, miura_map(se, v, λ) - u) / maximum(abs, u)
    @printf("      %-26s %9.3f %9.4f %9.4f %9s %11.1e\n", lab,
        dot(quadrature_weights(se), fieldat(se, u, 0)), λ0, λ,
        v !== nothing ? "yes" : "NO", rt)
    check("$lab DOES have a preimage once lam < lam_0", v !== nothing)
    check("$lab: and it is one, to round-off", rt < 1e-12, @sprintf("%.1e", rt))
end

let u = project(s20, U0), λ0 = hill_lambda0(s20, project(s20, U0))
    seeds = range(-4.0, 4.0; length = 9)
    found(λ) = any(miura_invert(s20, u; λ, seed = sd) !== nothing for sd in seeds)
    check("a preimage exists for lam comfortably BELOW lam_0",
        all(found(λ0 - d) for d in (0.05, 0.5, 5.0)))
    check("and none for lam comfortably above it",
        !any(found(λ0 + d) for d in (0.05, 0.5, 5.0)))

    # L does not see a constant, so the bracket is untouched; the pushforward is the PENCIL
    # member P2 - 4 lam P1, and it is exactly Poisson for every lam
    for λ in (-0.5, -2.0, -5.0)
        v = miura_invert(s20, u; λ)
        Pm = poisson_matrix(kdv_miura_bracket(s20, v), u)
        anti = maximum(abs, Pm + transpose(Pm)) / maximum(abs, Pm)
        check(@sprintf("lam = %+.1f: the pushforward is still exactly antisymmetric", λ),
            anti < 1e-13, @sprintf("%.1e", anti))
    end
end
println("      L P1 L^T is NOT the Galerkin P2 - 4 lam P1: those differ by exactly the")
println("      Jacobi defect the Miura construction exists to remove.")

# --------------------------------------------------------------------------
header("7. the semi-discrete Miura equations of Section 7")

let c = Chart(20; p = 3, v0 = x -> 1.0 + sin(x))
    v̂ = c.v̂
    û = Mmap(c, v̂)
    fv, û2, ẑ = three_field(c, v̂)
    check("the first solve reproduces uhat = M_h(vhat)",
        maximum(abs, û2 - û) / maximum(abs, û) < TOL)
    # z_h is the Galerkin approximation of M'^dagger u = -(2vu - u_x), which with
    # u = -(v^2 + v_x) is 2v^3 - v_xx again: the two sign flips cancel.
    vh, vxx = fieldat(c.s, v̂, 0), fieldat(c.s, v̂, 2)
    zref = c.M \ (basis_values(c.s, 0) * (c.W .* (2.0 .* vh .^ 3 .- vxx)))
    e = maximum(abs, ẑ - zref) / maximum(abs, zref)
    check(
        "the second solve is z_h = Pi( 2 v_h u_h - u_h,x ) = Pi( 2 v^3 - v_xx ) " *
        "to the discretisation error",
        e < 1e-3,
        @sprintf("rel. difference %.2e", e))
    check("the three-field system IS the bracket flow, vdot = P^1 dHtilde/dvhat",
        maximum(abs, fv - cf(c, v̂)) / maximum(abs, cf(c, v̂)) < TOL,
        @sprintf("%.2e", maximum(abs, fv - cf(c, v̂)) / maximum(abs, cf(c, v̂))))
    check("and its pushforward is udot = P^2_Miura dH_2,d/duhat",
        maximum(abs, Lop(c, v̂) * fv - P2M(c, v̂) * (c.M * û)) /
        maximum(abs, P2M(c, v̂) * (c.M * û)) < TOL)
end

println("\n      the two derivative pairings must be exact transposes; skew-")
println("      symmetrising them, as everywhere else in the notes, makes that")
println("      true at ANY quadrature:")
@printf("      %4s %12s %12s\n", "nq", "skew form", "plain S")
let plain = Dict{Int, Float64}()
    for nq in 2:6
        cn = Chart(16; p = 3, nq, uniform = false, v0 = x -> 1.0 + sin(x))
        ref = cf(cn, cn.v̂)
        a = maximum(abs, three_field(cn, cn.v̂; skew = true)[1] - ref) / maximum(abs, ref)
        b = maximum(abs, three_field(cn, cn.v̂; skew = false)[1] - ref) / maximum(abs, ref)
        plain[nq] = b
        @printf("      %4d %12.1e %12.1e\n", nq, a, b)
        check("nq = $nq: the skew form is the bracket flow exactly", a < 1e-12,
            @sprintf("%.1e", a))
    end
    check(
        "while the plain form is a different system until the quadrature " *
        "integrates d_x(phi_k phi_l) exactly",
        plain[2] > 1e-4 && plain[3] < 1e-12,
        @sprintf("nq = 2: %.1e, nq = 3: %.1e", plain[2], plain[3]))
end

for p in (1, 2, 3)
    cp = Chart(24; p, nq = 8, v0 = x -> 1.0 + sin(x))
    fv, _, _ = three_field(cp, cp.v̂)
    check("p = $p: the three-field system is well defined and equals the bracket flow",
        maximum(abs, fv - cf(cp, cp.v̂)) / maximum(abs, cf(cp, cp.v̂)) < TOL)
end
println("      -- only first derivatives of the basis enter, against p >= 2 for")
println("         the plain Galerkin scheme of eq. (semidiscrete-kdv-galerkin).")

# --------------------------------------------------------------------------
header("8. what the semi-discrete Miura flow conserves")

for (lab, kw) in (("uniform    ", (; uniform = true)),
    ("graded     ", (; uniform = false, graded = true)),
    ("non-uniform", (; uniform = false)))
    cm = Chart(16; p = 3, v0 = x -> 1.0 + sin(x), kw...)
    f = cf(cm, cm.v̂)
    dHt = dot(gHt(cm, cm.v̂), f)
    dv = dot(cm.g, f)
    check("$lab: Htilde = H_2,d is conserved exactly, by antisymmetry alone",
        abs(dHt) / (maximum(abs, gHt(cm, cm.v̂)) * maximum(abs, f) * cm.n) < 1e-13,
        @sprintf("%.1e", dHt))
    check("$lab: int v_h is conserved exactly, being a Casimir of P^1", abs(dv) < 1e-12,
        @sprintf("%.1e", dv))
end

println("\n      the KdV mass is a different matter.  C_0,d = int v_h^2 is the")
println("      mKdV momentum, and it is conserved only in the sense that the")
println("      respective other Hamiltonian is -- on the resolved modes:")
@printf("      %4s %22s %12s\n", "N", "|dC_0,d/dt|, resolved", "broadband")
local blast = 0.0
for n in (12, 16, 24, 32)
    cm = Chart(n; p = 3, v0 = x -> 1.0 + sin(x) + 0.3 * cos(2x))
    a = abs(dot(cm.g, Lop(cm, cm.v̂) * cf(cm, cm.v̂)))
    vb = 1.0 .+ 0.3 .* randn(MersenneTwister(3), n)
    b = abs(dot(cm.g, Lop(cm, vb) * cf(cm, vb)))
    global blast = b
    @printf("      %4d %22.2e %12.2e\n", n, a, b)
end
check(
    "dC_0,d/dt does not vanish identically: it is order one on a broadband " *
    "field, where every other conservation law of the notes is exact",
    blast > 1e-3, @sprintf("%.2e", blast))
for (lab, kw) in (("graded     ", (; uniform = false, graded = true)),
    ("non-uniform", (; uniform = false)))
    cm = Chart(16; p = 3, v0 = x -> 1.0 + sin(x), kw...)
    d = abs(dot(cm.g, Lop(cm, cm.v̂) * cf(cm, cm.v̂)))
    check("$lab: and it does not vanish for a resolved field either", d > 1e-9,
        @sprintf("%.1e", d))
end
println("      -- the one conservation law the Miura bracket gives up: for both")
println("         Galerkin brackets the mass is free to every method, by")
println("         eq. (mass-casimir).")

# --------------------------------------------------------------------------
header("9. time discretisation: the two charts are not the same method")

# Runge-Kutta methods are equivariant under AFFINE changes of variables and under no others.
# The positive control comes first: without it the negative result below would be
# indistinguishable from a bug.
const cc9 = Chart(20; p = 3, v0 = x -> 1.0 + sin(x))
const vd0 = cc9.v̂
const ud0 = Mmap(cc9, vd0)
let rng = MersenneTwister(0), DT = 8e-3
    T = Matrix{Float64}(I, cc9.n, cc9.n) + 0.1 .* randn(rng, cc9.n, cc9.n)
    sh = 0.05 .* randn(rng, cc9.n)
    Ti = inv(T)
    # the same flow in the coordinates w = T v + shift
    af = deepcopy(cc9)
    fA(w) = T * cf(af, Ti * (w - sh))
    DfA(w) = T * cDf(af, Ti * (w - sh)) * Ti
    Id = Matrix{Float64}(I, cc9.n, cc9.n)
    w1 = newton(y -> y - (T * vd0 + sh) - DT .* fA(0.5 .* ((T * vd0 + sh) + y)),
        y -> Id - 0.5DT .* DfA(0.5 .* ((T * vd0 + sh) + y)), T * vd0 + sh; itmax = 60)
    v1, _ = midpoint_v(cc9, vd0, DT)
    e = maximum(abs, T * v1 + sh - w1) / maximum(abs, w1)
    check(
        "positive control: a midpoint step commutes with an AFFINE change of " *
        "variables, to round-off",
        e < 1e-13,
        @sprintf("%.1e", e))
end

"The Miura flow in the u chart; needs M_h^{-1}, hence the seed."
function fu(c::Chart, û, seed)
    v = miura_invert(c.s, û; seed)
    Lm = Lop(c, v)
    Lm * c.P1 * transpose(Lm) * (c.M * û), v
end

function midpoint_u(c::Chart, û, dt, seed)
    fd_newton(y -> y - û - dt .* first(fu(c, 0.5 .* (û + y), seed)), û, c.n; itmax = 40)
end

const seed9 = sum(vd0) / length(vd0)
println("\n      one step of implicit midpoint, in v and pushed forward, against")
println("      one step of implicit midpoint in u:")
@printf("      %9s %13s %7s\n", "dt", "difference", "rate")
let prev = nothing, rs = Float64[]
    for dt in (1.6e-2, 8e-3, 4e-3, 2e-3, 1e-3)
        yv, _ = midpoint_v(cc9, vd0, dt)
        d = maximum(abs, Mmap(cc9, yv) - midpoint_u(cc9, ud0, dt, seed9))
        prev === nothing || push!(rs, log(prev / d) / log(2.0))
        @printf("      %9.1e %13.4e%s\n", dt, d,
            isempty(rs) ? "" : @sprintf(" %7.2f", rs[end]))
        prev = d
    end
    check(
        "the two differ at O(dt^3) per step: the Miura map is quadratic, and " *
        "only affine maps commute with a Runge-Kutta method",
        2.6 < sum(rs) / length(rs) < 3.4,
        "observed rates [" * join(["'" * (@sprintf("%.2f", r)) * "'" for r in rs], ", ") *
        "]")
end

println("\n      what does transport: the Poisson-map property")
@printf("      %-28s%14s%16s\n", "method", "in v, vs P^1", "in u, vs P^2_M")
let DT = 8e-3, pois = Dict{String, NTuple{2, Float64}}()
    for (name, stepper) in (("implicit midpoint", midpoint_v), (
        "discrete gradient", dgrad_v))
        y, Dy = stepper(cc9, vd0, DT)
        if Dy === nothing                       # central difference, as elsewhere
            Dy = zeros(cc9.n, cc9.n)
            eps = 1e-6
            for j in 1:cc9.n
                a, b = copy(vd0), copy(vd0)
                a[j] += eps
                b[j] -= eps
                Dy[:, j] = (stepper(cc9, a, DT)[1] - stepper(cc9, b, DT)[1]) ./ (2eps)
            end
        end
        ev = maximum(abs, Dy * cc9.P1 * transpose(Dy) - cc9.P1) / maximum(abs, cc9.P1)
        # the same map in u: Phi_u = M_h ∘ Phi_v ∘ M_h^{-1}, so D Phi_u = L(y) Dy L(v)^-1
        Du = Lop(cc9, y) * Dy * inv(Lop(cc9, vd0))
        Pin, Pout = P2M(cc9, vd0), P2M(cc9, y)
        eu = maximum(abs, Du * Pin * transpose(Du) - Pout) / maximum(abs, Pout)
        pois[name] = (ev, eu)
        @printf("      %-28s%14.1e%16.1e\n", name, ev, eu)
        check("$name: the residual is the same in both charts",
            abs(log10(max(ev, 1e-16) / max(eu, 1e-16))) < 1.0,
            @sprintf("%.1e vs %.1e", ev, eu))
    end
    check(
        "implicit midpoint in v pushes forward to a genuine Poisson integrator " *
        "of the Miura bracket -- the first one the notes have for the second " *
        "KdV structure",
        pois["implicit midpoint"][2] < 1e-11,
        @sprintf("%.1e", pois["implicit midpoint"][2]))
    check("the discrete-gradient method is not, here as on the first flow",
        pois["discrete gradient"][2] > 1e-6,
        @sprintf("%.1e", pois["discrete gradient"][2]))
end

println("\n      the drift table of Section 8: N = 20, p = 3, dt = 2e-3, 500")
println("      steps, from v0 = 1 + sin x, so u0 = M_h(v0):")
const DT9, NT9 = 2e-3, 500
const ref9 = [cH1(cc9, ud0), cH2(cc9, ud0), cC0(cc9, ud0)]
invs(u) = [cH1(cc9, u), cH2(cc9, u), cC0(cc9, u)]

let (Ku, K0) = K2_blocks(cc9.s)
    K2f(d) = K0 + sum(d[m] .* Ku[m, :, :] for m in axes(Ku, 1))
    ff1(d) = cc9.P1 * grad_H1(cc9.s, d)
    Df1(d) = cc9.P1 * hess_H1(cc9.s, d)
    ff2(d) = cc9.Minv * (K2f(d) * d)
    Df2(d) = cc9.Minv * (K2f(d) + [sum(Ku[j, i, l] * d[l] for l in eachindex(d))
               for i in eachindex(d), j in eachindex(d)])
    Iu = Matrix{Float64}(I, cc9.n, cc9.n)
    mid_u(fun, dfun, d, dt) = newton(y -> y - d - dt .* fun(0.5 .* (d + y)),
        y -> Iu - 0.5dt .* dfun(0.5 .* (d + y)), d; itmax = 60)
    function dgrad_u_flow(d, dt)
        function gbar(y)
            xb, dx = 0.5 .* (d + y), y - d
            g = grad_H1(cc9.s, xb)
            nn = dot(dx, dx)
            nn < 1e-30 ? g : g + ((cH1(cc9, y) - cH1(cc9, d)) - dot(g, dx)) / nn .* dx
        end
        fd_newton(y -> y - d - dt .* (cc9.P1 * gbar(y)), d, cc9.n)
    end

    @printf("      %-28s%10s%10s%10s\n", "method", "|dH1|", "|dH2|", "|dC0|")
    got = Dict{String, NTuple{3, Float64}}()
    for (name, stepper, chart) in ((
        "midpoint, flow 1", (d, dt) -> mid_u(ff1, Df1, d, dt), "u"),
        ("midpoint, flow 2", (d, dt) -> mid_u(ff2, Df2, d, dt), "u"),
        ("discrete gradient, flow 1", dgrad_u_flow, "u"),
        ("midpoint, Miura", (d, dt) -> midpoint_v(cc9, d, dt)[1], "v"),
        ("discrete gradient, Miura", (d, dt) -> dgrad_v(cc9, d, dt)[1], "v"))
        d = copy(chart == "v" ? vd0 : ud0)
        m = zeros(3)
        for _ in 1:NT9
            d = stepper(d, DT9)
            m = max.(m, abs.(invs(chart == "v" ? Mmap(cc9, d) : d) - ref9))
        end
        got[name] = (m[1] / abs(ref9[1]), m[2] / abs(ref9[2]), m[3])
        @printf("      %-28s%10.1e%10.1e%10.1e\n", name, got[name]...)
    end
    check(
        "both Galerkin brackets keep the mass to round-off on this data, by " *
        "eq. (mass-casimir)",
        maximum(got[n][3] for n in keys(got) if !occursin("Miura", n)) < 1e-11,
        string(Dict(n => @sprintf("%.1e", v[3])
        for (n, v) in got if !occursin("Miura", n))))
    check(
        "implicit midpoint does NOT conserve H_2,d in the v chart, where it is " *
        "quartic",
        got["midpoint, Miura"][2] > 1e-6,
        @sprintf("%.1e", got["midpoint, Miura"][2]))
    check(
        "the discrete gradient does, exactly -- the value of a function is " *
        "chart-free",
        got["discrete gradient, Miura"][2] < 1e-11,
        @sprintf("%.1e", got["discrete gradient, Miura"][2]))
    check(
        "both Miura runs lose the mass, which no other method-flow pair in the " *
        "notes does",
        minimum(got[k][3] for k in keys(got) if occursin("Miura", k)) > 1e-6,
        string(Dict(k => @sprintf("%.1e", v[3]) for (k, v) in got)))
end

# The sharpest form of the comparison: the SAME differential equation, the SAME method, run in
# u instead of v. There H_2,d is quadratic and an exact invariant.
let nt = 125, dts = 1.6e-3
    d, mv = copy(vd0), 0.0
    for _ in 1:nt
        d = midpoint_v(cc9, d, dts)[1]
        mv = max(mv, abs(cH2(cc9, Mmap(cc9, d)) - ref9[2]))
    end
    d, mu = copy(ud0), 0.0
    for _ in 1:nt
        d = midpoint_u(cc9, d, dts, seed9)
        mu = max(mu, abs(cH2(cc9, d) - ref9[2]))
    end
    @printf("\n      implicit midpoint on the Miura flow, t <= %.1f, dt = %.1e:\n",
        nt * dts, dts)
    @printf("      |dH2|/|H2| in the v chart: %.3e\n", mv / ref9[2])
    @printf("      |dH2|/|H2| in the u chart: %.3e\n", mu / ref9[2])
    check(
        "one equation, one method, two charts, two answers -- H_2,d is quadratic " *
        "in uhat and quartic in vhat, and that is the whole of it",
        mu / ref9[2] < 1e-12 < mv / ref9[2],
        @sprintf("v: %.1e, u: %.1e", mv / ref9[2], mu / ref9[2]))
end

println("\n      the same quantity along the same trajectory, integrated in u:")
@printf("      %9s %12s %25s %17s\n", "dt", "IMR in v", "IMR in u (Galerkin P^2)",
    "IMR in v, |dC0|")
let (Ku, K0) = K2_blocks(cc9.s)
    ff2(d) = cc9.Minv * ((K0 + sum(d[m] .* Ku[m, :, :] for m in axes(Ku, 1))) * d)
    Df2(d) = cc9.Minv * ((K0 + sum(d[m] .* Ku[m, :, :] for m in axes(Ku, 1))) +
              [sum(Ku[j, i, l] * d[l] for l in eachindex(d))
               for i in eachindex(d), j in eachindex(d)])
    Iu = Matrix{Float64}(I, cc9.n, cc9.n)
    sweep = NTuple{4, Float64}[]
    for dt in (3.2e-3, 1.6e-3, 8e-4, 4e-4, 2e-4)
        nt = round(Int, 1.0 / dt)
        d, mv, mc = copy(vd0), 0.0, 0.0
        for _ in 1:nt
            d = midpoint_v(cc9, d, dt)[1]
            u = Mmap(cc9, d)
            mv = max(mv, abs(cH2(cc9, u) - ref9[2]))
            mc = max(mc, abs(cC0(cc9, u) - ref9[3]))
        end
        d, mu = copy(ud0), 0.0
        for _ in 1:nt
            d = newton(y -> y - d - dt .* ff2(0.5 .* (d + y)),
                y -> Iu - 0.5dt .* Df2(0.5 .* (d + y)), d; itmax = 60)
            mu = max(mu, abs(cH2(cc9, d) - ref9[2]))
        end
        push!(sweep, (dt, mv / ref9[2], mu / ref9[2], mc))
        @printf("      %9.1e %12.4e %25.4e %17.4e\n", dt, mv / ref9[2], mu / ref9[2], mc)
    end
    r = [log(sweep[i][2] / sweep[i + 1][2]) / log(2.0) for i in 1:(length(sweep) - 1)]
    check(
        @sprintf("in v the error in H_2,d is a clean O(dt^2), fitted rate %.2f",
            sum(r) / length(r)),
        1.9 < sum(r) / length(r) < 2.1,
        "rates [" * join(["'" * (@sprintf("%.2f", x)) * "'" for x in r], ", ") * "]")
    check(
        "in u it sits at round-off at every step size, H_2,d being quadratic " *
        "there and an exact invariant",
        maximum(s[3] for s in sweep) < 1e-12,
        @sprintf("worst %.1e", maximum(s[3] for s in sweep)))

    println("\n      the mass, midpoint against the discrete gradient, t <= 0.25:")
    @printf("      %9s %13s %16s %12s\n", "dt", "midpoint", "disc. gradient", "excess")
    exc = Float64[]
    for dt in (3.2e-3, 1.6e-3, 8e-4, 4e-4)
        nt = round(Int, 0.25 / dt)
        row = Float64[]
        for stepper in (midpoint_v, dgrad_v)
            d, m = copy(vd0), 0.0
            for _ in 1:nt
                d = stepper(cc9, d, dt)[1]
                m = max(m, abs(cC0(cc9, Mmap(cc9, d)) - ref9[3]))
            end
            push!(row, m)
        end
        push!(exc, row[2] - row[1])
        @printf("      %9.1e %13.4e %16.4e %12.4e\n", dt, row[1], row[2], exc[end])
    end
    r2 = [log(exc[i] / exc[i + 1]) / log(2.0) for i in 1:(length(exc) - 1)]
    check(
        "the discrete gradient adds a bounded O(dt^2) to the mass where the " *
        @sprintf("midpoint rule adds nothing, fitted rate %.2f", sum(r2) / length(r2)),
        1.8 < sum(r2) / length(r2) < 2.2,
        "rates [" * join(["'" * (@sprintf("%.2f", x)) * "'" for x in r2], ", ") * "]")

    cvar = [s[4] for s in sweep]
    check(
        "the mass error, by contrast, is a dt-independent plateau: the time " *
        "discretisation adds nothing to it, by the increment identity of " *
        "Section 9 applied in the v chart",
        (maximum(cvar) - minimum(cvar)) / minimum(cvar) < 0.4 && minimum(cvar) > 1e-6,
        "[" * join(["'" * (@sprintf("%.3e", x)) * "'" for x in cvar], ", ") * "]")
end

summary("verify_kdv_miura.jl")
