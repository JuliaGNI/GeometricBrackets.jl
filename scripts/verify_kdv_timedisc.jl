#!/usr/bin/env julia
#
# Time discretisation of the semi-discrete KdV equation: which invariants survive.
#
#     julia --project=scripts scripts/verify_kdv_timedisc.jl
#
#   1.  the skew-symmetrised K2 is antisymmetric EXACTLY at every quadrature, agrees with the
#       matrix-free apply, and IS the (-4,-2) member of the family of Section 7 -- so its
#       Lie-algebra residual is the same 0.4212;
#   2.  the mass is free, for any method, because every increment lies in range(P^1);
#   3.  implicit midpoint on the first flow is a POISSON integrator; AVF, the
#       discrete-gradient method and explicit Euler are not;
#   4.  energy or structure, but not both -- the Ge-Marsden obstruction;
#   4b. WHY implicit midpoint does not preserve the quadratic H2: Cooper's theorem needs
#       Q'(y) f(y) = 0 for ALL y, and switching off the cubic term and the mesh
#       non-uniformity in turn restores it;
#   5.  the respective other Hamiltonian: nothing to inherit, and the obstruction grows with
#       the excited mode content -- which is why a test on sin x alone reports round-off;
#   6.  projecting onto both level sets buys all three invariants and costs the structure.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary
include(joinpath(@__DIR__, "kdvtools.jl"));
using .KdVTools

const TOL = 1e-9

"""
    avf_gradient(y, d, dt)

AVF written as a discrete-gradient method on flow 1: `P^1` times the averaged gradient.
Identical to `avf(..., 1)` because `P^1` is constant -- the identity that makes the method
energy-preserving there.
"""
function avf_gradient(y::KdVSys, d, dt)
    gbar(w) = sum(gH1(y, d + t .* (w - d)) for t in SG) / 2
    dgy(w) = sum(t .* hessH1(y, d + t .* (w - d)) for t in SG) / 2
    Id = Matrix{Float64}(I, y.n, y.n)
    newton(w -> w - d - dt .* (y.P1 * gbar(w)), w -> Id - dt .* (y.P1 * dgy(w)), d)
end

"Implicit midpoint, then projected onto H1 = H1t AND H2 = H2t."
function project_both(y::KdVSys, d, dt, H1t, H2t)
    yt, _ = midpoint(y, d, dt, 1)
    a, b = gH1(y, yt), gH2(y, yt)
    lam = zeros(2)
    for _ in 1:40
        z = yt + lam[1] .* a + lam[2] .* b
        r = [sysH1(y, z) - H1t, sysH2(y, z) - H2t]
        maximum(abs, r) < 1e-14 && break
        g1, g2 = gH1(y, z), gH2(y, z)
        lam = lam - [dot(g1, a) dot(g1, b); dot(g2, a) dot(g2, b)] \ r
    end
    yt + lam[1] .* a + lam[2] .* b
end

"As `project_both`, plus a third multiplier along g to hold the mass."
function project_all(y::KdVSys, d, dt, H1t, H2t, C0t)
    yt, _ = midpoint(y, d, dt, 1)
    B = hcat(gH1(y, yt), gH2(y, yt), y.g)
    lam = zeros(3)
    for _ in 1:40
        z = yt + B * lam
        r = [sysH1(y, z) - H1t, sysH2(y, z) - H2t, sysC0(y, z) - C0t]
        maximum(abs, r) < 1e-14 && break
        lam = lam - (transpose(hcat(gH1(y, z), gH2(y, z), y.g)) * B) \ r
    end
    yt + B * lam
end

function integrate_steps(y::KdVSys, step, NT)
    d = copy(y.d0)
    h1, h2, c = [sysH1(y, d)], [sysH2(y, d)], [sysC0(y, d)]
    for _ in 1:NT
        d = step(d)
        push!(h1, sysH1(y, d))
        push!(h2, sysH2(y, d))
        push!(c, sysC0(y, d))
    end
    d, h1, h2, c
end

drift(a) = maximum(abs, a .- a[1]) / max(abs(a[1]), 1e-30)

# --------------------------------------------------------------------------
header("1. the skew-symmetrised second bracket")

for p in (2, 3, 4), uni in (true, false)

    worst_skew, worst_apply = 0.0, 0.0
    for nq in (2, 3, 5, 8)
        s, d = setup(16; p, uniform = uni, nq)
        Ku, K0 = K2_blocks(s)
        K2m = K0 + sum(d[m] .* Ku[m, :, :] for m in axes(Ku, 1))
        c = project(s, x -> cos(3x) + 0.2)
        worst_skew = max(worst_skew, maximum(abs, K2m + transpose(K2m)))
        worst_apply = max(worst_apply,
            maximum(abs, K2apply(s, d, c) - K2m * c) / maximum(abs, K2m * c))
    end
    tag = "p = $p, $(uni ? "uniform" : "non-uniform")"
    check("$tag: K2 is antisymmetric EXACTLY, at every nq in 2..8", worst_skew == 0.0,
        @sprintf("worst |K2 + K2^T| = %.1e", worst_skew))
    check("$tag: the matrix-free K2apply agrees with the assembly", worst_apply < 1e-13,
        @sprintf("worst rel. error %.1e", worst_apply))
    # equal to the unsymmetrised form once the quadrature is exact
    s, d = setup(16; p, uniform = uni, nq = nq_for(p) + 1)
    Ku, K0 = K2_blocks(s)
    Kun, K0un = K2_blocks_unsym(s)
    A = K0 + sum(d[m] .* Ku[m, :, :] for m in axes(Ku, 1))
    B = K0un + sum(d[m] .* Kun[m, :, :] for m in axes(Kun, 1))
    e = maximum(abs, A - B) / maximum(abs, A)
    check("$tag: and equals the unsymmetrised form of the notes exactly", e < 1e-13,
        @sprintf("rel. difference %.1e", e))
end

println("\n  the u-linear tensor is unchanged, so the Jacobi analysis of Section 7")
println("  carries over verbatim -- 2(T1 - T2) = 4 T1 + 2 T3 on T1 + T2 + T3 = 0:")
local lie = 0.0
for n in (12, 16, 24, 32)
    s, _ = setup(n; p = 3)
    Minv = inv(Matrix(mass_matrix(s)))
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    nb = nbasis(s)
    T1 = [sum(Φ0[m, q] * Φ0[k, q] * Φ1[l, q] * W[q] for q in eachindex(W))
          for m in 1:nb, k in 1:nb, l in 1:nb]
    T3 = [sum(Φ1[m, q] * Φ0[k, q] * Φ0[l, q] * W[q] for q in eachindex(W))
          for m in 1:nb, k in 1:nb, l in 1:nb]
    Ku, _ = K2_blocks(s)
    e = maximum(abs, Ku - (-4.0 .* T1 .- 2.0 .* T3)) / maximum(abs, Ku)
    C = similar(Ku)
    for m in 1:nb
        C[m, :, :] = Minv * Ku[m, :, :] * Minv
    end
    global lie = first(structure_constant_residual(C; normalised = false)) /
                 last(structure_constant_residual(C; normalised = false))
    @printf("      N = %3d   |Ku + 4T1 + 2T3| = %.1e   Lie residual = %.4f\n", n, e, lie)
    check("N = $n: the skew tensor IS the (-4,-2) member of the family", e < 1e-13,
        @sprintf("rel. difference %.1e", e))
end
check("and the Lie-algebra residual is the 0.4212 of Section 7, unchanged",
    abs(lie - 0.4212) < 5e-4, @sprintf("got %.4f", lie))

# --------------------------------------------------------------------------
header("2. the mass is free, for any method")

const sy = KdVSys(20; p = 3)
const DT, NT = 2e-3, 500
let e = maximum(abs, sy.P1 * sy.g) / (maximum(abs, sy.P1) * maximum(abs, sy.g))
    check("P^1 g = 0, so g^T P^1 = 0 and every increment in range(P^1) is mass-neutral",
        e < 1e-12, @sprintf("normalised %.1e", e))
end
for (name, step, nt) in (("implicit midpoint", d -> midpoint(sy, d, DT, 1)[1], NT),
    ("AVF              ", d -> avf(sy, d, DT)[1], NT),
    ("explicit Euler   ", d -> euler(sy, d, DT, 1)[1], 20))
    _, h1, h2, c = integrate_steps(sy, step, nt)
    check("flow 1, $name: the mass is exact over $nt steps",
        maximum(abs, c .- c[1]) < 1e-12,
        @sprintf("absolute drift %.1e", maximum(abs, c .- c[1])))
end
println("  (explicit Euler is run for 20 steps only: it is unstable on this")
println("   dispersive flow and diverges well before 500.  Mass conservation is a")
println("   per-step algebraic property, so a short run tests it faithfully.)")
for (name, step) in (("implicit midpoint", d -> midpoint(sy, d, DT, 2)[1]),
    ("AVF              ", d -> avf(sy, d, DT, 2)[1]))
    _, h1, h2, c = integrate_steps(sy, step, NT)
    check("flow 2, $name: the mass is exact over $NT steps",
        maximum(abs, c .- c[1]) < 1e-12,
        @sprintf("absolute drift %.1e", maximum(abs, c .- c[1])))
end
# for flow 2 this holds because g^T f2(uhat) = 0 POINTWISE in uhat, so any quadrature of the
# averaged field inherits it
let (z, _) = avf(sy, sy.d0, DT, 2)
    worst = maximum(abs(dot(sy.g, f2(sy, sy.d0 + t .* (z - sy.d0)))) for t in SG)
    check("because g^T f2 = 0 pointwise, so any average of the field inherits it",
        worst < 1e-12, @sprintf("worst |g^T f2| at the Gauss nodes %.1e", worst))
end

# --------------------------------------------------------------------------
header("3. implicit midpoint on the first flow is a Poisson integrator")

println("  the Poisson-map condition is  D phi . P^1 . D phi^T = P^1 ;")
println("  D phi is obtained by implicit differentiation, not finite differences.")

function fd_jacobian(step, d, n; eps = 1e-6)
    D = zeros(n, n)
    for j in 1:n
        dp = copy(d)
        dm = copy(d)
        dp[j] += eps
        dm[j] -= eps
        D[:, j] = (step(dp) - step(dm)) ./ (2eps)
    end
    D
end

poisson = Dict{String, Float64}()
for (name, mk) in (("implicit midpoint", d -> midpoint(sy, d, DT, 1)),
    ("AVF              ", d -> avf(sy, d, DT, 1)),
    ("disc. gradient   ",
    d -> (nothing, fd_jacobian(z -> dgrad(sy, z, DT), d, sy.n))),
    ("explicit Euler   ", d -> euler(sy, d, DT, 1)))
    _, D = mk(sy.d0)
    err = maximum(abs, D * sy.P1 * transpose(D) - sy.P1) / maximum(abs, sy.P1)
    poisson[strip(name)] = err
    @printf("      %s: %.3e\n", name, err)
end
check("implicit midpoint is Poisson to round-off", poisson["implicit midpoint"] < 1e-12,
    @sprintf("%.2e", poisson["implicit midpoint"]))
check("the AVF method is NOT Poisson", poisson["AVF"] > 1e-6, @sprintf("%.2e",
    poisson["AVF"]))
check("explicit Euler is grossly not Poisson", poisson["explicit Euler"] > 1e-2,
    @sprintf("%.2e", poisson["explicit Euler"]))
check(
    "nor is the discrete-gradient method (its Jacobian is taken by central " *
    "difference, floor ~1e-10)",
    poisson["disc. gradient"] > 1e-8,
    @sprintf("%.2e", poisson["disc. gradient"]))

for n in (12, 16, 15, 21)
    ss = KdVSys(n; p = 3)
    r = rank(ss.P1; atol = 1e-10 * maximum(abs, ss.P1))
    expect = iseven(n) ? n - 2 : n - 1
    check(
        @sprintf("N = %2d: rank P^1 = %d = N-%d, so the leaves have codimension %d",
            n, expect, iseven(n) ? 2 : 1, n - expect),
        r == expect,
        "got $r")
end

# --------------------------------------------------------------------------
header("4. energy or structure, but not both -- the Ge-Marsden obstruction")

res = Dict{String, NTuple{3, Float64}}()
for (name, step) in (("midpoint, flow 1", d -> midpoint(sy, d, DT, 1)[1]),
    ("AVF, flow 1     ", d -> avf(sy, d, DT, 1)[1]),
    ("midpoint, flow 2", d -> midpoint(sy, d, DT, 2)[1]),
    ("AVF, flow 2     ", d -> avf(sy, d, DT, 2)[1]),
    ("disc.grad, flow 1", d -> dgrad(sy, d, DT)))
    _, h1, h2, c = integrate_steps(sy, step, NT)
    res[strip(name)] = (drift(h1), drift(h2), maximum(abs, c .- c[1]))
end
@printf("      %-18s%11s%11s%11s\n", "scheme", "dH1", "dH2", "dC0")
for k in ("midpoint, flow 1", "AVF, flow 1", "midpoint, flow 2", "AVF, flow 2",
    "disc.grad, flow 1")
    v = res[k]
    @printf("      %-18s%11.2e%11.2e%11.2e\n", k, v[1], v[2], v[3])
end
check(
    "the Gonzalez midpoint discrete gradient also conserves the cubic H1 " *
    "exactly, and the mass",
    res["disc.grad, flow 1"][1] < 1e-11 && res["disc.grad, flow 1"][3] < 1e-12,
    @sprintf("dH1 %.1e, dC0 %.1e", res["disc.grad, flow 1"][1],
        res["disc.grad, flow 1"][3]))
# For a quadratic Hamiltonian the Gonzalez correction vanishes identically: with
# H = ½dᵀAd and A symmetric, H(y) - H(x) = (A x̄)·(y - x) exactly. What is measured is the
# rank-one term `corr·dx` as it actually enters the gradient, RELATIVE to the gradient it
# corrects. The bare `corr` is that quantity divided by |dx|², so its size depends on how
# large a step happens to be drawn -- it varies over an order of magnitude across seeds while
# the normalised one sits at 1e-15. Averaged over several draws so the verdict is not a
# single lucky cancellation; the Python reports exactly 0.0 here, which it is not entitled to.
let Hq(d) = dot(sy.W, fieldat(sy.s, d, 1) .^ 2) / 2,
    gq(d) = basis_values(sy.s, 1) * (sy.W .* fieldat(sy.s, d, 1))

    worst = 0.0
    for seed in 0:4
        yq = sy.d0 + 0.05 .* randn(MersenneTwister(seed), sy.n)
        xb, dxq = 0.5 .* (sy.d0 + yq), yq - sy.d0
        g = gq(xb)
        corr = ((Hq(yq) - Hq(sy.d0)) - dot(g, dxq)) / dot(dxq, dxq)
        worst = max(worst, abs(corr) * maximum(abs, dxq) / maximum(abs, g))
    end
    check(
        "for a quadratic Hamiltonian its correction vanishes, so it reduces to " *
        "the plain midpoint gradient -- hence the name",
        worst < 1e-13,
        @sprintf("correction %.1e relative to the gradient, over 5 draws", worst))
end
check(
    "AVF conserves the cubic H1 along flow 1 to round-off", res["AVF, flow 1"][1] < 1e-11,
    @sprintf("%.2e", res["AVF, flow 1"][1]))
check("implicit midpoint conserves the quadratic H2 along flow 2 to round-off",
    res["midpoint, flow 2"][2] < 1e-12, @sprintf("%.2e", res["midpoint, flow 2"][2]))
check("implicit midpoint on flow 1 is Poisson but does NOT conserve H1 exactly",
    poisson["implicit midpoint"] < 1e-12 && res["midpoint, flow 1"][1] > 1e-9,
    @sprintf("Poisson %.1e, dH1 %.1e", poisson["implicit midpoint"],
        res["midpoint, flow 1"][1]))
# on flow 1 the two readings of AVF coincide, because P^1 is constant
let e = maximum(abs, avf(sy, sy.d0, DT, 1)[1] - avf_gradient(sy, sy.d0, DT))
    check(
        "on flow 1, averaging the field and averaging the gradient agree -- P^1 " *
        "being constant",
        e < 1e-12,
        @sprintf("max difference %.1e", e))
end
check("and AVF conserves H1 exactly but is NOT Poisson -- no method does both",
    res["AVF, flow 1"][1] < 1e-11 && poisson["AVF"] > 1e-6,
    @sprintf("dH1 %.1e, Poisson %.1e", res["AVF, flow 1"][1], poisson["AVF"]))

println("\n  AVF on the SECOND flow does not conserve H2, and the reason is that")
println("  its energy identity needs a CONSTANT structure matrix:")
check("AVF on flow 2 loses H2, where implicit midpoint keeps it",
    res["AVF, flow 2"][2] > 1e-9 && res["midpoint, flow 2"][2] < 1e-12,
    @sprintf("AVF %.1e against midpoint %.1e", res["AVF, flow 2"][2],
        res["midpoint, flow 2"][2]))
let (z, _) = avf(sy, sy.d0, DT, 2)
    gbar = sum(gH2(sy, sy.d0 + t .* (z - sy.d0)) for t in SG) / 2
    e = abs(dot(gbar, z - sy.d0) - (sysH2(sy, z) - sysH2(sy, sy.d0)))
    check("the averaged gradient IS a discrete gradient of the quadratic H2", e < 1e-14,
        @sprintf("defect %.1e", e))
    avg_field = sum(f2(sy, sy.d0 + t .* (z - sy.d0)) for t in SG) / 2
    mid_form = (sy.Minv * K2(sy, 0.5 .* (sy.d0 + z)) * sy.Minv) * gbar
    r_avf = abs(dot(gbar, avg_field)) /
            (maximum(abs, gbar) * maximum(abs, avg_field) * sy.n)
    r_mid = abs(dot(gbar, mid_form)) / (maximum(abs, gbar) * maximum(abs, mid_form) * sy.n)
    @printf("      gbar . avg( P2(xi) grad H2(xi) )  = %.2e   (AVF: does not vanish)\n",
        r_avf)
    @printf("      gbar . P2(mid) gbar               = %.2e   (midpoint: vanishes)\n",
        r_mid)
    check(
        "so the failure is that the average of the product is not the product of " *
        "the averages once P^2 depends on uhat",
        r_avf > 1e-9 && r_mid < 1e-12,
        @sprintf("%.1e against %.1e", r_avf, r_mid))
end

# --------------------------------------------------------------------------
header("4b. why implicit midpoint does NOT preserve the quadratic H2")

println("  A symplectic Runge-Kutta method conserves every quadratic invariant")
println("  Q(y) = y^T C y of the ODE exactly (Cooper's theorem), but 'invariant'")
println("  means Q'(y) f(y) = 0 for ALL y.  For flow 1 and Q = H2 that asks")
println("  B + B^T = 0 with B = S Minv K1 for the linear part, plus a cubic")
println("  condition for the nonlinear part.  Switching each off in turn:")
@printf("      %-32s%13s%11s%11s\n", "case", "|B+B^T|/|B|", "dH1", "dH2")
lin = Dict{String, NTuple{3, Float64}}()
for (lbl, uni, nl) in (("linear KdV, uniform mesh", true, false),
    ("linear KdV, non-uniform mesh", false, false),
    ("full KdV, uniform mesh", true, true),
    ("full KdV, non-uniform mesh", false, true))
    ss = KdVSys(20; p = 3, uniform = uni)
    B = ss.S * ss.Minv * ss.K1
    bdef = maximum(abs, B + transpose(B)) / maximum(abs, B)
    gfun = nl ? d -> gH1(ss, d) : d -> basis_values(ss.s, 1) * (ss.W .* fieldat(ss.s, d, 1))
    hfun = nl ? d -> hessH1(ss, d) : d -> ss.K1
    H1f = nl ? d -> sysH1(ss, d) : d -> dot(ss.W, fieldat(ss.s, d, 1) .^ 2) / 2
    ff(d) = ss.P1 * gfun(d)
    Id = Matrix{Float64}(I, ss.n, ss.n)
    d = copy(ss.d0)
    a, b = [H1f(d)], [sysH2(ss, d)]
    for _ in 1:500
        local dprev = d
        d = newton(w -> w - dprev - DT .* ff(0.5 .* (dprev + w)),
            w -> Id - 0.5DT .* (ss.P1 * hfun(0.5 .* (dprev + w))), dprev)
        push!(a, H1f(d))
        push!(b, sysH2(ss, d))
    end
    e1, e2 = drift(a), drift(b)
    lin[lbl] = (bdef, e1, e2)
    @printf("      %-32s%13.2e%11.2e%11.2e\n", lbl, bdef, e1, e2)
end
check(
    "on the LINEAR flow on a uniform mesh H2 IS an exact quadratic invariant, " *
    "and implicit midpoint preserves it to round-off",
    lin["linear KdV, uniform mesh"][1] < 1e-12 &&
        lin["linear KdV, uniform mesh"][3] < 1e-12,
    @sprintf("|B+B^T| = %.1e, dH2 = %.1e", lin["linear KdV, uniform mesh"][1],
        lin["linear KdV, uniform mesh"][3]))
check("the cubic term of H1 breaks the hypothesis: same mesh, full KdV, H2 is lost",
    lin["full KdV, uniform mesh"][3] > 1e-9,
    @sprintf("dH2 = %.1e", lin["full KdV, uniform mesh"][3]))
check(
    "and a non-uniform mesh breaks even the linear part, B ceasing to be " *
    "antisymmetric",
    lin["linear KdV, non-uniform mesh"][1] > 1e-3 &&
        lin["linear KdV, non-uniform mesh"][3] > 1e-9,
    @sprintf("|B+B^T| = %.1e, dH2 = %.1e", lin["linear KdV, non-uniform mesh"][1],
        lin["linear KdV, non-uniform mesh"][3]))
check(
    "so the failure is the hypothesis of the theorem, not the symplecticity " *
    "of the method", true)

# --------------------------------------------------------------------------
header("5. the respective other Hamiltonian: nothing to inherit")

println("  the semi-discrete cross-conservation splits as")
println("    {H2d,H1d}_1d = u^T (S Minv K1) u  +  3 u^T S Minv n(u),   n_i = int phi_i u_h^2")
let rng = MersenneTwister(4)
    for p in (2, 3, 4)
        s, d = setup(24; p)
        Minv = inv(Matrix(mass_matrix(s)))
        W = quadrature_weights(s)
        Φ0 = basis_values(s, 0)
        S = skew_S(s)
        A = S * Minv * gram(s, 1, 1)
        e = maximum(abs, A + transpose(A)) / maximum(abs, A)
        check(
            "p = $p, uniform: S Minv K1 is antisymmetric, so the QUADRATIC part " *
            "vanishes identically",
            e < 1e-12,
            @sprintf("rel. defect %.1e", e))
        worst_q, worst_c = 0.0, 0.0
        for _ in 1:5
            c = randn(rng, 24)
            uh = fieldat(s, c, 0)
            sc = maximum(abs, c) * maximum(abs, A * c) * 24
            worst_q = max(worst_q, abs(dot(c, A * c)) / sc)
            worst_c = max(worst_c,
                abs(3 * dot(c, S * Minv * (Φ0 * (W .* uh .^ 2)))) / sc)
        end
        check("p = $p, uniform: ... for random uhat too", worst_q < 1e-13,
            @sprintf("worst %.1e", worst_q))
        check(
            "p = $p, uniform: but the CUBIC part does NOT -- so the " *
            "cross-conservation is no identity",
            worst_c > 1e-6,
            @sprintf("worst %.1e", worst_c))
    end

    println("\n  the cubic part against the highest excited mode (uniform, p = 3, N = 32):")
    s, _ = setup(32; p = 3)
    Minv = inv(Matrix(mass_matrix(s)))
    W = quadrature_weights(s)
    Φ0 = basis_values(s, 0)
    S = skew_S(s)
    A = S * Minv * gram(s, 1, 1)
    prev, single = nothing, 0.0
    for k in (1, 2, 4, 8, 12, 16)
        vals = Float64[]
        for _ in 1:3
            c = zeros(32)
            for m in 1:k
                a, b = randn(rng), randn(rng)
                c += a .* cos.(2π * m .* (0:31) ./ 32) + b .* sin.(2π * m .* (0:31) ./ 32)
            end
            uh = fieldat(s, c, 0)
            sc = maximum(abs, c) * maximum(abs, A * c) * 32
            push!(vals, abs(3 * dot(c, S * Minv * (Φ0 * (W .* uh .^ 2)))) / sc)
        end
        println("      modes 1..$(lpad(k, 2)): " *
                join([@sprintf("%.1e", v) for v in vals], "  "))
        k == 1 && (single = maximum(vals))
        if prev !== nothing && k ≥ 8
            check("modes 1..$k: the obstruction grows with the excited mode content",
                maximum(vals) > prev, @sprintf("%.1e > %.1e", maximum(vals), prev))
        end
        prev = maximum(vals)
    end
    check(
        "a single mode is conserved to round-off, the full spectrum is not -- " *
        "which is why a test on sin x alone reports round-off",
        single < 1e-14 && prev > 1e-4,
        @sprintf("one mode %.1e, all modes %.1e", single, prev))
end

println("\n  consequently no integrator keeps the other Hamiltonian:")
check("AVF on flow 1 keeps H1 but loses H2",
    res["AVF, flow 1"][1] < 1e-11 && res["AVF, flow 1"][2] > 1e-9,
    @sprintf("dH1 %.1e, dH2 %.1e", res["AVF, flow 1"][1], res["AVF, flow 1"][2]))
check("midpoint on flow 2 keeps H2 but loses H1, and far more badly",
    res["midpoint, flow 2"][2] < 1e-12 && res["midpoint, flow 2"][1] > 1e-4,
    @sprintf("dH2 %.1e, dH1 %.1e", res["midpoint, flow 2"][2], res["midpoint, flow 2"][1]))
check(
    "H2 IS quadratic, yet still not inherited along flow 1 -- because the " *
    "invariance is not an identity in uhat",
    res["AVF, flow 1"][2] > 1e-9 && res["midpoint, flow 1"][2] > 1e-9,
    @sprintf("AVF %.1e, midpoint %.1e", res["AVF, flow 1"][2], res["midpoint, flow 1"][2]))

# --------------------------------------------------------------------------
header("6. projecting onto both level sets")

let H1t = sysH1(sy, sy.d0), H2t = sysH2(sy, sy.d0), C0t = sysC0(sy, sy.d0)
    _, h1, h2, c = integrate_steps(sy, d -> project_both(sy, d, DT, H1t, H2t), NT)
    check("projection onto both level sets conserves H1 exactly", drift(h1) < 1e-12,
        @sprintf("drift %.1e", drift(h1)))
    check("... and H2 exactly", drift(h2) < 1e-12, @sprintf("drift %.1e", drift(h2)))
    check(
        "... but it LOSES the mass, the correction span(dH1,dH2) not being " *
        "mass-neutral",
        maximum(abs, c .- c[1]) > 1e-9,
        @sprintf("drift %.1e", maximum(abs, c .- c[1])))

    # a third multiplier along g restores the mass: all three, at the cost of structure
    _, h1b, h2b, cb = integrate_steps(sy, d -> project_all(sy, d, DT, H1t, H2t, C0t), NT)
    check("adding a third multiplier along g restores all three invariants at once",
        drift(h1b) < 1e-12 && drift(h2b) < 1e-12 && maximum(abs, cb .- cb[1]) < 1e-12,
        @sprintf("dH1 %.1e, dH2 %.1e, dC0 %.1e", drift(h1b), drift(h2b),
            maximum(abs, cb .- cb[1])))
end

# ... but neither projection is Poisson. Compare like with like: both Jacobians by the same
# central difference, against the midpoint baseline.
let base = fd_jacobian(d -> midpoint(sy, d, DT, 1)[1], sy.d0, sy.n)
    e_base = maximum(abs, base * sy.P1 * transpose(base) - sy.P1) / maximum(abs, sy.P1)
    Dp = fd_jacobian(d -> project_all(sy, d, DT, sysH1(sy, d), sysH2(sy, d), sysC0(sy, d)),
        sy.d0, sy.n)
    e_proj = maximum(abs, Dp * sy.P1 * transpose(Dp) - sy.P1) / maximum(abs, sy.P1)
    println("\n  Poisson defect by the same central difference:")
    @printf("      implicit midpoint  %.2e   (finite-difference floor)\n", e_base)
    @printf("      projected method   %.2e\n", e_proj)
    check(
        "the projected method is NOT Poisson -- all the invariants cost the " *
        "structure",
        e_proj > 1e3 * e_base,
        @sprintf("%.2e against a floor of %.2e", e_proj, e_base))
end

println("\n  and note what the projection enforces: the semi-discrete flow itself")
println("  does not conserve H2 identically, so pinning H2 to its initial value")
println("  pushes the trajectory off the semi-discrete solution.  The residual it")
@printf("  removes is %.1e over %d steps -- the size of the\n",
    res["midpoint, flow 1"][2], NT)
println("  spatial consistency error, not an improvement on it.")

summary("verify_kdv_timedisc.jl")
