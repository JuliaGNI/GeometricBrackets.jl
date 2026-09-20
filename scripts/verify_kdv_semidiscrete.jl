#!/usr/bin/env julia
#
# The semi-discrete KdV equation: what each bracket generates, and what it conserves.
#
#     julia --project=scripts scripts/verify_kdv_semidiscrete.jl
#
#   1.  the two discrete gradients, against their closed forms and finite differences;
#   2.  the FIRST bracket gives a mixed two-field Galerkin method, M udot = S w with
#       w_h = Pi_h(-3u_h^2 - u_h,xx). It runs at p = 1, but the reading of w needs p >= 2;
#   3.  the SECOND bracket collapses to the plain Galerkin scheme -- and the form of the
#       nonlinearity is not a choice, needing quadrature exact to degree 3p-1;
#   4.  the two flows are NOT the same. They differ by exactly int phi_i' (I - Pi_h) r,
#       which converges at 2p -- the same order as each scheme's own consistency error;
#   5.  conservation of each Hamiltonian along its own flow, and of the mass along both;
#   5b. the mass is a Casimir of P^1 but NOT of P^2; instead Magri's first rung
#       P^2 g = -2 P^1 dH2/du holds EXACTLY at finite N;
#   5c. the skew-symmetrised K2 conserves H2 at EVERY quadrature; the unsymmetrised form
#       needs degree 3p-1;
#   6.  the neighbouring flows in the hierarchy, and the second Magri rung which holds only
#       to O(h^2p) -- the same signature as the Jacobi failure;
#   7.  implicit midpoint, the average-vector-field method, and explicit Euler as a control.
#
# Beware the broadcast orientation throughout: Φ is (nb × nq) and the weights are (nq,), so
# numpy's `W * P[d]` -- which scales the QUADRATURE axis -- is `Φ .* transpose(W)` here.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary
include(joinpath(@__DIR__, "kdvtools.jl"));
using .KdVTools

const TOL = 1e-9

KDV_RHS(x) = -3.0 * sin(2x) + cos(x)                      # -6uu_x - u_xxx, u = sin
KDV5_RHS(x) = 30.0 * sin(x)^2 * cos(x) - 30.0 * sin(x) * cos(x) + cos(x)

"udot = P^1 dH1/du, and the auxiliary variable w = M^-1 dH1/du."
function flow1(s, û)
    M = Matrix(mass_matrix(s))
    w = M \ grad_H1(s, û)
    return M \ (skew_S(s) * w), w
end

"udot = P^2(u) dH2/du = M^-1 K2(u) u."
flow2(s, û) = poisson_matrix(kdv_bracket_2(s), û) * grad_H2(s, û)

# --------------------------------------------------------------------------
header("1. the two discrete gradients")

for p in (2, 3, 4), uni in (true, false)

    tag = "p = $p, $(uni ? "uniform" : "non-uniform")"
    s, d = setup(24; p, uniform = uni)
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    uh, ux = fieldat(s, d, 0), fieldat(s, d, 1)

    g2 = grad_H2(s, d)
    e = maximum(abs, g2 - Φ0 * (W .* uh)) / maximum(abs, g2)
    check("$tag: dH2/du_i = (M u)_i = int phi_i u_h", e < TOL, @sprintf("rel. error %.2e",
        e))

    g1 = grad_H1(s, d)
    direct = Φ1 * (W .* ux) - 3.0 .* (Φ0 * (W .* uh .^ 2))
    e = maximum(abs, g1 - direct) / maximum(abs, g1)
    check("$tag: dH1/du_i = int ( phi_i' u_h,x - 3 phi_i u_h^2 )", e < TOL,
        @sprintf("rel. error %.2e", e))

    # both against a central difference of the quadrature Hamiltonians
    H1(dd) = dot(W, fieldat(s, dd, 1) .^ 2 - 2 .* fieldat(s, dd, 0) .^ 3) / 2
    H2(dd) = dot(W, fieldat(s, dd, 0) .^ 2) / 2
    for (name, H, g) in (("H1", H1, g1), ("H2", H2, g2))
        fd = zero(g)
        eps = 1e-6
        for i in eachindex(d)
            dp = copy(d)
            dm = copy(d)
            dp[i] += eps
            dm[i] -= eps
            fd[i] = (H(dp) - H(dm)) / (2eps)
        end
        e = maximum(abs, g - fd) / maximum(abs, fd)
        check("$tag: the $name gradient matches a finite difference", e < 1e-7,
            @sprintf("rel. error %.2e", e))
    end
end

# --------------------------------------------------------------------------
header("2. the first bracket gives a mixed two-field Galerkin method")

for p in (2, 3, 4), uni in (true, false)

    tag = "p = $p, $(uni ? "uniform" : "non-uniform")"
    s, d = setup(24; p, uniform = uni)
    M = Matrix(mass_matrix(s))
    W = quadrature_weights(s)
    Φ0 = basis_values(s, 0)
    f1, w = flow1(s, d)

    e = maximum(abs, M * f1 - skew_S(s) * w) / maximum(abs, M * f1)
    check("$tag: the flow is M udot = S w with w = M^-1 dH1/du", e < TOL,
        @sprintf("rel. error %.2e", e))

    # w is the Galerkin approximation of -3u^2 - u_xx
    r = -3.0 .* fieldat(s, d, 0) .^ 2 .- fieldat(s, d, 2)
    pr = M \ (Φ0 * (W .* r))
    e = maximum(abs, w - pr) / maximum(abs, w)
    check("$tag: w_h = Pi_h ( -3 u_h^2 - u_h,xx )", e < TOL, @sprintf("rel. error %.2e", e))
end

# only first derivatives of the basis enter, so the scheme runs at p = 1 ...
let (s, d) = setup(24; p = 1, uniform = false, nq = 4)
    M = Matrix(mass_matrix(s))
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    f1, w = flow1(s, d)
    check("p = 1: the mixed scheme is still well defined and nonzero",
        maximum(abs, f1) > 1e-3, @sprintf("|udot| = %.2e", maximum(abs, f1)))
    check("p = 1: only phi and phi' enter the assembly, no second derivative",
        maximum(abs,
            grad_H1(s, d) -
            (Φ1 * (W .* fieldat(s, d, 1)) - 3.0 .* (Φ0 * (W .* fieldat(s, d, 0) .^ 2)))) <
        1e-14)
    # ... but the identification of w_h breaks down, u_h,xx being zero in every cell
    r = -3.0 .* fieldat(s, d, 0) .^ 2 .- fieldat(s, d, 2)
    check("p = 1: u_h,xx vanishes identically inside the cells",
        maximum(abs, fieldat(s, d, 2)) < 1e-9,
        @sprintf("max |u_h,xx| = %.2e", maximum(abs, fieldat(s, d, 2))))
    gap = maximum(abs, w - M \ (Φ0 * (W .* r))) / maximum(abs, w)
    check(
        "p = 1: so the reading w_h = Pi_h ( -3 u_h^2 - u_h,xx ) FAILS -- it needs p >= 2",
        gap > 1e-2, @sprintf("rel. discrepancy %.2f", gap))
end

# --------------------------------------------------------------------------
header("3. the second bracket collapses to the plain Galerkin scheme")

for p in (2, 3, 4), uni in (true, false), n in (16, 24)
    tag = "p = $p, $(uni ? "uniform" : "non-uniform"), N = $n"
    s, d = setup(n; p, uniform = uni)
    M = Matrix(mass_matrix(s))
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    uh, ux, uxx = fieldat(s, d, 0), fieldat(s, d, 1), fieldat(s, d, 2)

    f2 = flow2(s, d)
    e = maximum(abs, M * f2 - K2apply(s, d, d)) / maximum(abs, M * f2)
    check("$tag: one M^-1 cancels the mass matrix: M udot = K2(u) u", e < TOL,
        @sprintf("rel. error %.2e", e))

    plain = -(Φ0 * (W .* 6.0 .* uh .* ux)) + Φ1 * (W .* uxx)
    e = maximum(abs, M * f2 - plain) / maximum(abs, plain)
    check("$tag: which IS int ( -6 u_h u_h,x phi_i + phi_i' u_h,xx )", e < TOL,
        @sprintf("rel. error %.2e", e))

    # the form of the nonlinearity is not a choice
    a = Φ0 * (W .* 6.0 .* uh .* ux)
    e = maximum(abs, a + 3.0 .* (Φ1 * (W .* uh .^ 2))) / maximum(abs, a)
    check("$tag: int 6 u_h u_h,x phi_i = -3 int u_h^2 phi_i'", e < TOL,
        @sprintf("rel. error %.2e", e))
end

println("\n  the quadrature must be exact to degree 3p-1 for that last identity:")
for nq in 2:6
    s, d = setup(16; p = 3, uniform = false, nq = nq)
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    uh, ux = fieldat(s, d, 0), fieldat(s, d, 1)
    a = Φ0 * (W .* 6.0 .* uh .* ux)
    e = maximum(abs, a + 3.0 .* (Φ1 * (W .* uh .^ 2))) / maximum(abs, a)
    @printf("      nq = %d  (%10s):   rel. error %.2e\n",
        nq, nq ≥ nq_for(3) ? "exact" : "too coarse", e)
end

println("\n  at p = 2 the un-integrated third-derivative block vanishes identically,")
println("  so the integration by parts of eq. (discrete-kdv-poisson-bracket-2-tensor)")
println("  is not a regularity nicety -- it is the only correct reading:")
for p in (2, 3)
    s, _ = setup(20; p, nq = max(nq_for(p), p + 2))
    raw = -gram(s, 0, 3)                 # -∫ φ_k φ_l'''
    byparts = gram(s, 1, 2)              #  ∫ φ_k' φ_l''
    @printf("      p = %d:  max |-int phi_k phi_l'''| = %.3e,   max |int phi_k' phi_l''| = %.3e\n",
        p, maximum(abs, raw), maximum(abs, byparts))
    if p == 2
        check("p = 2: -int phi_k phi_l''' is identically zero", maximum(abs, raw) < 1e-10,
            @sprintf("max %.2e", maximum(abs, raw)))
        check(
            "p = 2: while the by-parts block does not vanish", maximum(abs, byparts) > 1.0,
            @sprintf("max %.2e", maximum(abs, byparts)))
    else
        e = maximum(abs, raw - byparts) / maximum(abs, byparts)
        check("p = 3: the two agree, as in verify_kdv_discrete.jl", e < TOL,
            @sprintf("rel. error %.2e", e))
    end
end

# --------------------------------------------------------------------------
header("4. the two flows are not the same")

for p in (2, 3), uni in (true, false)

    tag = "p = $p, $(uni ? "uniform" : "non-uniform")"
    s, d = setup(24; p, uniform = uni)
    M = Matrix(mass_matrix(s))
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    f1, _ = flow1(s, d)
    f2 = flow2(s, d)
    rel = maximum(abs, f1 - f2) / maximum(abs, f1)
    check("$tag: the two flows differ at finite N", rel > 1e-6,
        @sprintf("rel. difference %.2e", rel))

    # the exact difference identity -- an ABSOLUTE check: the quantity is O(h^2p) and
    # normalising by it would only measure cancellation
    r = -3.0 .* fieldat(s, d, 0) .^ 2 .- fieldat(s, d, 2)
    pi_r = transpose(Φ0) * (M \ (Φ0 * (W .* r)))
    scale = maximum(abs, M * f1)
    e = maximum(abs, M * (f1 - f2) - Φ1 * (W .* (r - pi_r)))
    check("$tag: M (udot_1 - udot_2)_i = int phi_i' ( I - Pi_h ) r", e < 1e-9 * scale,
        @sprintf("absolute error %.2e against scale %.2e", e, scale))

    dphi = Φ1 - transpose(M \ (Φ0 * transpose(Φ1 .* transpose(W)))) * Φ0
    e = maximum(abs, M * (f1 - f2) - dphi * (W .* (r - pi_r)))
    check("$tag: = ( ( I - Pi_h ) phi_i' , ( I - Pi_h ) r )", e < 1e-9 * scale,
        @sprintf("absolute error %.2e against scale %.2e", e, scale))
end

# both constituents of the difference are separately nonzero
let (s, d) = setup(24; p = 3)
    M = Matrix(mass_matrix(s))
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    uh = fieldat(s, d, 0)
    pi_nl = transpose(Φ0) * (M \ (Φ0 * (W .* 3.0 .* uh .^ 2)))
    v = maximum(abs, Φ1 * (W .* (3.0 .* uh .^ 2 - pi_nl)))
    check("the nonlinear part of the projection error is not orthogonal to d_x V_h",
        v > 1e-12, @sprintf("max |int phi_i' ( I - Pi_h ) 3 u_h^2| = %.2e", v))
end

println("\n  the flows already differ on LINEAR KdV, u_t = -u_xxx, where flow 1 reads")
println("  M udot = S M^-1 K1 u with K1 the stiffness matrix and flow 2 M udot = K0 u.")
println("  The relative gap is O(1) and INDEPENDENT of the resolution:")
for p in (2, 3, 4)
    vals = Float64[]
    for n in (16, 24, 32)
        s, _ = setup(n; p)
        Minv = inv(Matrix(mass_matrix(s)))
        K1 = gram(s, 1, 1)                       # ∫φ_i'φ_j', = dH1_lin/du
        K0 = gram(s, 1, 2)                       # ∫φ_k'φ_l''
        push!(vals, maximum(abs, skew_S(s) * Minv * K1 - K0) / maximum(abs, K0))
    end
    println("      p = $p:  " * join([@sprintf("%.4f", v) for v in vals], "  "))
    check("p = $p: S M^-1 K1 =/= K0, and the gap does not shrink with N",
        minimum(vals) > 1e-2 && maximum(vals) - minimum(vals) < 1e-3,
        @sprintf("gap %.4f flat in N", vals[1]))
end

println("\n  the projection error itself converges only at about p-1 ...")
for p in (2, 3, 4)
    ns = (16, 32, 64)
    errs = Float64[]
    for n in ns
        s, d = setup(n; p, graded = true, uniform = false, f = sin)
        M = Matrix(mass_matrix(s))
        W = quadrature_weights(s)
        Φ0 = basis_values(s, 0)
        r = -3.0 .* fieldat(s, d, 0) .^ 2 .- fieldat(s, d, 2)
        pi_r = transpose(Φ0) * (M \ (Φ0 * (W .* r)))
        push!(errs, sqrt(dot(W, (r - pi_r) .^ 2)))
    end
    o = rates(errs, ns)
    println("      p = $p:  " * e3(errs) * "    orders $(o2(o))")
    check("p = $p: || ( I - Pi_h ) r || converges at about p-1 = $(p - 1)",
        abs(minimum(o) - (p - 1)) < 0.4, "observed $(o2(o))")
end

println("\n  ... but it is nearly orthogonal to d_x V_h, and the flow difference")
println("  converges at about 2p, on a uniform and on a smoothly graded mesh:")
for (p, ns) in ((2, (16, 32, 64, 128)), (3, (16, 32, 64)), (4, (12, 16, 24))),
    graded in (false, true)

    errs = Float64[]
    for n in ns
        s, d = setup(n; p, uniform = !graded, graded = graded, f = sin)
        W = quadrature_weights(s)
        Φ0 = basis_values(s, 0)
        f1, _ = flow1(s, d)
        f2 = flow2(s, d)
        push!(errs, sqrt(dot(W, (transpose(Φ0) * (f1 - f2)) .^ 2)))
    end
    o = rates(errs, ns)
    @printf("      p = %d, %7s:  %s    orders %s\n", p, graded ? "graded" : "uniform",
        e3(errs), o2(o))
    check(
        "p = $p, $(graded ? "graded" : "uniform"): the flow difference converges at " *
        "about 2p = $(2p)",
        minimum(o) > 2p - 0.6,
        "observed $(o2(o))")
end

println("\n  and 2p is also each scheme's own consistency order, so the difference")
println("  is no larger than the error either flow already makes:")
for (p, ns) in ((2, (16, 32, 64)), (3, (16, 32, 64)), (4, (12, 16, 24)))
    out = Vector{Float64}[]
    for which in (1, 2)
        errs = Float64[]
        for n in ns
            s, d = setup(n; p, f = sin)
            M = Matrix(mass_matrix(s))
            W = quadrature_weights(s)
            Φ0 = basis_values(s, 0)
            X = quadrature_nodes(s)
            f = which == 1 ? first(flow1(s, d)) : flow2(s, d)
            ex = M \ (Φ0 * (W .* KDV_RHS.(X)))
            push!(errs, sqrt(dot(W, (transpose(Φ0) * (f - ex)) .^ 2)))
        end
        o = rates(errs, ns)
        push!(out, o)
        check("p = $p: flow $which is consistent with 6uu_x - u_xxx at about 2p = $(2p)",
            minimum(o) > 2p - 0.6, "observed $(o2(o))")
    end
    println("      p = $p:  flow 1 orders $(o2(out[1])),   flow 2 orders $(o2(out[2]))")
end

# --------------------------------------------------------------------------
header("5. conservation of the Hamiltonians, and of the mass")

for p in (2, 3, 4), uni in (true, false), n in (16, 32)
    tag = "p = $p, $(uni ? "uniform" : "non-uniform"), N = $n"
    s, d = setup(n; p, uniform = uni)
    M = Matrix(mass_matrix(s))
    g1, g2 = grad_H1(s, d), grad_H2(s, d)
    f1, _ = flow1(s, d)
    f2 = flow2(s, d)

    e = abs(dot(g1, f1)) / (maximum(abs, g1) * maximum(abs, f1) * length(d))
    check("$tag: dH1/dt = 0 along the first flow, by antisymmetry", e < 1e-12,
        @sprintf("normalised rate %.2e", e))
    e = abs(dot(g2, f2)) / (maximum(abs, g2) * maximum(abs, f2) * length(d))
    check("$tag: dH2/dt = 0 along the second flow, by antisymmetry", e < 1e-12,
        @sprintf("normalised rate %.2e", e))

    g = mass_grad(s)
    e = maximum(abs, M * ones(length(d)) - g) / maximum(abs, g)
    check("$tag: partition of unity gives M 1 = g", e < TOL, @sprintf("rel. error %.2e", e))
    for (k, f) in ((1, f1), (2, f2))
        e = abs(dot(g, f)) / (maximum(abs, g) * maximum(abs, f) * length(d))
        check("$tag: the mass is conserved exactly along flow $k", e < 1e-12,
            @sprintf("normalised rate %.2e", e))
    end
end

header("5b. the mass under P^1 is a Casimir; under P^2 it is not")

for p in (2, 3), uni in (true, false)

    tag = "p = $p, $(uni ? "uniform" : "non-uniform")"
    s, d = setup(24; p, uniform = uni)
    g = mass_grad(s)
    P1 = poisson_matrix(kdv_bracket_1(s), d)
    P2 = poisson_matrix(kdv_bracket_2(s), d)
    e = maximum(abs, P1 * g) / (maximum(abs, P1) * maximum(abs, g))
    check("$tag: P^1 g = 0 -- C0 is an exact Casimir of the first bracket", e < 1e-12,
        @sprintf("normalised %.2e", e))
    rng = MersenneTwister(7)
    for (name, gr) in (("H1", grad_H1(s, d)), ("H2", grad_H2(s, d)),
        ("H3", grad_H3(s, d)),
        ("a random functional", randn(rng, length(d))))
        e = abs(dot(g, P1 * gr)) / (maximum(abs, g) * maximum(abs, P1 * gr) * length(d))
        check("$tag: hence the mass is conserved under the P^1 flow of $name", e < 1e-12,
            @sprintf("normalised rate %.2e", e))
    end

    e = maximum(abs, P2 * g) / (maximum(abs, P2) * maximum(abs, g))
    check("$tag: P^2 g =/= 0 -- C0 is NOT a Casimir of the second bracket", e > 1e-6,
        @sprintf("normalised %.2e", e))
    # Magri's first rung, exactly at finite N
    rung = -2.0 .* (P1 * grad_H2(s, d))
    e = maximum(abs, P2 * g - rung) / maximum(abs, rung)
    check("$tag: instead P^2 g = -2 P^1 dH2/du, exactly", e < TOL,
        @sprintf("rel. error %.2e", e))
    # so mass conservation along flow 2 is Hamiltonian-specific
    lhs = dot(g, P2 * grad_H2(s, d))
    rhs = -2.0 * dot(grad_H2(s, d), P1 * grad_H2(s, d))
    sc = maximum(abs, g) * maximum(abs, P2 * grad_H2(s, d)) * length(d)
    check("$tag: g^T P^2 dH2/du = -2 dH2/du . P^1 dH2/du = 0",
        abs(lhs - rhs) < 1e-12 * sc && abs(lhs) < 1e-12 * sc, @sprintf("both %.2e", lhs))
    gr = randn(rng, length(d))
    e = abs(dot(g, P2 * gr)) / (maximum(abs, g) * maximum(abs, P2 * gr) * length(d))
    check("$tag: but NOT along the P^2 flow of a generic functional", e > 1e-9,
        @sprintf("normalised rate %.2e", e))
end

header("5c. the skew form makes H2 conservation quadrature-independent")

println("  randomly non-uniform mesh, p = 3, N = 16.  The skew-symmetrised K2 of")
println("  eq. (discrete-kdv-poisson-bracket-2-tensor) is antisymmetric by")
println("  construction, so dH2/dt vanishes at EVERY nq; the unsymmetrised form")
println("  needs degree 3p-1, i.e. nq >= 5.  The mass needs only degree 2p-1.")
@printf("      %3s %10s %10s %10s %10s %10s %10s %10s\n",
    "nq", "|S+S^T|", "|K2+K2^T|", "|K2un+T|", "dH1/dt", "dH2/dt", "dH2 unsym", "dC0/dt")
sweep = Dict{Int, NTuple{4, Float64}}()
for nq in 2:7
    s, d = setup(16; p = 3, uniform = false, nq = nq)
    M = Matrix(mass_matrix(s))
    S = gram(s, 0, 1)
    Ku, K0 = K2_blocks(s)
    K2 = K0 + sum(d[m] .* Ku[m, :, :] for m in axes(Ku, 1))
    Kun, K0un = K2_blocks_unsym(s)
    K2un = K0un + sum(d[m] .* Kun[m, :, :] for m in axes(Kun, 1))
    g1, g2, g = grad_H1(s, d), grad_H2(s, d), mass_grad(s)
    f1, _ = flow1(s, d)
    f2 = M \ (K2 * d)
    f2un = M \ (K2un * d)
    nrm(a, f) = abs(dot(a, f)) / (maximum(abs, a) * maximum(abs, f) * length(d))
    sweep[nq] = (nrm(g1, f1), nrm(g2, f2), nrm(g2, f2un), nrm(g, f2))
    @printf("      %3d %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n",
        nq, maximum(abs, S + transpose(S)), maximum(abs, K2 + transpose(K2)),
        maximum(abs, K2un + transpose(K2un)), sweep[nq]...)
end
for nq in 2:7
    check("nq = $nq: the skew K2 is antisymmetric and conserves H2 to round-off",
        sweep[nq][2] < 1e-12, @sprintf("dH2/dt = %.2e", sweep[nq][2]))
end
check("the unsymmetrised form does NOT, at nq = 3", sweep[3][3] > 1e-9,
    @sprintf("dH2/dt = %.2e against %.2e for the skew form", sweep[3][3], sweep[3][2]))
check("both forms agree once the quadrature is exact, nq >= $(nq_for(3))",
    sweep[nq_for(3)][3] < 1e-12, @sprintf("dH2/dt = %.2e", sweep[nq_for(3)][3]))
check(
    "H1 is conserved at every nq too -- P^1 is skew-symmetrised by " *
    "construction, eq. (kdv-poisson-bracket-1-skew)",
    maximum(sweep[nq][1] for nq in keys(sweep)) < 1e-12,
    @sprintf("worst dH1/dt = %.2e", maximum(sweep[nq][1] for nq in keys(sweep))))
check("but the MASS still needs degree 2p-1: it is not conserved at nq = 2",
    sweep[2][4] > 1e-9, @sprintf("dC0/dt = %.2e", sweep[2][4]))
check("and is conserved for nq >= p = 3", sweep[3][4] < 1e-12,
    @sprintf("dC0/dt = %.2e", sweep[3][4]))
let (s, d) = setup(16; p = 3, uniform = true, nq = 2)
    S = gram(s, 0, 1)
    e = maximum(abs, S + transpose(S)) / maximum(abs, S)
    check(
        "on a UNIFORM mesh S stays antisymmetric even at nq = 2, by translation " *
        "invariance",
        e < 1e-12,
        @sprintf("rel. defect %.2e", e))
end

# --------------------------------------------------------------------------
header("6. the neighbouring flows in the hierarchy")

for p in (2, 3)
    s, d = setup(24; p)
    M = Matrix(mass_matrix(s))
    W = quadrature_weights(s)
    Φ0 = basis_values(s, 0)
    trans = poisson_matrix(kdv_bracket_1(s), d) * grad_H2(s, d)
    e = maximum(abs, trans - M \ (skew_S(s) * d)) / maximum(abs, trans)
    check("p = $p: P^1 dH2/du = M^-1 S u, the projected translation flow", e < TOL,
        @sprintf("rel. error %.2e", e))
    pi_ux = M \ (Φ0 * (W .* fieldat(s, d, 1)))
    e = maximum(abs, trans - pi_ux) / maximum(abs, trans)
    check("p = $p: i.e. int phi_i u_h,t = int phi_i u_h,x", e < TOL,
        @sprintf("rel. error %.2e", e))
    g2, g = grad_H2(s, d), mass_grad(s)
    check("p = $p: it conserves H2 and the mass exactly",
        abs(dot(g2, trans)) < 1e-11 && abs(dot(g, trans)) < 1e-11,
        @sprintf("%.2e, %.2e", abs(dot(g2, trans)), abs(dot(g, trans))))
end

println("\n  P^2 dH1/du is the fifth-order member, d_x ( 10u^3 + 10uu_xx + 5u_x^2")
println("  + u_xxxx ), again reproduced at about 2p:")
for (p, ns) in ((3, (16, 32, 64)), (4, (12, 16, 24)))
    errs = Float64[]
    for n in ns
        s, d = setup(n; p, f = sin)
        M = Matrix(mass_matrix(s))
        W = quadrature_weights(s)
        Φ0 = basis_values(s, 0)
        X = quadrature_nodes(s)
        f5 = poisson_matrix(kdv_bracket_2(s), d) * grad_H1(s, d)
        ex = M \ (Φ0 * (W .* KDV5_RHS.(X)))
        push!(errs, sqrt(dot(W, (transpose(Φ0) * (f5 - ex)) .^ 2)))
    end
    o = rates(errs, ns)
    println("      p = $p:  " * e3(errs) * "    orders $(o2(o))")
    check("p = $p: P^2 dH1/du converges to the fifth-order flow at about 2p = $(2p)",
        minimum(o) > 2p - 0.6, "observed $(o2(o))")
end

println("\n  the first rung of Magri's recursion holds EXACTLY at finite N, the")
println("  second only to O(h^2p) -- the same signature as the Jacobi failure:")
for (p, ns) in ((3, (16, 32, 64)), (4, (12, 16, 24)))
    errs = Float64[]
    for n in ns
        s, d = setup(n; p, f = sin)
        a = poisson_matrix(kdv_bracket_2(s), d) * grad_H1(s, d)
        b = poisson_matrix(kdv_bracket_1(s), d) * grad_H3(s, d)
        push!(errs, maximum(abs, a - b) / maximum(abs, a))
    end
    o = rates(errs, ns)
    println("      p = $p:  " * e3(errs) * "    orders $(o2(o))")
    check("p = $p: P^2 dH1/du =/= P^1 dH3/du at finite N", errs[1] > 1e-7,
        @sprintf("rel. difference %.2e at N = %d", errs[1], ns[1]))
    check("p = $p: but the mismatch converges at about 2p = $(2p)", minimum(o) > 2p - 0.8,
        "observed $(o2(o))")
end

# --------------------------------------------------------------------------
header("7. time discretisation")

"Newton on res(x) = 0 with a numerical Jacobian; the systems are small."
function solve_step(res, x0; tol = 1e-14, itmax = 60)
    x = copy(x0)
    for _ in 1:itmax
        r = res(x)
        maximum(abs, r) < tol && break
        J = zeros(length(x), length(x))
        eps = 1e-7
        for j in eachindex(x)
            xp = copy(x)
            xp[j] += eps
            J[:, j] = (res(xp) - r) ./ eps
        end
        x = x - J \ r
    end
    x
end

const NT, DT = 200, 1e-4
for p in (2, 3)
    s, d0 = setup(20; p)
    M = Matrix(mass_matrix(s))
    W = quadrature_weights(s)
    S = skew_S(s)
    H1(dd) = dot(W, fieldat(s, dd, 1) .^ 2 - 2 .* fieldat(s, dd, 0) .^ 3) / 2
    H2(dd) = dot(W, fieldat(s, dd, 0) .^ 2) / 2
    C0(dd) = dot(W, fieldat(s, dd, 0))
    scale = dot(W, abs.(fieldat(s, d0, 0)))          # C0 itself is ~0 for a zero-mean field

    # implicit midpoint on the second flow
    d = copy(d0)
    for _ in 1:NT
        local dprev = d
        d = solve_step(y -> y - dprev - DT .* flow2(s, 0.5 .* (dprev + y)), dprev)
    end
    e2 = abs(H2(d) - H2(d0)) / abs(H2(d0))
    m2 = abs(C0(d) - C0(d0)) / scale
    check("p = $p: implicit midpoint on flow 2 conserves H2 exactly", e2 < 1e-12,
        @sprintf("relative drift %.2e", e2))
    check("p = $p: ... and the mass", m2 < 1e-12, @sprintf("normalised drift %.2e", m2))

    # two-point-Gauss average-vector-field method on the first flow
    sg = (0.5 - sqrt(3.0) / 6.0, 0.5 + sqrt(3.0) / 6.0)
    d = copy(d0)
    for _ in 1:NT
        local dprev = d
        function avf(y)
            gbar = sum(grad_H1(s, dprev + si .* (y - dprev)) for si in sg) / 2
            y - dprev - DT .* (M \ (S * (M \ gbar)))
        end
        d = solve_step(avf, dprev)
    end
    e1 = abs(H1(d) - H1(d0)) / abs(H1(d0))
    m1 = abs(C0(d) - C0(d0)) / scale
    check("p = $p: the AVF method on flow 1 conserves the cubic H1 exactly", e1 < 1e-10,
        @sprintf("relative drift %.2e", e1))
    check("p = $p: ... and the mass, since P^1 g = 0", m1 < 1e-12,
        @sprintf("normalised drift %.2e", m1))

    # explicit Euler, as a negative control
    de, dd = copy(d0), copy(d0)
    for _ in 1:NT
        de = de + DT .* flow2(s, de)
        dd = dd + DT .* first(flow1(s, dd))
    end
    check("p = $p: explicit Euler drifts on both flows",
        abs(H2(de) - H2(d0)) / abs(H2(d0)) > 1e3 * max(e2, 1e-16) &&
            abs(H1(dd) - H1(d0)) / abs(H1(d0)) > 1e3 * max(e1, 1e-16),
        @sprintf("H2 drift %.2e, H1 drift %.2e",
            abs(H2(de) - H2(d0)) / abs(H2(d0)),
            abs(H1(dd) - H1(d0)) / abs(H1(d0))))
end

summary("verify_kdv_semidiscrete.jl")
