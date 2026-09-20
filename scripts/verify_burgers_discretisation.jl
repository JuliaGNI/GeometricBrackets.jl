#!/usr/bin/env julia
#
# The discrete Burgers bracket: structure, consistency, and conservation.
#
#     julia --project=scripts scripts/verify_burgers_discretisation.jl
#
# For a periodic Lagrange finite element basis on [0, 2pi] this assembles
#
#     M_ij = int phi_i phi_j ,
#     D_kl = int ( phi_k phi_l' - phi_l phi_k' ) ,
#     K    = M^-1 D M^-1 ,
#     J_ij(u) = sqrt(u_i) K_ij sqrt(u_j) ,
#
# and checks
#
#   1. K is antisymmetric of rank N-1, and M 1 spans its kernel;
#   2. C(u) = 2 sum_i (int phi_i) sqrt(u_i) is an EXACT Casimir of J, and a consistent
#      quadrature of the continuous Casimir 2 int sqrt(u);
#   3. J is second-order consistent: J grad H reproduces the nodal values of 3 u u_x;
#   4. the induced integrator in ubar_i = sqrt(u_i), where the bracket is the CONSTANT K/4,
#      conserves the Casimir exactly (it is linear there) and the energy without drift.
#
# The assembly comes from the package: `LagrangeSpace`, `burgers_bracket` and
# `burgers_casimir` are the same objects the test suite exercises, so this script diagnoses
# what the package actually ships rather than a copy of it.

using GeometricBrackets
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary

"A smooth, strictly positive periodic profile."
u_exact(x) = 2.0 + sin(x) + 0.3 * cos(2.0 * x)
dudx_exact(x) = cos(x) - 0.6 * sin(2.0 * x)

"(x, M, K, I, L) for the periodic Lagrange space of degree p on ne elements."
function assemble(p, ne)
    s = LagrangeSpace(p, ne)
    Minv = inverse_mass_matrix(s)
    S = derivative_matrix(s)
    return nodes(s), Matrix(mass_matrix(s)), Minv * (S - S') * Minv,
    basis_integrals(s), domainlength(s)
end

# Quoted like Python's list-of-strings repr, so the tables diff exactly against the script
# this replaces. Once the Python is gone the quotes can go with it.
sci(v) = "[" * join(["'" * (@sprintf("%.2e", x)) * "'" for x in v], ", ") * "]"
fx(v) = "[" * join(["'" * (@sprintf("%.2f", x)) * "'" for x in v], ", ") * "]"

header("1. Structure of K = M^-1 D M^-1")

# An antisymmetric matrix has EVEN rank, so the corank of K is 1 for odd N and at least 2 for
# even N. The physical kernel vector is always M 1, giving the Casimir of Section 2; for even
# N there is one further, spurious kernel vector -- the Nyquist / sawtooth mode -- and hence
# a spurious discrete Casimir with no continuum counterpart.
for (p, ne) in ((1, 25), (2, 13), (1, 24), (2, 12))
    x, M, K, I, L = assemble(p, ne)
    N = length(x)
    tag = @sprintf("P%d, %2d elements, N = %2d (%s)", p, ne, N, isodd(N) ? "odd " : "even")
    a = maximum(abs, K + K')
    check("$tag: K is antisymmetric", a < 1e-10, @sprintf("max|K + K^T| = %.2e", a))
    r = rank(K; atol = 1e-9)
    expected = isodd(N) ? N - 1 : N - 2
    check("$tag: rank K = $expected", r == expected, "rank = $r")
    n = M * ones(N)
    km = maximum(abs, K * n)
    check("$tag: K (M 1) = 0", km < 1e-9, @sprintf("max|K M 1| = %.2e", km))
    check("$tag: n_i = int phi_i, i.e. M 1 is the vector of basis integrals",
        maximum(abs, n - I) < 1e-10)
    check("$tag: sum_i int phi_i = |Omega|", abs(sum(I) - L) < 1e-10)
    if iseven(N)
        saw = [(-1.0)^i for i in 0:(N - 1)]
        ks = maximum(abs, K * (M * saw))
        check("$tag: the extra kernel vector is the sawtooth mode (spurious Casimir)",
            ks < 1e-9, @sprintf("max|K M saw| = %.2e", ks))
    end
end

header("2. C(u) = 2 sum_i (int phi_i) sqrt(u_i) is an exact Casimir of J")

for (p, ne) in ((1, 24), (2, 12))
    x, M, K, I, L = assemble(p, ne)
    u = u_exact.(x)
    su = sqrt.(u)
    J = su .* K .* su'
    gradC = I ./ su                      # ∂/∂u_i of 2 Σ_i I_i √u_i
    tag = "P$p, $ne elements"
    jc = maximum(abs, J * gradC)
    check("$tag: J grad C = 0", jc < 1e-9, @sprintf("max|J grad C| = %.2e", jc))
    Cd = 2.0 * dot(I, su)
    xs = range(0, L; length = 20001)
    ys = sqrt.(u_exact.(xs))
    Cc = 2.0 * (sum(ys) - (ys[1] + ys[end]) / 2) * step(xs)     # trapezoid
    check("$tag: C approximates 2 int sqrt(u)", abs(Cd - Cc) / abs(Cc) < 1e-3,
        @sprintf("discrete = %.8f, continuous = %.8f, rel. err = %.2e",
            Cd, Cc, abs(Cd - Cc) / abs(Cc)))
end

header("3. Consistency: J grad H reproduces 3 u u_x, second order")

for p in (1, 2)
    errs, hs = Float64[], Float64[]
    for ne in (16, 32, 64, 128)
        x, M, K, I, L = assemble(p, ne)
        u = u_exact.(x)
        su = sqrt.(u)
        J = su .* K .* su'
        udot = J * (M * u)                    # H = ½ u·M·u ≈ ½ ∫u²
        ex = 3.0 .* u_exact.(x) .* dudx_exact.(x)
        push!(errs, maximum(abs, udot - ex) / maximum(abs, ex))
        push!(hs, L / ne)
    end
    rates = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1])
             for i in 1:(length(errs) - 1)]
    println("   P$p: rel. errors $(sci(errs))")
    println("   P$p: observed rates $(fx(rates))")
    check("P$p: J grad H converges to 3 u u_x at order >= 2", rates[end] > 1.8,
        @sprintf("finest rate = %.2f", rates[end]))
end

header("4. Poisson integration in the variables ubar_i = sqrt(u_i)")

# In ubar the bracket is the CONSTANT tensor K/4, so implicit midpoint is a Poisson
# integrator: it conserves the linear Casimir C = 2 I·ubar exactly, and the energy without
# secular drift.
#
# The transformation and the factor move together: ubar = sqrt(u) goes with K/4 and
# dH/dubar_i = 2 ubar_i (M u)_i, while ubar = 2 sqrt(u) goes with K and no factor. The Python
# this replaces once paired ubar = 2 sqrt(u) WITH K/4, making the flow four times too slow.
# Nothing in this section could see it -- a constant rescaling of a Poisson vector field
# preserves every Casimir, every energy bound and every convergence rate -- so the only
# witness is the comparison against the pushforward of the u-space field in section 3.
let p = 2, ne = 16
    x, M, K, I, L = assemble(p, ne)
    u0 = u_exact.(x)
    ub0 = sqrt.(u0)

    energy(ub) = 0.5 * dot(ub .^ 2, M * (ub .^ 2))
    grad_energy(ub) = 2.0 .* ub .* (M * (ub .^ 2))     # dH/dūᵢ = 2 ūᵢ (M u)ᵢ

    function step!(ub, dt)
        y = copy(ub)
        for _ in 1:100
            mid = 0.5 .* (ub .+ y)
            yn = ub .+ dt * 0.25 .* (K * grad_energy(mid))
            if maximum(abs, yn - y) < 1e-14
                return yn
            end
            y = yn
        end
        return y
    end

    H0, C0 = energy(ub0), 2.0 * dot(I, ub0)
    T = 2.0
    dts, dHs, dCs = Float64[], Float64[], Float64[]
    ub = copy(ub0)
    for nsteps in (250, 500, 1000, 2000)
        dt = T / nsteps
        ub = copy(ub0)
        hmax = cmax = 0.0
        for _ in 1:nsteps
            ub = step!(ub, dt)
            hmax = max(hmax, abs(energy(ub) - H0) / abs(H0))
            cmax = max(cmax, abs(2.0 * dot(I, ub) - C0) / abs(C0))
        end
        push!(dts, dt)
        push!(dHs, hmax)
        push!(dCs, cmax)
    end

    println("   dt                    $(sci(dts))")
    println("   max rel. |H - H0|     $(sci(dHs))")
    println("   max rel. |C - C0|     $(sci(dCs))")
    rates = [log(dHs[i] / dHs[i + 1]) / log(dts[i] / dts[i + 1])
             for i in 1:(length(dts) - 1)]
    println("   energy error rates    $(fx(rates))")

    check("the Casimir C = 2 I . ubar is conserved to round-off for every dt",
        maximum(dCs) < 1e-12, @sprintf("max rel. drift = %.2e", maximum(dCs)))
    check("the energy error is bounded and second order in dt", rates[end] > 1.8,
        @sprintf("finest rate = %.2f", rates[end]))
    check("u stays positive over the integration", minimum(ub .^ 2) > 0)
end

summary("verify_burgers_discretisation.jl")
