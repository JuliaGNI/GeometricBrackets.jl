#!/usr/bin/env julia
#
# The singularity of the log-entropy weight (Section 5.1, Example 5.4).
#
#     julia --project=scripts scripts/verify_fourbracket_log_entropy.jl
#
# For S = int s(u), the weighted four-bracket of Section 5.1 generates
# {A,B}_S = 2 int Phi(u) [A_u,B_u] with Phi' = omega s' s''. Demanding the Lie-Poisson
# bracket, Phi = u/2, forces
#
#       omega = 1 / (2 s' s'') ,
#
# which for s = u log u is omega = u / (2 (1 + log u)), with a simple pole at u = 1/e where s'
# vanishes. Three claims settle whether that pole can be removed:
#
#   1. The pole is confined to the four-bracket. In the induced two-bracket the combination
#      omega * S_u * s'' is identically 1/2, so the pole is cancelled exactly by the zero of
#      S_u, and the Lie-Poisson bracket it generates is regular at u = 1/e -- even for fields
#      whose range straddles 1/e. The four-bracket, with four independent arguments, has no
#      such zero available and genuinely diverges. Refining the grid is what separates the two:
#      a genuine pole grows without bound as the grid resolves the curve u = 1/e, whereas a
#      merely large value settles.
#
#   2. No weight regular at the critical level can do the job. If s'(u*) = 0 and omega is C^1
#      near u*, the induced structure function obeys c'(u*) = 0, whereas the Lie-Poisson
#      bracket needs c' = 1 identically. So the pole is forced, not an artefact of a poor
#      choice of weight.
#
#   3. It can nonetheless be moved out of the way. int u is a Casimir of every bracket in the
#      family, so S may be shifted to s_alpha = u log u + alpha u without changing the bracket
#      generated. The critical level moves to u* = exp(-(1+alpha)), so on a state space whose
#      fields take values in a compact subinterval of (0, infinity), some alpha makes the
#      weight smooth on the whole range.
#
# This is the same obstruction the discrete Lie-Poisson brackets meet when the Casimir is
# prescribed -- see `verify_burgers_entropy_casimir.jl`, which finds the identical critical
# level u = 1/e from the structure-constant side -- seen here from the four-bracket side.

using GeometricBrackets
using Printf

include(joinpath(@__DIR__, "check.jl"))
using .Checks: header, check, check_exact, check_refined, summary, relerr
include(joinpath(@__DIR__, "torustools.jl"))
using .TorusTools

"The critical level of the unshifted entropy s = u log u, where s' vanishes."
const UCRIT = exp(-1.0)

# The entropy density s_alpha = u log u + alpha u, its derivatives, and the weight they force.
sprime(u, α) = 1 + α + log(u)
sdprime(u) = 1 / u
weight(u, α) = 1 / (2 * sprime(u, α) * sdprime(u))      # = u / (2 (1 + alpha + log u))

"The Section 5.1 weighted two-bracket for the shifted entropy, evaluated with the bare weight."
entropy_2bracket(g, a, b, u, α) = weighted_2bracket(g, a, b, sprime.(u, α), weight.(u, α))

# A field whose range straddles the critical level u = 1/e = 0.3679:
# u = 0.40 + 0.15 sin x + 0.10 cos y ranges over [0.15, 0.65]. Contrast `uu`, whose range
# [2.1, 4.0] misses it, and which section 3 below uses for exactly that reason.
ustraddle(g) = sample(g, (x, y) -> 0.40 + 0.15sin(x) + 0.10cos(y))

# The pointwise Lie-Poisson integrand, (1/2)(A_u [B_u,u] - B_u [A_u,u]). Its integral is
# int u[A_u,B_u] by cyclicity, but it is the integrand that has to stay bounded here.
function lp_density(g, a, b, u)
    (a .* canonical_bracket(g, b, u) .- b .* canonical_bracket(g, a, u)) ./ 2
end

header("1. the pole cancels in the two-bracket but not in the four-bracket")

check_exact("omega(u) * S_u * s''(u) = 1/2 identically  [rem:pole-cancels]",
    exact_residual(g -> begin
        u = ustraddle(g)
        weight.(u, 0.0) .* sprime.(u, 0.0) .* sdprime.(u) .- 0.5
    end); atol = 1e-14)

# log u is not band-limited, so this pointwise comparison converges rather than vanishing.
check_refined("2-bracket integrand is regular across u = 1/e  [rem:pole-cancels]",
    refined_residuals(g -> begin
        u = ustraddle(g)
        a, b = Au(g), Bu(g)
        weighted_2bracket_density(g, a, b, sprime.(u, 0.0), weight.(u, 0.0)) .-
        lp_density(g, a, b, u)
    end)...)

check_refined("2-bracket = int u[A_u,B_u] with u straddling 1/e  [rem:pole-cancels]",
    refined_residuals(g -> begin
        u = ustraddle(g)
        a, b = Au(g), Bu(g)
        relerr(entropy_2bracket(g, a, b, u, 0.0), lie_poisson_2bracket(g, a, b, u))
    end)...)

println()
println("       behaviour under refinement (u ranges over [0.15, 0.65], straddling 1/e):")
@printf("       %6s %14s %20s %20s\n", "N", "max|omega|", "max|4-bracket|",
    "max|2-bracket|")
let growth = Float64[], bounded = Float64[]
    for N in REFINEMENT_RESOLUTIONS
        g = finite_difference_grid(N)
        u = ustraddle(g)
        a, b, c, d = Au(g), Bu(g), Cu(g), Du(g)
        w = weight.(u, 0.0)
        four = maximum(abs, weighted_4bracket_density(g, a, b, c, d, w))
        two = maximum(abs, lp_density(g, a, b, u))
        push!(growth, four)
        push!(bounded, two)
        @printf("       %6d %14.3e %20.3e %20.3e\n", N, maximum(abs, w), four, two)
    end
    println()
    check("4-bracket integrand DIVERGES under refinement  [rem:pole-cancels]",
        growth[end] > 3 * growth[1],
        @sprintf("grew by %.2fx from N=%d to N=%d", growth[end] / growth[1],
            first(REFINEMENT_RESOLUTIONS), last(REFINEMENT_RESOLUTIONS)))
    check("2-bracket integrand stays bounded under refinement  [rem:pole-cancels]",
        maximum(bounded) / minimum(bounded) < 1.2,
        @sprintf("spread %.3fx over the same range", maximum(bounded) / minimum(bounded)))
end

header("2. no weight regular at the critical level reproduces the Lie-Poisson bracket")

# c(u) = 2 Phi(u) with Phi' = omega s' s''. At the critical level s'(u*) = 0, so
# c'(u*) = 2 omega(u*) s'(u*) s''(u*) = 0 for every bounded omega, whereas Lie-Poisson needs 1.
let α = 0.0, worst = 0.0
    regular_weights = [
        ("omega = 1", u -> 1.0),
        ("omega = u", u -> u),
        ("omega = u^2 + 3", u -> u^2 + 3),
        ("omega = exp(u)", u -> exp(u)),
        ("omega = 1/(1+u)", u -> 1 / (1 + u))
    ]
    println()
    println("       c'(u*) at the critical level u* = 1/e, for weights regular there:")
    for (name, w) in regular_weights
        cprime = 2 * w(UCRIT) * sprime(UCRIT, α) * sdprime(UCRIT)
        @printf("       %-20s  c'(u*) = %12.3e   (Lie-Poisson needs 1)\n", name, cprime)
        worst = max(worst, abs(cprime))
    end
    println()
    check("c'(u*) = 0 for every regular weight  [prop:no-regular-weight]", worst < 1e-14,
        @sprintf("largest of the five is %.2e, against the 1 required", worst))
end

header("3. the Casimir shift moves the critical level out of the range of u")

# With s_alpha = u log u + alpha u the critical level is u* = exp(-(1+alpha)). The field `uu`
# ranges over [2.1, 4.0], so alpha = -2 puts u* = e inside that range and every other alpha
# tested leaves it outside.
println()
println("       critical level u* = exp(-(1+alpha)) vs. the range of u = [2.1, 4.0]:")
for α in (0.0, -1.0, -2.0, -3.0, -4.0)
    ustar = exp(-(1 + α))
    @printf("       alpha = %5.1f   u* = %8.4f   %s\n", α,
        ustar,
        2.1 <= ustar <= 4.0 ? "INSIDE the range -> weight singular" :
        "outside the range -> weight smooth")
end
println()

for α in (-3.0, -4.0, -5.0)
    check_refined(
        "shifted entropy alpha = $(α) still gives LP  [rem:moving-critical-level]",
        refined_residuals(g -> begin
            u = uu(g)
            relerr(entropy_2bracket(g, Au(g), Bu(g), u, α), lie_poisson_2bracket(g, Au(g), Bu(g), u))
        end)...)
    let g = finite_difference_grid(FINE_N)
        wmax = maximum(abs, weight.(uu(g), α))
        check("  and its weight is bounded, max|omega| < 10  [rem:moving-critical-level]",
            wmax < 10.0, @sprintf("max|omega| = %.2e", wmax))
    end
end

summary("verify_fourbracket_log_entropy.jl")
