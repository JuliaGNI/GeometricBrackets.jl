#!/usr/bin/env julia
#
# Proposition 2.1: the metriplectic two-bracket induced by a Kulkarni-Nomizu product of two
# symmetric positive semi-definite tensors is itself positive semi-definite.
#
#     julia --project=scripts scripts/verify_fourbracket_metriplectic.jl
#
# The proof in the manuscript is analytic -- Cauchy-Schwarz followed by AM-GM. This script is
# an independent Monte-Carlo cross-check of it, in three parts:
#
#   1. That `metriplectic_bracket` implements equation (2.6). The closed form it evaluates in
#      O(n^2) is checked against the O(n^4) contraction of the four-index tensor
#      `kulkarni_nomizu` builds from the definition, so that what follows is known to be
#      testing the proposition rather than a convenient rearrangement of it.
#
#   2. The proposition itself, swept over dimension and over every combination of ranks --
#      full, rank-one, and deficient on either side. The singular cases are the point: an
#      inequality chain is likeliest to fail where its terms degenerate, and a sweep over
#      definite tensors only would never reach them.
#
#   3. That the hypothesis is not decoration. Drop semi-definiteness for a merely symmetric
#      indefinite sigma and negative values must appear -- otherwise the proposition would be
#      vacuous and the sweep above would be evidence for nothing.
#
# Finite-dimensional and unlike the rest of the manuscript's material: no grid, no fields.
# Both seeds are fixed, so a failure is reproducible.

using GeometricBrackets
using Printf
using Random

include(joinpath(@__DIR__, "check.jl"))
using .Checks: header, check, summary

"Random symmetric positive semi-definite `n`-by-`n` matrix of rank `r`."
random_psd(n, r) = (P = randn(n, r); P * P')

header("1. the closed form is the contraction of the Kulkarni-Nomizu tensor")

let worst = 0.0
    Random.seed!(4711)
    for n in (2, 3, 4, 5), _ in 1:20

        σ, μ = random_psd(n, n), random_psd(n, n)
        α, h = randn(n), randn(n)
        closed = metriplectic_bracket(σ, μ, α, h)
        tensor = metriplectic_bracket(kulkarni_nomizu(σ, μ), α, h)
        worst = max(worst, abs(closed - tensor) / max(abs(closed), 1.0))
    end
    check("eq. (2.6) closed form = R^{ijkl} contracted, n = 2..5", worst < 1e-12,
        @sprintf("worst relative difference %.2e", worst))
end

header("2. Proposition 2.1: positive semi-definiteness, swept over dimension and rank")

const TRIALS = 40_000

Random.seed!(20260820)
for n in (2, 3, 5, 8)
    for (rs, rm) in ((n, n), (1, n), (n, 1), (1, 1), (max(n - 1, 1), max(n - 2, 1)))
        worst = Inf
        for _ in 1:TRIALS
            worst = min(worst, metriplectic_bracket(random_psd(n, rs), random_psd(n, rm),
                randn(n), randn(n)))
        end
        check(
            @sprintf("n = %d, rank(sigma) = %d, rank(mu) = %d, %d trials", n, rs, rm,
                TRIALS),
            worst > -1e-8, @sprintf("min value %11.3e", worst))
    end
end

header("3. the hypothesis is necessary: an indefinite sigma must break positivity")

let found = nothing, ntried = 0
    Random.seed!(1)
    for i in 1:100_000
        A = randn(4, 4)
        σ = A + A'                       # symmetric, and indefinite with probability one
        v = metriplectic_bracket(σ, random_psd(4, 4), randn(4), randn(4))
        ntried = i
        if v < -1e-6
            found = v
            break
        end
    end
    check("an indefinite sigma does produce negative values, n = 4", found !== nothing,
        found === nothing ? "none in 100000 draws -- UNEXPECTED" :
        @sprintf("%.3e after %d draws", found, ntried))
end

summary("verify_fourbracket_metriplectic.jl")
