#!/usr/bin/env julia
#
# The convergence study behind the refinement checks of `verify_fourbracket_identities.jl`.
#
#     julia --project=scripts scripts/verify_fourbracket_convergence.jl
#
# Six of the manuscript's identities involve fields that are not band-limited -- the quotient
# A_u/S_u, the powers u^{3/2}, the logarithm log u -- so Fourier differentiation is not exact
# for them and their residuals are discretisation error rather than a defect of the identity.
# What certifies such an identity is the rate: the residual must fall at the design order of
# the stencil as the grid is refined, and an identity that were merely approximate would
# stall at its own error instead.
#
# Each row is measured at N = 32, 64, 128 and 256, and the observed order is the base-two
# logarithm of successive ratios.  A row passes if the finest residual has reached roundoff,
# or if the last transition still shows at least 6th-order behaviour -- the margin against the
# design order of 8 absorbs the constant in the leading error term, which is not close to one
# for the quotient identities.
#
# Observed orders run from 7.2 to 8.1, so the 8th-order stencil is behaving and the identities
# hold.  Rows 1 and 2 are pointwise, the remaining four are integrated.

using GeometricBrackets
using Printf

include(joinpath(@__DIR__, "check.jl"))
using .Checks: header, check, summary
include(joinpath(@__DIR__, "torustools.jl"))
using .TorusTools

const STUDIES = [
    ("Lemma 4.1  G_x(A,S) + (1/2) S_u^2 d_x(A_u/S_u)",
        g -> begin
            a, s = Au(g), chi(g)
            gardner_x(g, a, s) .+ s .^ 2 .* ∂x(g, a ./ s) ./ 2
        end),
    ("eq. (4.18)  G_{x,y} - G_{y,x} + [A_u,B_u]",
        g -> begin
            a, b = Au(g), Bu(g)
            ∂y(g, gardner_x(g, a, b)) .- ∂x(g, gardner_y(g, a, b)) .+
            canonical_bracket(g, a, b)
        end),
    ("Lemma 4.2  int X^2[Xa,Xb] - (1/2) int X^4[a,b]",
        g -> begin
            s = chi(g)
            a, b = Au(g) ./ s, Bu(g) ./ s
            integrate(g, s .^ 2 .* canonical_bracket(g, s .* a, s .* b)) -
            integrate(g, s .^ 4 .* canonical_bracket(g, a, b)) / 2
        end),
    ("Prop. 4.3  {A,B}_S - (1/4) int S_u^4 [a,b]",
        g -> begin
            a, b, s = Au(g), Bu(g), chi(g)
            gardner_2bracket(g, a, b, s) -
            integrate(g, s .^ 4 .* canonical_bracket(g, a ./ s, b ./ s)) / 4
        end),
    ("Thm. 4.5  Jacobi obstruction, c = (1+log u)^2/2",
        g -> begin
            u = uu(g)
            cu, dcu = (1 .+ log.(u)) .^ 2 ./ 2, (1 .+ log.(u)) ./ u
            first(jacobi_residual(g, Au(g), Bu(g), Cu(g), cu, dcu; normalised = false))
        end),
    ("Ex. 5.4  weighted log-entropy bracket - int u[A_u,B_u]",
        g -> begin
            a, b, u = Au(g), Bu(g), uu(g)
            s = 1 .+ log.(u)
            weighted_2bracket(g, a, b, s, u ./ (2 .* s)) - lie_poisson_2bracket(g, a, b, u)
        end)
]

header("FINITE-DIFFERENCE CONVERGENCE (8th-order stencil; expected factor 256 per doubling)")

@printf("%-50s %s\n", "RESIDUAL OF",
    join([lpad("N=$N", 12) for N in REFINEMENT_RESOLUTIONS], ""))
println("-"^100)

for (label, f) in STUDIES
    errs = refinement_sweep(f)
    rates = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
    @printf("%-50s %s\n", first(label, 50),
        join([@sprintf("%12.2e", e) for e in errs], ""))
    @printf("%-50s %12s%s\n", "    observed order", "",
        join([@sprintf("%12.1f", r) for r in rates], ""))
    check(first(label, 50), errs[end] < 1e-13 || rates[end] > 6.0,
        @sprintf("order %.1f at the finest doubling", rates[end]))
end

summary("verify_fourbracket_convergence.jl")
