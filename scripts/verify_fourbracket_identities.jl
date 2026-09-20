#!/usr/bin/env julia
#
# Every displayed identity of Sections 3 to 6 of `poisson-brackets-from-four-brackets.tex`.
#
#     julia --project=scripts scripts/verify_fourbracket_identities.jl
#
# Each check carries the manuscript's own label in brackets, so a line here names the
# equation, lemma or proposition it settles.  Sections, in the order below:
#
#   1. Section 4.2, the auxiliary identities: cyclicity of the canonical bracket, the Gardner
#      operator as a flux plus a total derivative, its integration by parts, and the curl
#      identity G_{x,y} - G_{y,x} = -[A_u,B_u] that makes the reduction possible.
#
#   2. Section 4.1, reduction to Lie-Poisson form: Lemma 4.1 in product and quotient form,
#      Lemma 4.2, Proposition 4.3 in both normalisations, and the two structure functions that
#      reach int u[A_u,B_u] -- exactly, at S_u = sqrt(2u).
#
#   3. Section 4.2, the Jacobi identity: the pointwise Plücker relation that collapses the
#      obstruction, then Theorem 4.5 for five unrelated structure functions.
#
#   4. Section 4.3, the Casimir invariants.
#
#   5. Section 5, the symmetric/antisymmetric bracket: the same reduction without the factor
#      of 1/2, so that S_u = sqrt(u) already suffices.
#
#   6. Section 5.1, the weighted brackets, including the log-entropy weight of Example 5.4.
#
#   7. Section 6, the two symmetric operators: the bracket they build vanishes identically,
#      and the alternative pairing gives int S_ux S_uy [A_u,B_u] instead.
#
#   8. Section 3, Lemma 3.1: the two antisymmetry conditions, and that each family satisfies
#      exactly one of them.  The violations are asserted as sharply as the identities.
#
# Band-limited claims are settled on a spectral grid at roundoff; claims involving u^{3/2},
# log u or a quotient A_u/S_u are settled by refinement, since Fourier differentiation is not
# exact for those.  `verify_fourbracket_convergence.jl` measures the orders behind that.

using GeometricBrackets
using Printf

include(joinpath(@__DIR__, "check.jl"))
using .Checks: header, check, check_exact, check_refined, summary, relerr, normerr
include(joinpath(@__DIR__, "torustools.jl"))
using .TorusTools

header("1. Section 4.2: the auxiliary identities")

check_exact("cyclicity  int A[B,S] = int S[A,B]  [eq:cyclicity]",
    exact_residual(g -> begin
        a, b, s = Au(g), Bu(g), chi(g)
        relerr(integrate(g, a .* canonical_bracket(g, b, s)),
            integrate(g, s .* canonical_bracket(g, a, b)))
    end))

check_exact("G_x = A_u B_ux - (1/2) d_x(A_u B_u)  [eq:gardner-total-derivative]",
    exact_residual(g -> begin
        a, b = Au(g), Bu(g)
        gardner_x(g, a, b) .- (a .* ∂x(g, b) .- ∂x(g, a .* b) ./ 2)
    end))

# Both sides of eq. (4.15) happen to vanish for these test fields, so the error is measured
# against the size of the integrand rather than of the integral.
check_exact("int F G_x = int F A_u B_ux + (1/2) int F_x A_u B_u  [eq:gardner-ibp]",
    exact_residual(g -> begin
        a, b, F = Au(g), Bu(g), Cu(g)
        normerr(integrate(g, F .* gardner_x(g, a, b)),
            integrate(g, F .* a .* ∂x(g, b)) + integrate(g, ∂x(g, F) .* a .* b) / 2,
            integrate(g, abs.(F .* gardner_x(g, a, b))))
    end))

check_exact("G_{x,y} - G_{y,x} = -[A_u,B_u]  [eq:gardner-curl]",
    exact_residual(g -> begin
        a, b = Au(g), Bu(g)
        ∂y(g, gardner_x(g, a, b)) .- ∂x(g, gardner_y(g, a, b)) .+
        canonical_bracket(g, a, b)
    end))

header("2. Section 4.1: reduction to Lie-Poisson form")

# Lemma 4.1 in product form is exact algebra, independent of the grid.
check_exact("2 G_x(A,S) = A_u S_ux - S_u A_ux  [lem:gardner-quotient]",
    exact_residual(g -> begin
        a, s = Au(g), chi(g)
        2 .* gardner_x(g, a, s) .- (a .* ∂x(g, s) .- s .* ∂x(g, a))
    end))

# Lemma 4.1 as stated: A_u/S_u is not band-limited, so this one converges rather than vanishes.
check_refined("G_x(A,S) = -(1/2) S_u^2 d_x(A_u/S_u)  [lem:gardner-quotient]",
    refined_residuals(g -> begin
        a, s = Au(g), chi(g)
        gardner_x(g, a, s) .+ s .^ 2 .* ∂x(g, a ./ s) ./ 2
    end)...)

check_refined("int X^2[Xa,Xb] = (1/2) int X^4[a,b]  [lem:quotient-identity]",
    refined_residuals(g -> begin
        s = chi(g)
        a, b = Au(g) ./ s, Bu(g) ./ s
        relerr(integrate(g, s .^ 2 .* canonical_bracket(g, s .* a, s .* b)),
            integrate(g, s .^ 4 .* canonical_bracket(g, a, b)) / 2)
    end)...)

check_refined("{A,B}_S = (1/4) int S_u^4 [A_u/S_u, B_u/S_u]  [prop:gardner-reduction]",
    refined_residuals(g -> begin
        a, b, s = Au(g), Bu(g), chi(g)
        relerr(gardner_2bracket(g, a, b, s),
            integrate(g, s .^ 4 .* canonical_bracket(g, a ./ s, b ./ s)) / 4)
    end)...)

check_exact("{A,B}_S = (1/2) int S_u^2 [A_u, B_u]  [prop:gardner-reduction]",
    exact_residual(g -> begin
        a, b, s = Au(g), Bu(g), chi(g)
        relerr(gardner_2bracket(g, a, b, s),
            integrate(g, s .^ 2 .* canonical_bracket(g, a, b)) / 2)
    end))

check_refined(
    "S = (2/3) int u^{3/2}  ->  (1/2) int u[A_u,B_u]  [eq:gardner-structure-function]",
    refined_residuals(g -> begin
        a, b, u = Au(g), Bu(g), uu(g)
        relerr(gardner_2bracket(g, a, b, sqrt.(u)), lie_poisson_2bracket(g, a, b, u) /
                                                    2)
    end)...)

check_refined("S_u = sqrt(2u)  ->  int u[A_u,B_u] EXACTLY  [eq:lie-poisson]",
    refined_residuals(g -> begin
        a, b, u = Au(g), Bu(g), uu(g)
        relerr(gardner_2bracket(g, a, b, sqrt.(2 .* u)), lie_poisson_2bracket(g, a, b, u))
    end)...)

header("3. Section 4.2: the Jacobi identity")

check_exact("Plücker relation, pointwise  [lem:plucker]",
    exact_residual(g -> plucker_residual(g, Au(g), Bu(g), Cu(g), uu(g))))

# Jacobi for {A,B}_c = int c(u)[A_u,B_u], on linear functionals A = int(alpha u), for which
# the second variations vanish identically and the residual is exactly the obstruction (4.22).
for (name, cfun, cprime) in [
    ("c(u) = u", u -> u, u -> 1.0),
    ("c(u) = u^3", u -> u^3, u -> 3u^2),
    ("c(u) = (1+log u)^2 / 2", u -> (1 + log(u))^2 / 2, u -> (1 + log(u)) / u),
    ("c(u) = exp(u)", u -> exp(u), u -> exp(u)),
    ("c(u) = sin(u)", u -> sin(u), u -> cos(u))
]
    check_refined("Jacobi identity, $name  [thm:jacobi]",
        refined_residuals(g -> begin
            u = uu(g)
            jacobi_residual(g, Au(g), Bu(g), Cu(g), cfun.(u), cprime.(u))
        end)...)
end

header("4. Section 4.3: the Casimir invariants")

check_refined("int f(u) [A_u, u] = 0  [lem:casimir-lemma]",
    refined_residuals(g -> begin
        a, u = Au(g), uu(g)
        f = u .^ 2 .+ sin.(u)
        normerr(integrate(g, f .* canonical_bracket(g, a, u)), 0.0, integrate(g, abs.(f)))
    end)...)

check_refined("{A, int kappa(u)}_c = 0  [prop:casimirs]",
    refined_residuals(g -> begin
        a, u = Au(g), uu(g)
        c = (1 .+ log.(u)) .^ 2 ./ 2          # some structure function
        dkappa = 3 .* u .^ 2 .+ cos.(u)       # kappa'(u) for kappa = u^3 + sin u
        normerr(integrate(g, c .* canonical_bracket(g, a, dkappa)), 0.0,
            integrate(g, abs.(c .* canonical_bracket(g, a, dkappa))))
    end)...)

header("5. Section 5: the symmetric/antisymmetric bracket")

check_exact("{A,B}_S = int S_u^2 [A_u,B_u], no factor 1/2  [prop:sym-antisym-reduction]",
    exact_residual(g -> begin
        a, b, s = Au(g), Bu(g), chi(g)
        relerr(symmetric_2bracket(g, a, b, s),
            integrate(g, s .^ 2 .* canonical_bracket(g, a, b)))
    end))

check_refined(
    "S = (2/3) int u^{3/2}  ->  int u[A_u,B_u] EXACTLY  [eq:sym-antisym-reduction]",
    refined_residuals(g -> begin
        a, b, u = Au(g), Bu(g), uu(g)
        relerr(symmetric_2bracket(g, a, b, sqrt.(u)), lie_poisson_2bracket(g, a, b, u))
    end)...)

header("6. Section 5.1: the weighted brackets")

# s = (2/3) u^{3/2}  =>  s' = sqrt(u), s'' = 1/(2 sqrt u), s' s'' = 1/2
# omega = u          =>  Phi' = u/2  =>  Phi = u^2/4
check_refined("weighted: {A,B}_S = 2 int Phi(u)[A_u,B_u]  [prop:weighted-reduction]",
    refined_residuals(g -> begin
        a, b, u = Au(g), Bu(g), uu(g)
        relerr(weighted_2bracket(g, a, b, sqrt.(u), u),
            2 * integrate(g, (u .^ 2 ./ 4) .* canonical_bracket(g, a, b)))
    end)...)

check_refined("omega = u/(2(1+log u)), S = int u log u  ->  LP  [ex:log-entropy]",
    refined_residuals(g -> begin
        a, b, u = Au(g), Bu(g), uu(g)
        s = 1 .+ log.(u)
        relerr(weighted_2bracket(g, a, b, s, u ./ (2 .* s)),
            lie_poisson_2bracket(g, a, b, u))
    end)...)

# s = (2/3) u^{3/2}, omega = u  =>  Phi' = u/2, Phi = u^2/4,  c = omega s'^2/4 + Phi/2 = 3u^2/8
check_refined("weighted Gardner: c = omega s'^2/4 + Phi/2  [rem:weighted-gardner]",
    refined_residuals(g -> begin
        a, b, u = Au(g), Bu(g), uu(g)
        relerr(integrate(g, u .* gardner_2bracket_density(g, a, b, sqrt.(u))),
            integrate(g, (3 .* u .^ 2 ./ 8) .* canonical_bracket(g, a, b)))
    end)...)

header("7. Section 6: the two symmetric operators")

check_exact("the two-bracket vanishes identically  [prop:sym-sym-vanishes]",
    exact_residual(g -> begin
        a, b, s = Au(g), Bu(g), chi(g)
        num = integrate(g,
            symmetric_x(g, a, s) .* symmetric_y(g, b, s) .-
            symmetric_x(g, b, s) .* symmetric_y(g, a, s))
        normerr(num, 0.0, integrate(g, abs.(symmetric_x(g, a, s) .*
                                            symmetric_y(g, b, s))))
    end))

check_exact("alternative pairings -> int S_ux S_uy [A_u,B_u]  [eq:sym-sym-alternative]",
    exact_residual(g -> begin
        a, b, s = Au(g), Bu(g), chi(g)
        lhs = integrate(g,
            ∂x(g, a) .* ∂x(g, s) .* ∂y(g, b) .* ∂y(g, s) .-
            ∂x(g, b) .* ∂x(g, s) .* ∂y(g, a) .* ∂y(g, s))
        relerr(lhs, integrate(g, ∂x(g, s) .* ∂y(g, s) .* canonical_bracket(g, a, b)))
    end))

header("8. Section 3: the two antisymmetry conditions")

# Condition (i) holds for the Gardner four-bracket but not (ii); the reverse for the
# four-bracket of Section 5.  The violations matter as much as the identities: a bracket that
# was antisymmetric in every slot would pass the vanishing half and mean nothing.  Hence
# generic test fields, with no symmetry of their own under any exchange.
let g = spectral_grid(SPECTRAL_N), a = Au(g), b = Bu(g), c = Cu(g), d = chi(g)
    gard = antisymmetry_residuals(gardner_4bracket, g, a, b, c, d)
    symm = antisymmetry_residuals(symmetric_4bracket, g, a, b, c, d)

    check("Gardner satisfies (i), slot exchange   [~ 0]  [lem:antisymmetry]",
        gard.slot_exchange < 1e-11, @sprintf("%.2e", gard.slot_exchange))
    check("Gardner VIOLATES  (ii), pair exchange [>> 0]  [lem:antisymmetry]",
        gard.pair_exchange > 1e-3, @sprintf("%.2e", gard.pair_exchange))
    check("Sec. 5 VIOLATES   (i), slot exchange  [>> 0]  [lem:antisymmetry]",
        symm.slot_exchange > 1e-3, @sprintf("%.2e", symm.slot_exchange))
    check("Sec. 5 satisfies  (ii), pair exchange  [~ 0]  [lem:antisymmetry]",
        symm.pair_exchange < 1e-11, @sprintf("%.2e", symm.pair_exchange))
end

summary("verify_fourbracket_identities.jl")
