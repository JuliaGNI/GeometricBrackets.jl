#
# The fixed test fields and refinement drivers the four-bracket scripts share.
#
#     include(joinpath(@__DIR__, "torustools.jl"))
#     using .TorusTools
#
# The fields are the same in every script because reproducibility is the point: an identity
# that holds is expected to produce the same residual today as it did when the manuscript was
# written, and a residual that moves is then a signal rather than noise.  They were carried in
# three copies before this module existed, and `Cu` in a fourth as a local `c3`.
#
# Which resolutions a refinement uses is policy, not mathematics, so it lives here rather than
# in the package: `src/torus.jl` will build a grid at any `N`, and choosing 64 and 128 -- fine
# enough for the leading error term to dominate, coarse enough to stay quick -- is this
# manuscript's judgement, not something a library should impose.

module TorusTools

using GeometricBrackets

export Au, Bu, Cu, Du, chi, uu
export SPECTRAL_N, COARSE_N, FINE_N, REFINEMENT_RESOLUTIONS
export exact_residual, refined_residuals, refinement_sweep

# ---------------------------------------------------------------------------
# test fields
# ---------------------------------------------------------------------------
# A_u, B_u, C_u, D_u: arbitrary functional derivatives, band-limited, and generic -- no
# symmetry under exchanging any two of them, which the sign tests of Lemma 3.1 require.
# chi: an arbitrary S_u, used where a result holds for arbitrary S.
# u: bounded away from zero, so that u^{3/2} and log u are smooth on the whole grid.

"The functional derivative ``A_u``: a fixed band-limited trigonometric polynomial."
Au(g) = sample(g, (x, y) -> sin(x) + 0.6cos(2y) + 0.4sin(x + y) + 0.25cos(2x - 3y))

"The functional derivative ``B_u``."
Bu(g) = sample(g, (x, y) -> cos(y) + 0.5sin(2x) + 0.3cos(x - y) + 0.2sin(3x + y))

"The functional derivative ``C_u``, needed by the Jacobi and Plücker tests."
Cu(g) = sample(g, (x, y) -> sin(x + 2y) + 0.4cos(3x) + 0.3sin(y))

"The functional derivative ``D_u``, needed only where a four-bracket takes four independent arguments."
Du(g) = sample(g, (x, y) -> cos(2x - y) + 0.35sin(x))

"An arbitrary ``S_u``, bounded away from zero, for the results that hold for arbitrary ``S``."
chi(g) = sample(g, (x, y) -> 2.0 + 0.5sin(x - y) + 0.3cos(2x + y))

"""
    uu(g)

A positive field, ranging over `[2.1, 4.0]`.

Bounded away from zero so that ``u^{3/2}`` and ``\\log u`` are smooth, and bounded away from
``e^{-1}`` so that the log-entropy weight is regular on it — which is what makes it the
control against which `ustraddle` in `verify_fourbracket_log_entropy.jl` is read.  Its range
also misses every critical level ``e^{-(1+\\alpha)}`` of the shifted entropy except
``\\alpha = -2``, so the same field exhibits both the smooth and the singular case.
"""
uu(g) = sample(g, (x, y) -> 3.0 + 0.4sin(x) + 0.3cos(y) + 0.2sin(x + 2y) + 0.15cos(2x + y))

# ---------------------------------------------------------------------------
# refinement
# ---------------------------------------------------------------------------

"The resolution at which a band-limited identity is checked; 16 is ample for fields whose highest mode is three."
const SPECTRAL_N = 16

"The coarse resolution of a two-grid refinement check."
const COARSE_N = 64

"The fine resolution of a two-grid refinement check; double `COARSE_N`, so the ratio reads as a rate."
const FINE_N = 128

"The resolutions of a full convergence study, doubling so that the ratios read as orders."
const REFINEMENT_RESOLUTIONS = (32, 64, 128, 256)

# A residual is a scalar if the claim is about an integral and an array if it is about an
# integrand; reduce both to one number here so that the call sites do not each repeat it.
_residual(x::AbstractArray) = maximum(abs, x)
_residual(x::Number) = abs(x)

"""
    exact_residual(f; N = SPECTRAL_N)

The residual of `f(grid)` on a [`spectral_grid`](@ref) of `N` points per direction, reduced to
a single number: `maximum(abs, ·)` if `f` returns an integrand, `abs` if it returns an integral.

For a claim that is exact algebra plus integration by parts on band-limited fields, so that
the answer should be at roundoff.  Feed it to `Checks.check_exact`.
"""
exact_residual(f; N::Int = SPECTRAL_N) = _residual(f(spectral_grid(N)))

"""
    refined_residuals(f; coarse = COARSE_N, fine = FINE_N)

The residuals of `f(grid)` on two [`finite_difference_grid`](@ref)s, as the tuple
`(coarse, fine)`.

For a claim involving fields that are not band-limited, where the residual is discretisation
error and only its decay certifies the identity.  Splat it into `Checks.check_refined`.
"""
function refined_residuals(f; coarse::Int = COARSE_N, fine::Int = FINE_N)
    (_residual(f(finite_difference_grid(coarse))),
        _residual(f(finite_difference_grid(fine))))
end

"""
    refinement_sweep(f; resolutions = REFINEMENT_RESOLUTIONS)

The residuals of `f(grid)` at each of `resolutions`, as a vector.

The full study behind [`refined_residuals`](@ref): two grids give a factor, four give an
observed order and show whether it is holding up or the residual has reached the roundoff
floor and stopped falling.
"""
function refinement_sweep(f; resolutions = REFINEMENT_RESOLUTIONS)
    [_residual(f(finite_difference_grid(N))) for N in resolutions]
end

end # module
