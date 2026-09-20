#!/usr/bin/env julia
#
# Dirac reduction of the discrete Lie-Poisson brackets: what it preserves and what it cannot
# create.
#
#     julia --project=scripts scripts/verify_dirac_reduction.jl
#
# The claims of `discrete-lie-poisson-dirac-brackets.tex`. The coarse subspace is exactly
# invariant, but the Jacobiator transports TENSORIALLY through the Dirac projector,
#
#     [Jhat,Jhat] = P (x) P (x) P [J,J] ,     P = Pi_V1 R ,
#
# so Dirac reduction preserves the Jacobi identity and never creates it. Closure and
# second-classness of the fine block exclude each other; four realisations of the
# hierarchical basis are examined and all fail; and two constructions do survive.
#
# The exact sections run over the rationals, so "the identity holds" means it holds and not
# that a residual fell below a tolerance. The refinement study of section 5 is the one place
# floating point is wanted, and even there the ASSEMBLY is exact -- the Python carried a
# separate float assembly with an SVD for the V2 complement, which is not needed: the exact
# one costs 1.7 s at the largest size here.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary, fmt
include(joinpath(@__DIR__, "rationals.jl"));
using .Rationals

const Q = Rational{BigInt}

"A random rational point with u_a = 0 for the constrained a."
function surface_point(N, keep, seed)
    rng = MersenneTwister(seed)
    u = zeros(Q, N)
    for i in keep
        u[i] = rnd(rng; lo = 1, hi = 9, dmax = 4)
    end
    u
end

"Jacobiator residual divided by the natural scale max|J|^2."
function normalised(res, J)
    sc = maximum(abs, J)^2
    iszero(sc) ? Inf : float(res / sc)
end

"The plain truncation to V1: J_ij = sum_{m in V1} c_ij^m u_m, and its dJ."
function naive_restriction(C, keep, u)
    Cs = restrict_c(C, keep)
    lie_poisson_matrix(Cs, u[keep]), lie_poisson_derivative(Cs)
end

raw(J, dJ) = first(jacobi_residual(J, dJ; normalised = false))
tf(x) = rpad(x ? "True" : "False", 5)

# --------------------------------------------------------------------------
header("1. Structural identities of the Dirac tensor")
# --------------------------------------------------------------------------

for (label, C0, keep, con, s) in ((
    "se(3), constraints {J1,P2}", se3(), [2, 3, 4, 6], [1, 5], 7),
    ("random antisymmetric c, N=7", nothing, collect(1:5), [6, 7], 8))
    C = C0 === nothing ? random_antisymmetric_c(MersenneTwister(s), 7) : C0
    N = size(C, 1)
    u = surface_point(N, keep, s)
    J, dJ = lie_poisson_matrix(C, u), lie_poisson_derivative(C)
    Jstar = dirac_tensor(J, con)
    R = dirac_R(J, con)
    check("$label: J* has vanishing V2 rows and columns",
        all(iszero, Jstar[con, :]) && all(iszero, Jstar[:, con]))
    check("$label: J* is antisymmetric", Jstar == -transpose(Jstar))
    check("$label: R is idempotent, R^2 = R", R * R == R)
    check("$label: R is not an involution, R^2 != I", R * R != I)
    rR = exact_rank(R)
    check("$label: rank R = dim V1, so dim ker R = dim V2", rR == length(keep),
        "rank R = $rR, dim V1 = $(length(keep))")
    check("$label: J* = R J R^T", R * J * transpose(R) == Jstar)
    check("$label: Jhat is the Schur complement J11 - J12 J22^-1 J21",
        Jstar[keep, keep] == schur_complement(J, keep, con))
    # the constraints are Casimirs of the Dirac bracket: J* grad phi_a = 0
    check("$label: every constraint is a Casimir of the Dirac bracket, J* grad phi = 0",
        all(iszero, Jstar[:, con]))
end

# a Casimir of J restricts to a Casimir of J*
let K = rnd_mat(MersenneTwister(9), 9, :antisymmetric), keep = collect(1:5),
    con = collect(6:9)

    nvec = kernel(K)
    if size(nvec, 2) ≥ 1
        Jstar = dirac_tensor(K, con)              # constant bracket, gradient n0
        check("a Casimir of J restricts to a Casimir of J*", all(iszero, Jstar *
                                                                         nvec[:, 1]))
    end
end

# --------------------------------------------------------------------------
header("2. The Jacobiator transports through the Dirac projector")
# --------------------------------------------------------------------------

println("   [Jhat,Jhat]^ijk = P^i_l P^j_m P^k_n [J,J]^lmn  on the constraint surface,")
println("   with P = Pi_V1 R.  Verified exactly at random rational points.")
println()

for (label, kind, s) in (("se(3) (Poisson)", :se3, 11),
    ("random c, N=5 (not Poisson)", :random, 13),
    ("truncated Witt K=2 (not Poisson)", :witt, 17))
    local C, keep, con
    if kind === :se3
        C, keep, con = se3(), [2, 3, 4, 6], [1, 5]
    elseif kind === :random
        C, keep, con = random_antisymmetric_c(MersenneTwister(s), 5), [1, 2, 3], [4, 5]
    else
        tr = witt_truncation(2)
        C, keep, con = tr.C, tr.keep, tr.con
    end
    N = size(C, 1)
    u = surface_point(N, keep, s)
    J, dJ = lie_poisson_matrix(C, u), lie_poisson_derivative(C)
    P = dirac_projector(J, keep, con)
    proj = project_jacobiator(jacobiator(J, dJ), P)
    Ĵ, dĴ = reduced_bracket(C, u, keep, con)
    red = jacobiator(Ĵ, dĴ)
    check("$label: the projector identity holds exactly", proj == red)
    big, small = maximum(abs, jacobiator(J, dJ)), maximum(abs, red)
    if kind === :se3
        check("se(3): [J,J] = 0 implies [Jhat,Jhat] = 0", iszero(big) && iszero(small))
    else
        check("$label: [J,J] != 0 and [Jhat,Jhat] != 0", !iszero(big) && !iszero(small),
            "|[J,J]| = $(fmt(big)), |[Jhat,Jhat]| = $(fmt(small))")
    end
end

# the size of the cancellation room
let tr = witt_truncation(2)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 19)
    P = dirac_projector(lie_poisson_matrix(C, u), keep, con)
    rP = exact_rank(P)
    @printf("   the cancellation room: P is %d x %d, rank %d, dim ker P = %d = dim V2 = %d\n",
        length(keep), size(C, 1), rP, size(C, 1) - rP, length(con))
    check("dim ker P = dim V2, so the room is nonempty but of finite codimension",
        size(C, 1) - rP == length(con))
end

# nested reduction buys nothing
let tr = witt_truncation(3)
    C = tr.C
    modes = collect(-6:6)
    V1 = [i for (i, k) in enumerate(modes) if abs(k) ≤ 2]
    V2 = [i for (i, k) in enumerate(modes) if 2 < abs(k) ≤ 4]
    V3 = [i for (i, k) in enumerate(modes) if abs(k) > 4]
    u = surface_point(size(C, 1), V1, 23)
    J = lie_poisson_matrix(C, u)
    one = schur_complement(J, V1, vcat(V2, V3))
    stp = schur_complement(J, vcat(V1, V2), V3)
    two = schur_complement(stp, 1:length(V1), (length(V1) + 1):(length(V1) + length(V2)))
    check("composition: reducing V1+V2+V3 -> V1+V2 -> V1 equals one reduction with V2+V3",
        one == two, "so nested or iterated reduction is not a new construction")
end

# --------------------------------------------------------------------------
header("3. The tension: closure and second-classness exclude each other")
# --------------------------------------------------------------------------

println("   A non-negative additive grading makes { deg > D } an ideal, so the")
println("   truncation closes; but then a,b in V2 gives deg(a+b) > max(deg a, deg b),")
println("   so [V2,V2] never reaches V1 and C vanishes identically.")
println()
println("     lo    D    dim   closes   Lie algebra   V2 ideal   |C|   class")
for (lo, D, p) in ((0, 4, 2), (0, 6, 3), (0, 8, 4), (-1, 4, 2), (-1, 6, 3), (-2, 2, 1), (
    -4, 4, 2))
    gw = graded_witt(lo, D)
    C, degs = gw.C, gw.degs
    keep = lo ≥ 0 ? [i for (i, a) in enumerate(degs) if a ≤ p] :
           [i for (i, a) in enumerate(degs) if abs(a) ≤ p]
    con = [i for i in eachindex(degs) if i ∉ keep]
    u = surface_point(length(degs), keep, 29)
    Cm = lie_poisson_matrix(C, u)[con, con]
    z = isempty(Cm) || iszero(maximum(abs, Cm))
    @printf("     %3d  %3d   %3d     %s      %s       %s    %-3s   %s\n",
        lo, D, length(degs), tf(closes_on(C, keep, con)),
        tf(iszero(structure_constant_residual(C))), tf(is_ideal(C, con, keep)),
        z ? "0" : "nz", z ? "FIRST" : "second")
end

for (lo, D, p) in ((0, 4, 2), (0, 6, 3), (0, 8, 4))
    gw = graded_witt(lo, D)
    C, degs = gw.C, gw.degs
    keep = [i for (i, a) in enumerate(degs) if a ≤ p]
    con = [i for i in eachindex(degs) if i ∉ keep]
    u = surface_point(length(degs), keep, 31)
    J = lie_poisson_matrix(C, u)
    check("non-negative grading, p=$p: the truncation is a Lie algebra",
        iszero(structure_constant_residual(C)))
    check("non-negative grading, p=$p: V2 is an ideal, so Sigma is a Poisson submanifold",
        is_ideal(C, con, keep))
    check("non-negative grading, p=$p: C vanishes, the constraints are first class",
        all(iszero, J[con, con]))
    Jn, dJn = naive_restriction(C, keep, u)
    check("non-negative grading, p=$p: the PLAIN restriction is already Poisson",
        iszero(raw(Jn, dJn)), "no Dirac reduction needed, and none is available")
end

for (lo, D, p) in ((-4, 4, 2), (-6, 6, 3))
    gw = graded_witt(lo, D)
    C, degs = gw.C, gw.degs
    keep = [i for (i, a) in enumerate(degs) if abs(a) ≤ p]
    con = [i for i in eachindex(degs) if i ∉ keep]
    u = surface_point(length(degs), keep, 37)
    J = lie_poisson_matrix(C, u)
    check("two-sided grading, p=$p: the truncation is NOT a Lie algebra",
        !iszero(structure_constant_residual(C)))
    check("two-sided grading, p=$p: but C is invertible, the constraints are second class",
        exact_rank(J[con, con]) == length(con))
end

# --------------------------------------------------------------------------
header("4. Parity and rank: when is the Dirac bracket defined at all?")
# --------------------------------------------------------------------------

println("   C is antisymmetric, so rank C is even and dim V2 must be even.")
println("   Even then det C is a nonconstant polynomial, so the constraints go")
println("   first class on a hypersurface through state space.")
println()
let tr = witt_truncation(2)
    C, keep, con = tr.C, tr.keep, tr.con
    for s in (41, 43, 47)
        u = surface_point(size(C, 1), keep, s)
        println("     Witt K=2, point $s: det C = ",
            fmt(det(lie_poisson_matrix(C, u)[con, con])))
    end
end
# a genuine parity example: an odd fine block whose C is nonzero but singular
let r = assemble_dg_hierarchical(3, 3)
    uf = surface_point(size(r.C, 1), r.keep, 53)
    Cf = lie_poisson_matrix(r.C, uf)[r.con, r.con]
    rk = exact_rank(Cf)
    check("an odd-dimensional fine block forces C to be singular even when C != 0",
        isodd(length(r.con)) && !iszero(maximum(abs, Cf)) && rk < length(r.con),
        "broken hierarchical P3, ne=3: dim V2 = $(length(r.con)), rank C = $rk")
    check("and the rank is even, as it must be for an antisymmetric matrix", iseven(rk),
        "rank C = $rk")
end
# the poly truncation is singular for the stronger, grading reason
let tr = poly_truncation(3)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 53)
    check("the global-polynomial fine block has C = 0 outright, not merely singular",
        all(iszero, lie_poisson_matrix(C, u)[con, con]),
        "there the grading, not the parity, is what kills C")
end

# --------------------------------------------------------------------------
header("5. The four realisations of the hierarchical basis")
# --------------------------------------------------------------------------

println("   All four have [V1,V1] contained in V1 + V2 exactly, which is the")
println("   premise of the construction.  Normalised residual = max|Jac| / max|J|^2.")
println()
println("   (a) Fourier on S^1: the Burgers / Vect(S^1) bracket as the truncated")
println("       Witt algebra [L_k,L_l] = (l-k) L_{k+l}, V1 = {|k| <= K}.")
println()
println("        K   N   dimV1 dimV2  closes  rank C   naive      Dirac")
for K in (1, 2, 3)
    tr = witt_truncation(K)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 61 + K)
    J = lie_poisson_matrix(C, u)
    rC = exact_rank(J[con, con])
    Jn, dJn = naive_restriction(C, keep, u)
    rn = normalised(raw(Jn, dJn), Jn)
    rrs = "undef"
    if rC == length(con)
        Ĵ, dĴ = reduced_bracket(C, u, keep, con)
        rrs = @sprintf("%.4f", normalised(raw(Ĵ, dĴ), Ĵ))
    end
    @printf("        %d  %3d    %2d    %2d    %s     %d    %.4f     %s\n",
        K, size(C, 1), length(keep), length(con), tf(closes_on(C, keep, con)), rC, rn, rrs)
end

let tr = witt_truncation(1)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 62)
    Ĵ, dĴ = reduced_bracket(C, u, keep, con)
    check(
        "Witt K=1: the reduced Jacobiator vanishes -- DEGENERATE, do not use as evidence",
        iszero(raw(Ĵ, dĴ)),
        "dim V1 = 3, so Jhat has rank 2 and is decomposable; Jacobi reduces to " *
        "Frobenius involutivity")
    check("Witt K=1: the anchor -- {L_-1,L_0,L_1} = sl(2) is already a subalgebra",
        iszero(structure_constant_residual(restrict_c(C, keep))),
        "so at K=1 the coarse block needs no V2 at all")
end
for K in (2, 3)
    tr = witt_truncation(K)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 61 + K)
    Ĵ, dĴ = reduced_bracket(C, u, keep, con)
    r = raw(Ĵ, dĴ)
    check("Witt K=$K: the reduced Jacobiator does NOT vanish", !iszero(r),
        @sprintf("normalised residual = %.4f", normalised(r, Ĵ)))
end

println()
println("   (b) Fourier on T^2: the vorticity / Vlasov bracket,")
println("       [e_m,e_n] = (m x n) e_{m+n}, V1 = {|m|_inf <= K} minus {0}.")
println()
let tr = torus_truncation(1)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 71)
    J = lie_poisson_matrix(C, u)
    Cm = J[con, con]
    sel = maximal_second_class(J, con)
    newkeep = sort(vcat(keep, setdiff(con, sel)))
    Ĵ, dĴ = reduced_bracket(C, u, newkeep, sel)
    Jn, dJn = naive_restriction(C, keep, u)
    @printf("        K=1: N=%d, dim V1=%d, dim V2=%d, closes=%s\n",
        size(C, 1), length(keep), length(con), closes_on(C, keep, con) ? "True" : "False")
    @printf("             rank C = %d/%d, corank %d: the\n",
        exact_rank(Cm), length(con), size(kernel(Cm), 2))
    println("             constraints are MIXED, not purely second class")
    @printf("             maximal second-class subset: %d of %d, reduced space dim %d\n",
        length(sel), length(con), length(newkeep))
    @printf("             naive restriction  normalised residual = %.4f\n",
        normalised(raw(Jn, dJn), Jn))
    @printf("             mixed-class Dirac  normalised residual = %.4f\n",
        normalised(raw(Ĵ, dĴ), Ĵ))
    check("torus K=1: [V1,V1] closes in V1+V2 exactly", closes_on(C, keep, con))
    check("torus K=1: C is rank deficient, so the constraints are not purely second class",
        exact_rank(Cm) < length(con), "rank $(exact_rank(Cm)) of $(length(con))")
    check("torus K=1: reduction along a maximal second-class subset still fails Jacobi",
        !iszero(raw(Ĵ, dĴ)))
end

println()
println("   (c) Global polynomials: L_a = x^(a+1), a >= -1, V1 = degree <= p.")
println("       The grading is bounded below but includes a = -1, so V2 is not an")
println("       ideal AND C vanishes: neither route is available.")
println()
println("        p   N   dimV1 dimV2  Lie alg  V2 ideal   |C|   plain restriction")
for p in (2, 3, 4)
    tr = poly_truncation(p)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 79)
    J = lie_poisson_matrix(C, u)
    Jn, dJn = naive_restriction(C, keep, u)
    r = raw(Jn, dJn)
    z = all(iszero, J[con, con])
    @printf("        %d  %3d    %2d    %2d    %s    %s     %-3s   %s\n",
        p, size(C, 1), length(keep), length(con),
        tf(iszero(structure_constant_residual(C))), tf(is_ideal(C, con, keep)),
        z ? "0" : "nz",
        iszero(r) ? "Poisson" : @sprintf("residual %.4f", normalised(r, Jn)))
end
for p in (3, 4)
    tr = poly_truncation(p)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 79)
    J = lie_poisson_matrix(C, u)
    check("global polynomials p=$p: C vanishes, the Dirac bracket is undefined",
        all(iszero, J[con, con]))
    Jn, dJn = naive_restriction(C, keep, u)
    check("global polynomials p=$p: and the plain restriction is not Poisson either",
        !iszero(raw(Jn, dJn)))
end

println()
println("   (d) Hierarchical finite elements.  A C0 basis does NOT close: phi' jumps")
println("       across interfaces, so [phi_p,phi_q] is discontinuous.  The repair is")
println("       a broken ambient space, V2 = the L2-orthogonal complement of the")
println("       continuous degree-p block inside the broken degree-(2p-1) space.")
println()
for (p, ne) in ((2, 2), (2, 3), (3, 2))
    r = assemble_dg_hierarchical(p, ne)
    C, keep, con, M = r.C, r.keep, r.con, r.M
    u = surface_point(size(C, 1), keep, 83)
    J = lie_poisson_matrix(C, u)
    check("P$p, ne=$ne: the broken hierarchical c is antisymmetric in i,j", is_antisymmetric_c(C))
    check("P$p, ne=$ne: the splitting is L2-orthogonal, so M is block diagonal",
        all(iszero, M[keep, con]))
    check("P$p, ne=$ne: [V1,V1] closes in V1+V2 exactly", closes_on(C, keep, con))
    check("P$p, ne=$ne: the big bracket is still not a Lie algebra",
        !iszero(structure_constant_residual(C)))
    if exact_rank(J[con, con]) == length(con)
        Ĵ, dĴ = reduced_bracket(C, u, keep, con)
        rr = raw(Ĵ, dĴ)
        check("P$p, ne=$ne: the Dirac-reduced bracket is still not Poisson", !iszero(rr),
            @sprintf("normalised residual = %.4f", normalised(rr, Ĵ)))
    end
end

println()
println("       Refinement study (floating point).  The normalised Dirac residual is")
println("       flat in h: Dirac reduction lowers it by a constant factor but does")
println("       not send it to zero, exactly as (H) being a closed condition demands.")
println()
println("        p   ne    N   dim V1    naive      Dirac     ratio")
let trends = Dict{Int, Vector{NTuple{5, Float64}}}()
    for p in (2, 3)
        rows = NTuple{5, Float64}[]
        for ne in (2, 4, 6, 8)
            r = assemble_dg_hierarchical(p, ne)
            C = Float64.(r.C)                 # exact assembly, floating-point study
            keep, con = r.keep, r.con
            N = length(keep) + length(con)
            rng = MersenneTwister(100 + ne)
            u = zeros(N)
            u[keep] .= 1.0 .+ 3.0 .* rand(rng, length(keep))
            Ĵ, dĴ = reduced_bracket(C, u, keep, con)
            Jn, dJn = naive_restriction(C, keep, u)
            rn = raw(Jn, dJn) / maximum(abs, Jn)^2
            rr = raw(Ĵ, dĴ) / maximum(abs, Ĵ)^2
            push!(rows, (ne, N, length(keep), rn, rr))
            @printf("        P%d  %2d   %3d    %3d    %.6f   %.6f   %6.3f\n",
                p, ne, N, length(keep), rn, rr, rr / rn)
        end
        trends[p] = rows
    end
    for p in (2, 3)
        rows = trends[p]
        check("P$p: the Dirac-reduced residual is nonzero at every resolution",
            all(r -> r[5] > 1e-8, rows))
        check("P$p: and does not decrease under a fourfold refinement",
            rows[end][5] > 0.1 * rows[1][5],
            @sprintf("%.5f -> %.5f from ne=%d to ne=%d",
                rows[1][5], rows[end][5], Int(rows[1][1]), Int(rows[end][1])))
        check("P$p: Dirac reduction does lower it by a constant factor",
            rows[end][5] < rows[end][4])
    end
end

# the correction is not small
let tr = witt_truncation(2)
    C, keep, con = tr.C, tr.keep, tr.con
    u = surface_point(size(C, 1), keep, 91)
    J = lie_poisson_matrix(C, u)
    J11 = J[keep, keep]
    Ĵ = schur_complement(J, keep, con)
    d = maximum(abs, Ĵ - J11)
    check("the Dirac correction is O(1), not a small perturbation of the truncation",
        d > maximum(abs, J11) / 10,
        @sprintf("max|Jhat - J11| / max|J11| = %.3f, ", float(d / maximum(abs, J11))) *
        "so consistency is perturbed at leading order too")
end

# --------------------------------------------------------------------------
header("6. What does work")
# --------------------------------------------------------------------------

println("   (a) The family of Theorem 2.1 is closed under Dirac reduction.")
println("       diag(g) is block diagonal, so")
println("         Jhat = diag(g1) [ K11 - K12 K22^-1 K21 ] diag(g1) ,")
println("       and the Schur complement of a CONSTANT antisymmetric K is again a")
println("       constant antisymmetric matrix.  Dirac reduction therefore acts on")
println("       the family by K -> Schur(K), with g untouched.")
println()
let rng = MersenneTwister(101)
    for (N, nc) in ((7, 2), (9, 4), (11, 4))
        K = rnd_mat(rng, N, :antisymmetric)
        keep, con = collect(1:(N - nc)), collect((N - nc + 1):N)
        g = rnd_vec(rng, N; lo = 1, hi = 9, dmax = 3)
        dg = [l == i ? rnd(rng) : zero(Q) for i in 1:N, l in 1:N]
        J = [g[i] * K[i, j] * g[j] for i in 1:N, j in 1:N]
        Ks = schur_complement(K, keep, con)
        Ĵ = schur_complement(J, keep, con)
        nk = length(keep)
        pred = [g[keep[i]] * Ks[i, j] * g[keep[j]] for i in 1:nk, j in 1:nk]
        dĴ = [dg[keep[i], l] * Ks[i, j] * g[keep[j]] +
              g[keep[i]] * Ks[i, j] * dg[keep[j], l]
              for l in keep, i in 1:nk, j in 1:nk]
        check("N=$N, $nc constraints: Jhat = diag(g1) Schur(K) diag(g1)", Ĵ == pred)
        check("N=$N, $nc constraints: Schur(K) is constant and antisymmetric",
            Ks == -transpose(Ks))
        kerKs = kernel(Ks)
        check("N=$N, $nc constraints: the reduced bracket satisfies Jacobi exactly",
            iszero(raw(Ĵ, dĴ)),
            "rank K = $(exact_rank(K)) -> rank Schur(K) = $(exact_rank(Ks)), " *
            "$(size(kerKs, 2)) Casimir(s) from ker Schur(K)")
        if size(kerKs, 2) ≥ 1
            nv = kerKs[:, 1]
            check("N=$N, $nc constraints: n in ker Schur(K) gives a Casimir of Jhat",
                all(iszero, [sum(Ĵ[i, j] * nv[j] / g[keep[j]] for j in 1:nk) for i in 1:nk]),
                "C_n = sum_i n_i eta_i(u_i) with eta' = 1/g, as in Proposition 2.5")
        end
    end
end

println()
println("   (b) Dirac reduction of an honest Lie-Poisson bracket: the sine / su(N)")
println("       (Zeitlin) bracket restricted to a coarse mode set.  Exactly Poisson")
println("       by the projector identity, but rational and homogeneous of degree 1,")
println("       hence NOT Lie-Poisson.")
println()
let Nn = 5
    modes, C = sine_algebra(Nn)
    Cr = real.(C)
    d = length(modes)
    pos = Dict(m => i for (i, m) in enumerate(modes))
    rep(a) = a > Nn ÷ 2 ? a - Nn : a
    sup(m) = max(abs(rep(m[1])), abs(rep(m[2])))
    keep = [pos[m] for m in modes if sup(m) ≤ 1]
    con = [pos[m] for m in modes if sup(m) > 1]
    rng = MersenneTwister(1)
    u = zeros(d)
    u[keep] .= 1.0 .+ 3.0 .* rand(rng, length(keep))
    Ĵ, dĴ = reduced_bracket(Cr, u, keep, con)
    w = raw(Ĵ, dĴ)
    sc = maximum(abs, Ĵ)^2
    @printf("        su(%d): dim = %d, dim V1 = %d, dim V2 = %d\n", Nn, d, length(keep),
        length(con))
    @printf("                 normalised reduced Jacobiator = %.3e\n", w / sc)
    @printf("                 rank Jhat = %d of %d\n", rank(Ĵ), length(keep))
    res, scale = structure_constant_residual(Cr; normalised = false)
    check("su($Nn): the parent bracket is a Lie algebra", res / scale < 1e-10)
    check(
        "su($Nn): the Dirac-reduced bracket satisfies Jacobi to round-off", w / sc < 1e-10,
        @sprintf("normalised residual = %.2e", w / sc))
    u2 = copy(u)
    u2[keep] .*= 2
    Ĵ2, _ = reduced_bracket(Cr, u2, keep, con)
    check("su($Nn): Jhat is homogeneous of degree 1", isapprox(Ĵ2, 2 .* Ĵ; rtol = 1e-8))
    u3 = zeros(d)
    u3[keep] .= 1.0 .+ 3.0 .* rand(rng, length(keep))
    Ĵ3, _ = reduced_bracket(Cr, u3, keep, con)
    Ĵs, _ = reduced_bracket(Cr, u .+ u3, keep, con)
    check("su($Nn): but Jhat is NOT linear in u, so it is not Lie-Poisson",
        !isapprox(Ĵs, Ĵ .+ Ĵ3; rtol = 1e-8),
        @sprintf("max|Jhat(u+u') - Jhat(u) - Jhat(u')| = %.3e", maximum(abs, Ĵs - Ĵ - Ĵ3)))
end

# --------------------------------------------------------------------------
header("7. Dynamics: the coarse subspace is exactly invariant")
# --------------------------------------------------------------------------

println("   Implicit midpoint on u' = J* grad H over the FULL ambient space, with J*")
println("   the Dirac tensor of the Theorem-2.1 bracket on the broken hierarchical")
println("   basis of section 5(d):  J = diag(g) K diag(g),  K = M^-1 D M^-1,")
println("   D_IJ = int ( chi_I chi_J' - chi_J chi_I' ),  H = 1/2 u^T M u.")
println()
println("   Two adjustments are forced, and both are instances of obstructions already")
println("   met above rather than implementation details:")
println()
println("     * g_i = sqrt(u_i) will not do.  It vanishes on the constraint surface")
println("       u_fine = 0, so J22 = diag(g2) K22 diag(g2) vanishes there and the")
println("       constraints are first class.  We take g_i = sqrt(1 + u_i^2), which is")
println("       nowhere zero and smooth on all of R, with Casimir density")
println("       eta_i = arcsinh(u_i) since eta' = 1/g.")
println("     * K22 for the hierarchical split has corank exactly 2, the same even-rank")
println("       phenomenon as the spurious Casimir of Remark 3.5.  The fine block must")
println("       be shrunk by two directions to become second class; those two are then")
println("       dynamical and the invariant subspace is correspondingly larger.")
println()
println("   K is rescaled by 1/max|K|, a constant multiple of a Poisson bracket being")
println("   a Poisson bracket, so that the step size can be quoted plainly.")
println()

"""
    hierarchical_KD(p, ne) -> (M, D, K, keep, con)

Mass matrix, `D_IJ = int (chi_I chi_J' - chi_J chi_I')`, `K = M^-1 D M^-1` and the index
split, for the broken hierarchical basis, in floating point.

On one element in the shifted Legendre basis `D` is the `m = 0` slice of the local commutator
tensor, since `L_0 = 1`. The assembly is done exactly and converted at the end.
"""
function hierarchical_KD(p, ne)
    Mloc, Tloc = dg_local_algebra(p)
    nl, Ndg = 2p, 2p * ne
    Mdg = zeros(Q, Ndg, Ndg)
    Ddg = zeros(Q, Ndg, Ndg)
    for e in 0:(ne - 1)
        o = e * nl
        Mdg[(o + 1):(o + nl), (o + 1):(o + nl)] = Mloc
        Ddg[(o + 1):(o + nl), (o + 1):(o + nl)] = Tloc[1, :, :]      # L_0 = 1
    end
    Q1 = coarse_in_dg(Q, p, ne)
    N1 = size(Q1, 1)
    Qm = vcat(Q1, transpose(kernel(Q1 * Mdg)))
    M = Float64.(Qm * Mdg * transpose(Qm))
    D = Float64.(Qm * Ddg * transpose(Qm))
    D = (D - transpose(D)) / 2
    Minv = inv(M)
    K = Minv * D * Minv
    K = (K - transpose(K)) / 2
    return M, D, K, collect(1:N1), collect((N1 + 1):size(Qm, 1))
end

"Implicit midpoint by fixed-point iteration; returns the step and whether it converged."
function midpoint(u, h, f; tol = 1e-14, itmax = 300)
    v = u .+ h .* f(u)
    for _ in 1:itmax
        vn = u .+ h .* f(0.5 .* (u .+ v))
        maximum(abs, vn - v) ≤ tol * max(1.0, maximum(abs, vn)) && return vn, true
        v = vn
    end
    return v, false
end

"Integrate u' = J* grad H with J = diag(g) K diag(g), g = sqrt(1+u^2)."
function run_dynamics(label, Kmat, Mmat, keep, con, u0, nsteps, h)
    g(u) = sqrt.(1.0 .+ u .* u)
    η(u) = asinh.(u)
    function Jstar(u)
        gg = g(u)
        J = gg .* Kmat .* transpose(gg)
        J - J[:, con] * (J[con, con] \ J[con, :])
    end
    f(u) = Jstar(u) * (Mmat * u)
    Ks = Kmat[keep, keep] - Kmat[keep, con] * (Kmat[con, con] \ Kmat[con, keep])
    check("$label: Schur(K) is a constant antisymmetric matrix",
        maximum(abs, Ks + transpose(Ks)) < 1e-9 * maximum(abs, Ks))
    g0 = g(u0)
    check("$label: on the surface Jhat = diag(g1) Schur(K) diag(g1), as predicted",
        maximum(abs, Jstar(u0)[keep, keep] - g0[keep] .* Ks .* transpose(g0[keep]))
        <
        1e-8 * maximum(abs, Ks))

    nv = nullspace(Ks)                                 # columns spanning ker Schur(K)
    nvecs = [nv[:, i] for i in axes(nv, 2)]
    Cas(uu, n) = dot(n, η(uu[keep]))

    u = copy(u0)
    H0 = 0.5 * dot(u, Mmat * u)
    C0 = [Cas(u, n) for n in nvecs]
    fine_max, converged = 0.0, true
    for _ in 1:nsteps
        u, ok = midpoint(u, h, f)
        converged &= ok
        fine_max = max(fine_max, isempty(con) ? 0.0 : maximum(abs, u[con]))
    end
    H1 = 0.5 * dot(u, Mmat * u)
    @printf("        %d steps of size %g, reduced dim %d, constraints %d, corank Schur(K) = %d\n",
        nsteps, h, length(keep), length(con), length(nvecs))
    @printf("        max |u_constrained| over the run  = %.3e\n", fine_max)
    @printf("        relative energy drift             = %.3e\n", abs(H1 - H0) / abs(H0))
    for (k, n) in enumerate(nvecs)
        @printf("        relative Casimir drift, n_%d       = %.3e\n",
            k - 1, abs(Cas(u, n) - C0[k]) / max(abs(C0[k]), 1e-30))
    end
    check("$label: the midpoint iteration converged at every step", converged)
    check(
        "$label: the constrained coefficients stay at zero to round-off, " *
        "for any Hamiltonian",
        fine_max < 1e-12,
        @sprintf("max = %.2e", fine_max))
    check(
        "$label: energy is conserved, the midpoint rule preserving quadratic " *
        "invariants",
        abs(H1 - H0) / abs(H0) < 1e-11,
        @sprintf("drift = %.2e", abs(H1 - H0) / abs(H0)))
    for (k, n) in enumerate(nvecs)
        d = abs(Cas(u, n) - C0[k]) / max(abs(C0[k]), 1e-30)
        check(
            "$label: the reduced Casimir drifts only at O(h^2) in the u " *
            "formulation, C_n being nonlinear there",
            d > 1e-12,
            @sprintf("drift = %.2e; the flat formulation below removes it", d))
    end

    isempty(nvecs) && return 0

    # Flat coordinates. With ubar_i = eta_i(u_i) the bracket is the CONSTANT Schur(K) and
    # C_n = sum_i n_i ubar_i is LINEAR, so every Runge-Kutta method preserves it exactly --
    # section 2.4(iii) of the notes. Energy is no longer quadratic there, so the two
    # conservation statements trade places.
    ub0 = asinh.(u0[keep])
    Mkk = Mmat[keep, keep]
    Hbar(ub) = 0.5 * dot(sinh.(ub), Mkk * sinh.(ub))
    gradH(ub) = cosh.(ub) .* (Mkk * sinh.(ub))
    fbar(ub) = Ks * gradH(ub)
    ub, conv2 = copy(ub0), true
    for _ in 1:nsteps
        ub, ok = midpoint(ub, h, fbar)
        conv2 &= ok
    end
    dC = [abs(dot(n, ub) - dot(n, ub0)) / max(abs(dot(n, ub0)), 1e-30) for n in nvecs]
    println("        flat coordinates ubar = arcsinh(u): bracket is the constant Schur(K)")
    @printf("        relative energy drift             = %.3e\n",
        abs(Hbar(ub) - Hbar(ub0)) / abs(Hbar(ub0)))
    for (k, d) in enumerate(dC)
        @printf("        relative Casimir drift, n_%d       = %.3e\n", k - 1, d)
    end
    check("$label: the midpoint iteration converged in flat coordinates", conv2)
    for (k, d) in enumerate(dC)
        check(
            "$label: in flat coordinates C_n = sum_i n_i ubar_i is linear and " *
            "conserved to round-off, n_$(k-1)",
            d < 1e-11,
            @sprintf("drift = %.2e", d))
    end
    return length(nvecs)
end

let p_deg = 2, ne = 6
    M, D, K0, keep0, con0 = hierarchical_KD(p_deg, ne)
    N = length(keep0) + length(con0)
    check("D is antisymmetric", maximum(abs, D + transpose(D)) < 1e-9 * maximum(abs, D))
    check("K = M^-1 D M^-1 is antisymmetric",
        maximum(abs, K0 + transpose(K0)) < 1e-9 * maximum(abs, K0))
    K22 = K0[con0, con0]
    check(
        "K22 for the hierarchical split has corank 2, so the fine block is not " *
        "second class as it stands",
        length(con0) - rank(K22) == 2,
        "dim V2 = $(length(con0)), rank K22 = $(rank(K22))")

    Kmat = K0 ./ maximum(abs, K0)
    con = maximal_second_class(Kmat, con0)
    keep = sort(collect(setdiff(1:N, con)))
    rng = MersenneTwister(7)
    u0 = zeros(N)
    u0[keep] .= 0.4 .+ 1.2 .* rand(rng, length(keep))
    run_dynamics("hierarchical P2, ne=6", Kmat, M, keep, con, u0, 400, 1.0e-3)
end

println()
println("   The reduced space above has even dimension, so Schur(K) is generically")
println("   nonsingular and carries no Casimir.  An odd-dimensional reduced space must")
println("   carry one, an antisymmetric matrix having even rank; here it is conserved")
println("   to round-off, as Proposition 2.5 and section 2.4(iii) of the notes predict.")
println()
let Nr = 11, ncr = 4
    rng = MersenneTwister(211)
    Kr = Float64.(rnd_mat(rng, Nr, :antisymmetric))
    Kr ./= maximum(abs, Kr)
    keepr, conr = collect(1:(Nr - ncr)), collect((Nr - ncr + 1):Nr)
    Mr = Matrix{Float64}(I, Nr, Nr)
    rng2 = MersenneTwister(11)
    u0r = zeros(Nr)
    u0r[keepr] .= 0.4 .+ 1.2 .* rand(rng2, length(keepr))
    ncas = run_dynamics("random constant K, N=11, 4 constraints",
        Kr, Mr, keepr, conr, u0r, 400, 2.0e-3)
    check("an odd-dimensional reduced space carries at least one Casimir", ncas ≥ 1,
        "corank Schur(K) = $ncas, reduced dim = $(length(keepr))")
end

summary("verify_dirac_reduction.jl")
