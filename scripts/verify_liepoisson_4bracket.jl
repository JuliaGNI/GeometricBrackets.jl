#!/usr/bin/env julia
#
# The antisymmetric 4-bracket generates EVERY finite-dimensional Lie-Poisson bracket.
#
#     julia --project=scripts scripts/verify_liepoisson_4bracket.jl
#
# Set
#
#     K^{iajb} = delta^{ab} c_ij^a   (no sum over a),   c_ij^m = -c_ji^m,
#     s        = (2/3) sum_i z_i^{3/2}   so that   s_a = sqrt(z_a).
#
# Then the two-bracket {f,g}_s = {f,s;g,s} has J_ij = s_a K^{iajb} s_b = sum_m c_ij^m z_m, a
# Lie-Poisson bracket, and Jacobi holds iff the c_ij^m are the structure constants of an
# N-dimensional Lie algebra.
#
#   1. the required 4-bracket symmetries of K, and that K is not trivialised;
#   2. J_ij = sum_m c_ij^m z_m, and Jacobi <=> structure constants;
#   3. that the 3/2 power is FORCED within the separable ansatz s = sum_a f_a(z_a);
#   4. that this K is reachable from the Gardner-like ansatz once mu and sigma are allowed to
#      be neither antisymmetric nor diagonal;
#   5. end to end, an explicit realisation of se(3).
#
# Exact rational arithmetic throughout.
#
# Note on index conventions: the package stores structure constants as C[m,i,j] = c_ij^m,
# where the Python prototype used c[i][j][m]. The leading index is what lets C[m,:,:] be a
# matrix, and it is what `structure_constant_residual` already expected.

using GeometricBrackets
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary, fmt
include(joinpath(@__DIR__, "rationals.jl"));
using .Rationals

const Q = Rational{BigInt}
const rng = MersenneTwister(31)

"K[i,a,j,b] = delta_ab c_ij^a, from C[m,i,j]."
function K_from_c(C)
    n = size(C, 1)
    K = zeros(eltype(C), n, n, n, n)
    for i in 1:n, j in 1:n, a in 1:n
        K[i, a, j, a] = C[a, i, j]
    end
    K
end

"J_ij = sum_{a,b} s_a K^{iajb} s_b."
function contract(K, sgrad)
    [sum(sgrad[a] * K[i, a, j, b] * sgrad[b]
     for a in eachindex(sgrad), b in eachindex(sgrad))
     for i in eachindex(sgrad), j in eachindex(sgrad)]
end

header("1. Symmetries of K^{i a j b} = delta^{ab} c_ij^a")

C = se3()
n = size(C, 1)
K = K_from_c(C)
check("antisymmetric in slots 1<->3 (the identity required in the notes)",
    all(K[i, a, j, b] == -K[j, a, i, b] for i in 1:n, a in 1:n, j in 1:n, b in 1:n))
check("symmetric in slots 2<->4 (already in the normal form, so not trivialised)",
    all(K[i, a, j, b] == K[i, b, j, a] for i in 1:n, a in 1:n, j in 1:n, b in 1:n))
check("NOT antisymmetric in slots 2<->4 (Trivialisation I does not apply)",
    any(K[i, a, j, b] != -K[i, b, j, a] for i in 1:n, a in 1:n, j in 1:n, b in 1:n))

header("2. J_ij = sum_m c_ij^m z_m, and Jacobi <=> structure constants")

for (label, C) in (("so(3) (padded to N=5)", so3(Q, 5)),
    ("se(3)", se3()),
    ("random antisymmetric c (N=5)", random_antisymmetric_c(rng, 5)))
    local n = size(C, 1)
    local K = K_from_c(C)
    # s = (2/3) Σ z^{3/2} ⟹ s_a = √z_a. Work with w_a = √z_a so the arithmetic stays exact:
    # s_a s_a = w_a² = z_a, and no square root is ever taken.
    w = rnd_vec(rng, n; lo = 1, hi = 9, dmax = 3)
    z = w .^ 2
    J = contract(K, w)
    J_direct = lie_poisson_matrix(C, z)
    dJ = lie_poisson_derivative(C)
    jac = first(jacobi_residual(J, dJ; normalised = false))
    sc = first(structure_constant_residual(C; normalised = false))
    check("$label: contraction reproduces sum_m c_ij^m z_m", J == J_direct)
    check("$label: J antisymmetric", J == -transpose(J))
    check("$label: Jacobi vanishes <=> structure constants vanish",
        iszero(jac) == iszero(sc), "jacobi=$(fmt(jac)), structconst=$(fmt(sc))")
end

header("3. Within separable s = sum_a f_a(z_a), the 3/2 power is forced")

# With separable s, J_ij = Σ_a c_ij^a f_a'(z_a)². Write w_a := f_a'(z_a)². The bracket is
# Poisson for every Lie algebra c only if w_a = λ z_a + β with a slope λ common to all a.
function sweep(C, label)
    n = size(C, 1)
    println("  -- test algebra: $label")
    z = rnd_vec(rng, n; lo = 1, hi = 9, dmax = 1)
    lam = rnd_vec(rng, n; lo = 1, hi = 5, dmax = 1)
    beta = rnd_vec(rng, n; lo = -5, hi = 5, dmax = 1)
    cases = [
        ("w_a = z_a            (s = 2/3 sum z^{3/2})", z, ones(Q, n), true),
        ("w_a = 3 z_a + 2      (affine, common slope)", 3 .* z .+ 2, fill(Q(3), n), true),
        # A per-index shift β_a is harmless: the constant part of the bracket is
        # Θ_ij = β(c_ij^a) = β([e_i,e_j]), the coboundary of the linear functional β, hence a
        # 2-cocycle, so the affine bracket is again Poisson. Only the SLOPE must be uniform.
        ("w_a = 3 z_a + beta_a (affine, per-index shift)",
            3 .* z .+ beta, fill(Q(3), n), true),
        ("w_a = lambda_a z_a   (affine, per-index slope)", lam .* z, lam, nothing),
        ("w_a = z_a^2          (s = z^2/2)", z .^ 2, 2 .* z, nothing),
        ("w_a = z_a^3", z .^ 3, 3 .* z .^ 2, nothing)
    ]
    results = Dict{String, Q}()
    for (name, wv, dwv, expect_zero) in cases
        J = lie_poisson_matrix(C, wv)
        dJ = [C[l, i, j] * dwv[l] for l in 1:n, i in 1:n, j in 1:n]
        res = first(jacobi_residual(J, dJ; normalised = false))
        results[name] = res
        if expect_zero === true
            check("     $name: Poisson", iszero(res))
        else
            println("     $name: residual = $(fmt(res))")
        end
    end
    return results
end

sweep(so3(), "so(3) -- DEGENERATE, passes everything, do not use as a test case")
res = sweep(se3(), "se(3) -- discriminating")
check("se(3): per-index slopes lambda_a z_a are NOT Poisson",
    !iszero(res["w_a = lambda_a z_a   (affine, per-index slope)"]))
check("se(3): w_a = z_a^2 is NOT Poisson", !iszero(res["w_a = z_a^2          (s = z^2/2)"]))
check("se(3): w_a = z_a^3 is NOT Poisson", !iszero(res["w_a = z_a^3"]))

header("4. Reachable from the Gardner-like ansatz with unconstrained mu, sigma")

# K^{iajb} = Σ_χ (mu^χ_{ia} sigma^χ_{jb} - mu^χ_{ja} sigma^χ_{ib}) with χ = (a, r) and
#     mu^{(a,r)}_{iα} = δ_{αa} A^{(a,r)}_i ,   sigma^{(a,r)}_{jβ} = δ_{βa} B^{(a,r)}_j .
# These mu, sigma are neither antisymmetric nor diagonal, and dropping both assumptions is
# exactly what makes the construction work.
const N, R = 5, 3
A = [[rnd_vec(rng, N) for _ in 1:R] for _ in 1:N]
B = [[rnd_vec(rng, N) for _ in 1:R] for _ in 1:N]

Kg = zeros(Q, N, N, N, N)
for a in 1:N, r in 1:R, i in 1:N, j in 1:N
    Kg[i, a, j, a] += A[a][r][i] * B[a][r][j] - A[a][r][j] * B[a][r][i]
end

check("the off-diagonal (alpha != beta) part of K vanishes identically",
    all(iszero(Kg[i, a, j, b]) for i in 1:N, a in 1:N, j in 1:N, b in 1:N if a != b))
check("antisymmetric in slots 1<->3",
    all(Kg[i, a, j, b] == -Kg[j, a, i, b] for i in 1:N, a in 1:N, j in 1:N, b in 1:N))
check("symmetric in slots 2<->4",
    all(Kg[i, a, j, b] == Kg[i, b, j, a] for i in 1:N, a in 1:N, j in 1:N, b in 1:N))

Cg = [Kg[i, a, j, a] for a in 1:N, i in 1:N, j in 1:N]
check("c^a_ij is antisymmetric in i,j for every a", is_antisymmetric_c(Cg))
ranks = [exact_rank(Cg[a, :, :]) for a in 1:N]
check("c^a_ij reaches rank > 2 with R = $R terms (single term would give rank 2)",
    all(>(2), ranks), "ranks = $ranks")

# One term per Greek slot really is rank 2, which is the obstruction of the notes.
Kg1 = zeros(Q, N, N, N, N)
for a in 1:N, i in 1:N, j in 1:N
    Kg1[i, a, j, a] = A[a][1][i] * B[a][1][j] - A[a][1][j] * B[a][1][i]
end
r1 = exact_rank([Kg1[i, 1, j, 1] for i in 1:N, j in 1:N])
check("a single term (R = 1) is stuck at rank 2", r1 == 2, "rank = $r1")

# and the resulting bracket is Poisson exactly when Cg happens to be a Lie algebra
w = rnd_vec(rng, N; lo = 1, hi = 9, dmax = 3)
J = contract(Kg, w)
jac = first(jacobi_residual(J, lie_poisson_derivative(Cg); normalised = false))
sc = first(structure_constant_residual(Cg; normalised = false))
check("Jacobi vanishes <=> structure constants vanish (generic A, B: neither does)",
    iszero(jac) == iszero(sc), "jacobi=$(fmt(jac)), structconst=$(fmt(sc))")

header("5. End to end: choose A, B realising se(3) and check the resulting bracket")

C = se3()
n = size(C, 1)

"""
    skew_decompose(M)

Write an antisymmetric `M` as `Σ_r (A^r_i B^r_j - A^r_j B^r_i)`, by exact skew Gram-Schmidt.
Every antisymmetric matrix of rank `2r` is such a sum of `r` terms, which is what lets an
arbitrary Lie algebra be reached from the Gardner-like ansatz one Greek slot at a time.
"""
function skew_decompose(M)
    m = copy(M)
    nn = size(m, 1)
    terms = Tuple{Vector{Q}, Vector{Q}}[]
    while any(!iszero, m)
        i0, j0 = Tuple(findfirst(!iszero, m))
        piv = m[i0, j0]
        Av = m[:, j0]
        Bv = m[i0, :] ./ piv
        push!(terms, (Av, Bv))
        for i in 1:nn, j in 1:nn

            m[i, j] -= Av[i] * Bv[j] - Av[j] * Bv[i]
        end
    end
    terms
end

Kr = zeros(Q, n, n, n, n)
nterms = Int[]
for a in 1:n
    terms = skew_decompose(C[a, :, :])
    push!(nterms, length(terms))
    for (Av, Bv) in terms, i in 1:n, j in 1:n
        Kr[i, a, j, a] += Av[i] * Bv[j] - Av[j] * Bv[i]
    end
end
Cr = [Kr[i, a, j, a] for a in 1:n, i in 1:n, j in 1:n]
check("skew decomposition of each c^a reproduces se(3)", Cr == C,
    "terms per Greek slot = $nterms")
w = rnd_vec(rng, n; lo = 1, hi = 9, dmax = 3)
J = contract(Kr, w)
check("the Gardner-like 4-bracket built from these mu, sigma is Poisson",
    iszero(first(jacobi_residual(J, lie_poisson_derivative(Cr); normalised = false))))

summary("verify_liepoisson_4bracket.jl")
