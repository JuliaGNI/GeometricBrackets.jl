#!/usr/bin/env julia
#
# The Gardner-like 4-bracket: three identities that collapse the appendix.
#
#     julia --project=scripts scripts/verify_gardner_4bracket.jl
#
# For K^{iajb} = mu_ia sigma_jb - mu_ja sigma_ib and {f,g}_{s1,s2} = {f,s1;g,s2}, this
# verifies, for arbitrary mu, sigma and arbitrary s -- neither antisymmetry nor diagonality
# is used anywhere:
#
#   (a) J_ij = m_i s_j - s_i m_j, i.e. J = m ^ s is DECOMPOSABLE, with m = mu grad(s1) and
#       s = sigma grad(s2). Hence rank J <= 2, which is why the ansatz cannot reproduce the
#       discrete Gardner operator M^-1 D M^-1, of rank N-1.
#
#   (b) P - Q = [m, s], the Lie bracket of the two vector fields, with P_i = (sigma H mu grad s)_i
#       and Q_i = (mu H sigma grad s)_i, H = Hess(s).
#
#   (c) The 24-term reduced Jacobi condition of the notes equals EXACTLY det(m, P-Q, s).
#       With (b) that says
#
#           Jacobi  <=>  m ^ s ^ [m, s] = 0 ,
#
#       Frobenius involutivity -- the classical criterion [[X^Y, X^Y]] = 2 X^Y^[X,Y] for a
#       decomposable bivector.
#
#   (d) The multi-term ansatz with diagonal mu, sigma and separable s gives
#       J_ij = f_i'(z_i) K_ij f_j'(z_j) with K an ARBITRARY antisymmetric matrix;
#       f_i' = sqrt(z_i) recovers the Burgers bracket of Section 1.
#
# Exact rational arithmetic throughout.

using GeometricBrackets
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary, fmt
include(joinpath(@__DIR__, "rationals.jl"));
using .Rationals

const N = 5
const Q = Rational{BigInt}
const rng = MersenneTwister(11)

"""
    make_s(separable)

A general `s(z) = ½ z·C·z + ⅙ T_abc z_a z_b z_c`, so that `Hess(s)` is NOT constant — which
is what stops the identities below from being trivial. `separable` restricts to
`s = Σ_i f_i(z_i)`, i.e. `C` diagonal and `T` supported on `a = b = c`.
"""
function make_s(separable::Bool)
    C = zeros(Q, N, N)
    T = zeros(Q, N, N, N)
    for a in 1:N, b in a:N

        (separable && a != b) && continue
        C[a, b] = C[b, a] = rnd(rng)
    end
    for a in 1:N, b in a:N, c in b:N
        v = (separable && !(a == b == c)) ? zero(Q) : rnd(rng)
        for p in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a))
            T[p...] = v
        end
    end
    return C, T
end

function grad_hess(C, T, z)
    g = [sum(C[a, b] * z[b] for b in 1:N) +
         sum(T[a, b, c] * z[b] * z[c] for b in 1:N, c in 1:N) // 2 for a in 1:N]
    H = [C[a, l] + sum(T[a, l, c] * z[c] for c in 1:N) for a in 1:N, l in 1:N]
    return g, H
end

"J, dJ, m, s, P, Q and [m,s] for the single-term Gardner-like K (s1 = s2 = s)."
function gardner(mu, sg, C, T, z)
    gs, H = grad_hess(C, T, z)
    m = mu * gs
    s = sg * gs
    dm = mu * H
    ds = sg * H
    J = [m[i] * s[j] - s[i] * m[j] for i in 1:N, j in 1:N]
    dJ = [dm[i, l] * s[j] + m[i] * ds[j, l] - ds[i, l] * m[j] - s[i] * dm[j, l]
          for l in 1:N, i in 1:N, j in 1:N]
    P = sg * H * mu * gs
    Qv = mu * H * sg * gs
    lie = [sum(m[l] * ds[i, l] - s[l] * dm[i, l] for l in 1:N) for i in 1:N]
    return J, dJ, m, s, P, Qv, lie
end

"J_ij straight from the tensor definition, as a cross-check on the closed form."
function contract_K(mu, sg, gs)
    [sum(gs[a] * (mu[i, a] * sg[j, b] - mu[j, a] * sg[i, b]) * gs[b]
     for a in 1:N, b in 1:N)
     for i in 1:N, j in 1:N]
end

"The 3×3 determinant of three vectors restricted to components i, j, k."
function det3(a, b, c, i, j, k)
    a[i] * (b[j] * c[k] - b[k] * c[j]) -
    a[j] * (b[i] * c[k] - b[k] * c[i]) +
    a[k] * (b[i] * c[j] - b[j] * c[i])
end

pad(x, n) = rpad(string(x), n)

header("1-3. J = m ^ s, P - Q = [m, s], and Jacobi = det(m, P-Q, s)")

for mu_kind in (:general, :antisymmetric, :diagonal, :symmetric),
    s_kind in (:general, :separable)

    mu, sg = rnd_mat(rng, N, mu_kind), rnd_mat(rng, N, mu_kind)
    C, T = make_s(s_kind === :separable)
    z = rnd_vec(rng, N; lo = 1, hi = 9, dmax = 2)
    J, dJ, m, s, P, Qv, lie = gardner(mu, sg, C, T, z)
    gs, _ = grad_hess(C, T, z)
    tag = "mu,sigma $(pad(mu_kind, 13)) / s $(pad(s_kind, 9))"

    check("$tag: J from the tensor equals m ^ s", contract_K(mu, sg, gs) == J)
    check("$tag: J antisymmetric", J == -transpose(J))
    r = exact_rank(J)
    check("$tag: rank J <= 2", r ≤ 2, "rank = $r")
    check("$tag: P - Q = [m, s]", P - Qv == lie)

    # The Jacobiator of `jacobiator` is the negative of the form the notes write the identity
    # in -- the two differ by the antisymmetry of J, term for term -- so the identity reads
    # with a minus here. That the check passes at all is what pins the sign down.
    PQ = P - Qv
    Jac = jacobiator(J, dJ)
    ok = all(Jac[i, j, k] == -det3(m, PQ, s, i, j, k) for i in 1:N, j in 1:N, k in 1:N)
    check("$tag: Jacobi residual = det(m, P-Q, s) for all i,j,k", ok)
end

header("4. The Frobenius criterion decides Jacobi")

for (mu_kind, s_kind, expect) in ((:diagonal, :separable, true),
    (:diagonal, :general, false),
    (:general, :separable, false),
    (:general, :general, false))
    mu, sg = rnd_mat(rng, N, mu_kind), rnd_mat(rng, N, mu_kind)
    C, T = make_s(s_kind === :separable)
    z = rnd_vec(rng, N; lo = 1, hi = 9, dmax = 2)
    J, dJ, m, s, P, Qv, _ = gardner(mu, sg, C, T, z)
    PQ = P - Qv
    involutive = all(iszero(det3(m, PQ, s, i, j, k)) for i in 1:N, j in 1:N, k in 1:N)
    jac = first(jacobi_residual(J, dJ; normalised = false))
    tag = "mu,sigma $(pad(mu_kind, 9)) / s $(pad(s_kind, 9))"
    check("$tag: Jacobi <=> m ^ s ^ [m,s] = 0", iszero(jac) == involutive,
        "jacobi=$(iszero(jac) ? "0" : "nonzero"), involutive=$involutive")
    check("$tag: Poisson = $expect", iszero(jac) == expect)
end

header("5. Sufficient condition: sigma H mu symmetric => P = Q => Jacobi")

mu, sg = rnd_mat(rng, N, :diagonal), rnd_mat(rng, N, :diagonal)
C, T = make_s(true)          # separable s ⟹ diagonal Hessian ⟹ everything commutes
z = rnd_vec(rng, N; lo = 1, hi = 9, dmax = 2)
_, H = grad_hess(C, T, z)
SHM = sg * H * mu
check("sigma H mu is symmetric for diagonal mu, sigma and separable s", SHM ==
                                                                        transpose(SHM))
J, dJ, m, s, P, Qv, _ = gardner(mu, sg, C, T, z)
check("P = Q in that case", P == Qv)
check("hence Jacobi holds", iszero(first(jacobi_residual(J, dJ; normalised = false))))

header("6. Multi-term ansatz: arbitrary antisymmetric K, recovering Section 1")

const NC = 4
mus = [rnd_vec(rng, N) for _ in 1:NC]
sgs = [rnd_vec(rng, N) for _ in 1:NC]
Kmat = [sum(mus[x][i] * sgs[x][j] - mus[x][j] * sgs[x][i] for x in 1:NC)
        for i in 1:N, j in 1:N]
K1 = [mus[1][i] * sgs[1][j] - mus[1][j] * sgs[1][i] for i in 1:N, j in 1:N]
check("K is antisymmetric", Kmat == -transpose(Kmat))
r1, rn = exact_rank(K1), exact_rank(Kmat)
check("a single term has rank 2, $NC terms reach higher rank", r1 == 2 && rn > 2,
    "rank(1 term) = $r1, rank($NC terms) = $rn")

# separable s = Σ_i f_i(z_i) with f_i' = p_i z² + q_i z + r_i
p, q, r = rnd_vec(rng, N), rnd_vec(rng, N), rnd_vec(rng, N)
z = rnd_vec(rng, N; lo = 1, hi = 9, dmax = 2)
fp = [p[i] * z[i]^2 + q[i] * z[i] + r[i] for i in 1:N]
fpp = [2 * p[i] * z[i] + q[i] for i in 1:N]
J = [fp[i] * Kmat[i, j] * fp[j] for i in 1:N, j in 1:N]
dJ = [(l == i ? fpp[i] * Kmat[i, j] * fp[j] : zero(Q)) +
      (l == j ? fp[i] * Kmat[i, j] * fpp[j] : zero(Q)) for l in 1:N, i in 1:N, j in 1:N]
check("multi-term diagonal mu, sigma with separable s is Poisson",
    iszero(first(jacobi_residual(J, dJ; normalised = false))))

# f_i' = sqrt(z_i): use w_i = sqrt(z_i) as the primitive variable to stay exact
w = rnd_vec(rng, N; lo = 1, hi = 9, dmax = 2)
J = [w[i] * Kmat[i, j] * w[j] for i in 1:N, j in 1:N]
dJ = [(l == i ? Kmat[i, j] * w[j] // (2 * w[i]) : zero(Q)) +
      (l == j ? w[i] * Kmat[i, j] // (2 * w[j]) : zero(Q)) for l in 1:N, i in 1:N, j in 1:N]
check("f_i' = sqrt(z_i) recovers the Section 1 bracket, and it is Poisson",
    iszero(first(jacobi_residual(J, dJ; normalised = false))))

summary("verify_gardner_4bracket.jl")
