#!/usr/bin/env julia
#
# Theorem: J_ij(u) = g_i(u_i) K_ij g_j(u_j) is a Poisson tensor.
#
#     julia --project=scripts scripts/verify_burgers_jacobi_family.jl
#
# The hypotheses that are actually needed, and that the statement of the theorem in the notes
# left implicit:
#
#   (i)  K is a *constant* antisymmetric matrix, independent of u;
#   (ii) each g_i is a function of u_i ALONE.
#
# The special case g_i(u_i) = sqrt(u_i) is the Burgers bracket of Section 1. This script
# verifies the theorem exactly and shows that both hypotheses are sharp: relax either and the
# Jacobi identity fails.
#
# It also verifies the Casimirs C_n = sum_i n_i G_i(u_i) with G_i' = 1/g_i, for every n in
# ker K.
#
# Exact rational arithmetic throughout, so "the residual vanishes" means it is zero and not
# that it is smaller than a tolerance someone chose.

using GeometricBrackets
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary, fmt
include(joinpath(@__DIR__, "rationals.jl"));
using .Rationals

const N = 5
const rng = MersenneTwister(7)

"""
    build(g, dg, K)

J_ij = g_i K_ij g_j together with dJ[l,i,j] = ∂J_ij/∂u_l, where `dg[i,l]` is ∂g_i/∂u_l.

Written out rather than taken from `GaugedBracket`, which carries a single scalar `g`
applied componentwise; the point of sections 1 and 3 is to let g_i vary with the index, and
then to let it depend on components other than its own.
"""
function build(g, dg, K)
    n = length(g)
    J = [g[i] * K[i, j] * g[j] for i in 1:n, j in 1:n]
    dJ = [dg[i, l] * K[i, j] * g[j] + g[i] * K[i, j] * dg[j, l]
          for l in 1:n, i in 1:n, j in 1:n]
    return J, dJ
end

raw_residual(J, dJ) = first(jacobi_residual(J, dJ; normalised = false))

K = rnd_mat(rng, N, :antisymmetric)
u = positive_vec(rng, N)                       # u_i > 0, the domain of √u

header("1. J_ij = g_i(u_i) K_ij g_j(u_j) with g_i a function of u_i alone")

# a deliberately arbitrary, index-dependent g_i(u_i) = a_i u_i^2 + b_i u_i + c_i
a, b, c = rnd_vec(rng, N), rnd_vec(rng, N), rnd_vec(rng, N)
g = [a[i] * u[i]^2 + b[i] * u[i] + c[i] for i in 1:N]
dg = [l == i ? 2 * a[i] * u[i] + b[i] : zero(eltype(u)) for i in 1:N, l in 1:N]
J, dJ = build(g, dg, K)
check("J is antisymmetric", J == -transpose(J))
check("Jacobi residual vanishes for general g_i(u_i)", iszero(raw_residual(J, dJ)))

header("2. The Section 1 case g_i = sqrt(u_i), i.e. J_ij = sqrt(u_i) K_ij sqrt(u_j)")

# Work with sqrt(u_i) as the primitive variable so the arithmetic stays exact: put
# w_i = sqrt(u_i), so u_i = w_i^2, g_i = w_i and dg_i/du_i = 1/(2 w_i). Taking u rational and
# then its square root would leave the field.
w = rnd_vec(rng, N; lo = 1, hi = 9, dmax = 3)
dgw = [l == i ? 1 // (2 * w[i]) : zero(eltype(w)) for i in 1:N, l in 1:N]
J, dJ = build(w, dgw, K)
check("Jacobi residual vanishes for g_i = sqrt(u_i)", iszero(raw_residual(J, dJ)))

# the same bracket through the package's GaugedBracket, which is what the Burgers system uses
gb = GaugedBracket(K, sqrt, u -> inv(2 * sqrt(u)))
usq = [w[i]^2 for i in 1:N]
check("GaugedBracket agrees with the hand-built J to round-off",
    maximum(abs, poisson_matrix(gb, float.(usq)) - float.(J)) < 1e-12)

header("3. Hypothesis (ii) is sharp: g_i depending on all of u breaks Jacobi")

e = rnd_mat(rng, N)
g = [a[i] * u[i]^2 + sum(e[i, m] * u[m] for m in 1:N) for i in 1:N]
dg = [(l == i ? 2 * a[i] * u[i] : zero(eltype(u))) + e[i, l] for i in 1:N, l in 1:N]
J, dJ = build(g, dg, K)
r = raw_residual(J, dJ)
check("Jacobi residual is nonzero when g_i = g_i(u)", !iszero(r), "residual = $(fmt(r))")

header("4. Hypothesis (i) is needed: a u-dependent K breaks Jacobi")

Ku = [K[i, j] * (u[1] + 1) for i in 1:N, j in 1:N]
dKu = [l == 1 ? K[i, j] : zero(eltype(u)) for l in 1:N, i in 1:N, j in 1:N]
J = [w[i] * Ku[i, j] * w[j] for i in 1:N, j in 1:N]
dJ = [dgw[i, l] * Ku[i, j] * w[j] + w[i] * dKu[l, i, j] * w[j] +
      w[i] * Ku[i, j] * dgw[j, l]
      for l in 1:N, i in 1:N, j in 1:N]
r = raw_residual(J, dJ)
check("Jacobi residual is nonzero when K depends on u", !iszero(r), "residual = $(fmt(r))")

header("5. Casimirs: C_n = sum_i n_i G_i(u_i) with G_i' = 1/g_i, for n in ker K")

# N = 5 is odd and an antisymmetric matrix has even rank, so ker K is nontrivial for free.
ker = kernel(K)
check("ker K is nontrivial", size(ker, 2) ≥ 1, "dim ker K = $(size(ker, 2))")
J = [w[i] * K[i, j] * w[j] for i in 1:N, j in 1:N]
for idx in axes(ker, 2)
    n = ker[:, idx]
    # g_i = sqrt(u_i) = w_i  ⟹  G_i(u_i) = 2 sqrt(u_i) = 2 w_i, so dC/du_i = n_i / w_i
    dC = [n[i] / w[i] for i in 1:N]
    flow = J * dC
    check("C_n = 2 sum_i n_i sqrt(u_i) is a Casimir (kernel vector $(idx - 1))",
        all(iszero, flow))
end

summary("verify_burgers_jacobi_family.jl")
