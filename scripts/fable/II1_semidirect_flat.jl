#!/usr/bin/env julia
#
# Idea II.1 -- pointwise flat coordinates for semidirect-product Lie-Poisson brackets.
#
#     julia --project=scripts scripts/fable/II1_semidirect_flat.jl
#
# The 1-D compressible (barotropic) Euler / shallow-water bracket in Lie-Poisson variables
# (rho, m = rho u),
#
#     J(rho, m) = [ 0          d rho            ]
#                 [ rho d      m d + d m        ]        (d = d/dx; sign convention below),
#
# is the push-forward of the CONSTANT bracket  K2 = [0 d; d 0]  in the variables (rho, u)
# under the pointwise map (rho, u) -> (rho, rho u).  Hence, for ANY antisymmetric matrix K
# (a nodal discretisation of d), the block matrix
#
#     J = Dphi K2 Dphi^T,   K2 = [0 K; K 0],   Dphi = [I 0; diag(u) diag(rho)],
#
# i.e.  J_{rho m} = K diag(rho),  J_{m rho} = diag(rho) K,
#       J_{m m}   = diag(u) K diag(rho) + diag(rho) K diag(u),
#
# is exactly Poisson and as local as K.  This script checks
#
#   0. the continuum identities (SymPy): the push-forward reproduces the (rho, m) bracket and its
#      Casimirs; the three-component bracket with entropy (rho, m, sigma) has Casimirs int rho f(s),
#      is NOT constant in (rho, u, s) -- the term s_x/rho survives -- and has a degenerate symbol
#      (rank 2), which is what rules the entropy case out of the construction (II.1.md, Prop. 3);
#      for one advected density a of weight mu (Vect x F_mu, symbolic mu) the chart rho = a^(1/mu)
#      carries the Lie-Poisson bracket  J_ma = mu a d - (1 - mu) a_x  to the (rho, m) bracket, and
#      the symbol entry mu a vanishes exactly for the advected scalar mu = 0;
#   1. Jacobi for J(rho, m) exactly over Q, for random antisymmetric K and random rational data,
#      N = 6..10, with the derivative tensor hand-coded and cross-checked exactly against SymPy
#      differentiation at N = 3 (and by central differences in Float64 at N = 7);
#      CONTROLS (must fail): the direct collocation  J_mm = diag(m) K + K diag(m)  of the same
#      operator, and the natural three-component candidate with entropy;
#  1b. the general theorem with two components per node: P = G K2 G^T is Poisson when G^{-1} is
#      the Jacobian of a chart (pointwise or not; pointwise buys locality); CONTROLS: G^{-1} not a
#      Jacobian, and the Jacobian in the wrong slot G = D eta(z) -- harmless with one component
#      per node (the theorem of the notes), fatal with two;
#   2. the Casimirs  sum_i n_i rho_i  and  sum_i n_i m_i / rho_i,  n in ker K, exactly, and
#      rank J = 2 rank K;
#   3. consistency of the three blocks with  d rho,  rho d,  m d + d m:  order 2 for P1 and P2
#      Lagrange elements with the consistent mass matrix (P2 reaches it late: its Galerkin
#      derivative is only second order at the nodes, the P1 one is fourth), order 2 and 4 for
#      second- and fourth-order central differences (diagonal mass);
#   4. the direct control under refinement with smooth fields: its normalised Jacobiator is
#      nonzero at every resolution and decays at first order (O(1) for rough, random data);
#   5. boundaries: on [0, L] with a nodal basis, the two ways of making K2 antisymmetric --
#      the skew part  (S - S^T)/2  in both blocks, or  S  and  S - B  (B = S + S^T the boundary
#      matrix) in the two off-diagonal blocks -- are both exactly Poisson; the second IS the bracket
#      int (F_u d G_rho - G_u d F_rho) on the finite element space (exact identity), the first the
#      mean of that form and int (F_rho d G_u - G_rho d F_u), the two differing by the boundary
#      term [F_rho G_u - G_rho F_u]; the second encodes the
#      wall condition m = 0 weakly, keeps the mass as an exact Casimir, and carries a sawtooth
#      Casimir on the u block for every N (forced: K_rho m = -K_m rho^T makes the two kernels
#      equal in dimension); the first keeps no physical Casimir at all.  Consistency of the wall
#      closure: second order at fixed distance from the walls, first order at the wall nodes when
#      the data satisfy m = 0 there, O(1/h) at the wall nodes when they do not.
#
# Sign convention: J above is the (+) Lie-Poisson bracket; the physical equations are
# z_t = -J grad H.  The sign plays no role in any check.
#
# Exact rational arithmetic wherever the claim is "this vanishes"; Float64 with a refinement
# study wherever it is "this converges".

using GeometricBrackets
using LinearAlgebra
using Random
using Printf
using SymPyPythonCall

include(joinpath(@__DIR__, "..", "check.jl"));
using .Checks: header, check, summary, fmt
include(joinpath(@__DIR__, "..", "rationals.jl"));
using .Rationals

const Q = Rational{BigInt}
const rng = MersenneTwister(2026)

# ---------------------------------------------------------------------------------------------
# Assembly: periodic or interval Lagrange P1/P2 matrices, exact when T is rational
# ---------------------------------------------------------------------------------------------

"""
    lagrange_matrices(T, p, ne; periodic, L) -> (x, M, S, B)

Nodes `x`, mass matrix `M_kl = ∫ φ_k φ_l`, convection matrix `S_kl = ∫ φ_k φ_l'` and the
boundary matrix `B = S + Sᵀ` (zero on a periodic mesh, `diag(-1, 0, …, 0, 1)` on an interval)
for the nodal Lagrange space of degree `p` on `ne` uniform elements of `[0, L]`.
Exact when `T` is rational and `L` is.
"""
function lagrange_matrices(::Type{T}, p::Int, ne::Int; periodic::Bool, L) where {T}
    h = T(L) / ne
    if p == 1
        Me = h / 6 * T[2 1; 1 2]
        Se = T[-1//2 1//2; -1//2 1//2]
    elseif p == 2
        Me = h / 30 * T[4 2 -1; 2 16 2; -1 2 4]
        Se = T(1 // 6) * T[-3 4 -1; -4 0 4; 1 -4 3]
    else
        throw(ArgumentError("p = 1 or 2 only"))
    end
    N = periodic ? p * ne : p * ne + 1
    M = zeros(T, N, N)
    S = zeros(T, N, N)
    for e in 0:(ne - 1)
        gidx = [mod1(e * p + a + 1, N) for a in 0:p]
        for a in 1:(p + 1), b in 1:(p + 1)

            M[gidx[a], gidx[b]] += Me[a, b]
            S[gidx[a], gidx[b]] += Se[a, b]
        end
    end
    x = [T(i - 1) * h / p for i in 1:N]
    return x, M, S, S + transpose(S)
end

"Central-difference matrices of order 2 or 4 on a periodic grid of N points, spacing h."
function central_difference(::Type{T}, N::Int, h, order::Int) where {T}
    D = zeros(T, N, N)
    c = order == 2 ? Dict(1 => T(1 // 2)) : Dict(1 => T(2 // 3), 2 => T(-1 // 12))
    for i in 1:N, (k, v) in c

        D[i, mod1(i + k, N)] += v / h
        D[i, mod1(i - k, N)] -= v / h
    end
    return D
end

"""
    sbp2(T, N, L) -> (x, W, Q, B)

The classical second-order summation-by-parts operator on N points of [0, L]: diagonal norm
W = h diag(1/2, 1, …, 1, 1/2), Q + Qᵀ = B = diag(-1, 0, …, 0, 1), D = W⁻¹ Q.
"""
function sbp2(::Type{T}, N::Int, L) where {T}
    h = T(L) / (N - 1)
    W = Diagonal([T(i == 1 || i == N ? 1 // 2 : 1) * h for i in 1:N])
    Qm = zeros(T, N, N)
    for i in 1:(N - 1)
        Qm[i, i + 1] += T(1 // 2)
        Qm[i + 1, i] -= T(1 // 2)
    end
    Qm[1, 1] = T(-1 // 2)
    Qm[N, N] = T(1 // 2)
    x = [T(i - 1) * h for i in 1:N]
    return x, Matrix(W), Qm, Qm + transpose(Qm)
end

# ---------------------------------------------------------------------------------------------
# The brackets and their exact derivative tensors, in the variables z = (rho; m)
# ---------------------------------------------------------------------------------------------

"""
    flat_bracket(Ka, Kb, ρ, m) -> (P, dP)

The push-forward `Dφ K2 Dφᵀ` of `K2 = [0 Ka; Kb 0]` (antisymmetric iff `Kb = -Kaᵀ`) under
`(ρ, u) ↦ (ρ, m = ρ u)`, written in `(ρ, m)`:

    P_{ρm} = Ka diag(ρ),  P_{mρ} = diag(ρ) Kb,
    P_{mm} = diag(u) Ka diag(ρ) + diag(ρ) Kb diag(u),   u = m/ρ,

together with `dP[l, i, j] = ∂P_ij / ∂z_l`, exact in the element type.
"""
function flat_bracket(Ka::AbstractMatrix, Kb::AbstractMatrix, ρ::AbstractVector,
        m::AbstractVector)
    N = length(ρ)
    T = promote_type(eltype(Ka), eltype(ρ), eltype(m))
    P = zeros(T, 2N, 2N)
    dP = zeros(T, 2N, 2N, 2N)
    u = m ./ ρ
    for i in 1:N, j in 1:N

        P[i, N + j] = Ka[i, j] * ρ[j]
        P[N + i, j] = ρ[i] * Kb[i, j]
        P[N + i, N + j] = u[i] * Ka[i, j] * ρ[j] + ρ[i] * Kb[i, j] * u[j]
        # ∂/∂ρ_l
        dP[j, i, N + j] += Ka[i, j]
        dP[i, N + i, j] += Kb[i, j]
        dP[i, N + i, N + j] += -u[i] / ρ[i] * Ka[i, j] * ρ[j] + Kb[i, j] * u[j]
        dP[j, N + i, N + j] += u[i] * Ka[i, j] + ρ[i] * Kb[i, j] * (-u[j] / ρ[j])
        # ∂/∂m_l
        dP[N + i, N + i, N + j] += Ka[i, j] * ρ[j] / ρ[i]
        dP[N + j, N + i, N + j] += ρ[i] * Kb[i, j] / ρ[j]
    end
    return P, dP
end

flat_bracket(K::AbstractMatrix, ρ, m) = flat_bracket(K, K, ρ, m)

"The matrix alone, without the (2N)³ derivative tensor, for the refinement studies."
function flat_matrix(Ka::AbstractMatrix, Kb::AbstractMatrix, ρ::AbstractVector, m::AbstractVector)
    u = m ./ ρ
    Dρ, Du = Diagonal(ρ), Diagonal(u)
    return [zeros(eltype(Ka), length(ρ), length(ρ)) Ka*Dρ; Dρ*Kb Du*Ka*Dρ+Dρ*Kb*Du]
end
flat_matrix(K::AbstractMatrix, ρ, m) = flat_matrix(K, K, ρ, m)

"""
    direct_bracket(K, ρ, m) -> (P, dP)

The CONTROL: the same off-diagonal blocks, but `m ∂ + ∂ m` collocated directly as
`diag(m) K + K diag(m)`.  Affine in `(ρ, m)`, so `dP` is constant.
"""
function direct_bracket(K::AbstractMatrix, ρ::AbstractVector, m::AbstractVector)
    N = length(ρ)
    T = promote_type(eltype(K), eltype(ρ), eltype(m))
    P = zeros(T, 2N, 2N)
    dP = zeros(T, 2N, 2N, 2N)
    for i in 1:N, j in 1:N

        P[i, N + j] = K[i, j] * ρ[j]
        P[N + i, j] = ρ[i] * K[i, j]
        P[N + i, N + j] = K[i, j] * (m[i] + m[j])
        dP[j, i, N + j] += K[i, j]
        dP[i, N + i, j] += K[i, j]
        dP[N + i, N + i, N + j] += K[i, j]
        dP[N + j, N + i, N + j] += K[i, j]
    end
    return P, dP
end

"""
    entropy_candidate(K, ρ, m, σ) -> (P, dP)

The natural three-component candidate for Euler with entropy, z = (ρ; m; σ = ρ s): the flat
(ρ, m) blocks above plus `P_{mσ} = diag(σ) K`, `P_{σm} = K diag(σ)`.  It has the right
symbol and the right Casimir `Σ n_i ρ_i`, and it is NOT Poisson (section 0 says why).
"""
function entropy_candidate(K::AbstractMatrix, ρ::AbstractVector, m::AbstractVector,
        σ::AbstractVector)
    N = length(ρ)
    P2, dP2 = flat_bracket(K, ρ, m)
    T = eltype(P2)
    P = zeros(T, 3N, 3N)
    dP = zeros(T, 3N, 3N, 3N)
    P[1:2N, 1:2N] = P2
    dP[1:2N, 1:2N, 1:2N] = dP2
    for i in 1:N, j in 1:N

        P[N + i, 2N + j] = σ[i] * K[i, j]
        P[2N + i, N + j] = K[i, j] * σ[j]
        dP[2N + i, N + i, 2N + j] += K[i, j]
        dP[2N + j, 2N + i, N + j] += K[i, j]
    end
    return P, dP
end

raw_residual(P, dP) = first(jacobi_residual(P, dP; normalised = false))

"Nodal derivative acting on coefficient gradients: K = ½ M⁻¹ (S - Sᵀ) M⁻¹, antisymmetric."
skew_derivative(M, S) = (Minv = inv(M); Minv * (S - transpose(S)) * Minv / 2)

# =============================================================================================

header("0. Continuum identities (SymPy)")

@syms x::real
@syms ρf() uf() sf() af() bf() cf()
ρ, u, s, a, b = ρf(x), uf(x), sf(x), af(x), bf(x)
∂(e) = diff(e, x)
m = ρ * u
σ = ρ * s

# J(ρ,m) applied to (a, b) = (δF/δρ, δF/δm)
Jρ_direct = ∂(ρ * b)
Jm_direct = ρ * ∂(a) + m * ∂(b) + ∂(m * b)
# The push-forward Dφ [0 ∂; ∂ 0] Dφᵀ (a, b), built through the chain rather than restated: the row
# written out is the row the check cannot then test.
w1, w2 = a + u * b, ρ * b                       # Dφᵀ(a,b),  Dφ = [1 0; u ρ]
k1, k2 = ∂(w2), ∂(w1)                           # [0 ∂; ∂ 0] (w1, w2)
Jρ_push, Jm_push = k1, u * k1 + ρ * k2          # Dφ (k1, k2)
check("push-forward of [0 ∂; ∂ 0] under (ρ,u) ↦ (ρ,ρu) is the (ρ,m) bracket: ρ row",
    simplify(Jρ_push - Jρ_direct) == 0)
check("push-forward of [0 ∂; ∂ 0] under (ρ,u) ↦ (ρ,ρu) is the (ρ,m) bracket: m row",
    simplify(Jm_push - Jm_direct) == 0)

# Casimirs of the two-component bracket: ∫ρ and ∫u = ∫m/ρ
# gradients (C_ρ, C_m) of the densities ρ and m/ρ
for (name, Cρ, Cm, expect) in (("∫ρ", Sym(1), Sym(0), true), (
    "∫ m/ρ", -m / ρ^2, 1 / ρ, true),
    ("∫ m (control)", Sym(0), Sym(1), false))
    r1 = simplify(∂(ρ * Cm))
    r2 = simplify(ρ * ∂(Cρ) + m * ∂(Cm) + ∂(m * Cm))
    iscas = (r1 == 0) && (r2 == 0)
    check(
        "$name is $(expect ? "" : "NOT ")a Casimir of the (ρ,m) bracket", iscas == expect,
        expect ? "" : "m row = $r2")
end

# Three components with entropy, z = (ρ, m, σ = ρ s):
#   J3 = [0 ∂ρ 0; ρ∂ m∂+∂m σ∂; 0 ∂σ 0]
function J3(Cρ, Cm, Cσ)
    (simplify(∂(ρ * Cm)),
        simplify(ρ * ∂(Cρ) + m * ∂(Cm) + ∂(m * Cm) + σ * ∂(Cσ)),
        simplify(∂(σ * Cm)))
end
for (fname, f, fp) in (("s^2", s^2, 2s), ("s^3", s^3, 3s^2), ("exp(s)", exp(s), exp(s)))
    # C = ∫ ρ f(s):  C_ρ = f - s f',  C_m = 0,  C_σ = f'
    r = J3(f - s * fp, Sym(0), fp)
    check("∫ ρ f(s) is a Casimir of the entropy bracket, f = $fname", all(==(0), r))
    # C = ∫ f(s):  C_ρ = -s f'/ρ,  C_σ = f'/ρ
    r = J3(-s * fp / ρ, Sym(0), fp / ρ)
    check("∫ f(s) is NOT a Casimir of the entropy bracket, f = $fname",
        simplify(r[2] + fp * ∂(s)) == 0 && r[2] != 0, "m row = $(r[2])")
end
r = J3(-m / ρ^2, 1 / ρ, Sym(0))
check("∫ m/ρ is NOT a Casimir once entropy is present", r[3] != 0, "σ row = $(r[3])")

# The entropy bracket in the candidate flat variables (ρ, u, s):  J = Dψ J3 Dψᵀ with
# ψ(ρ, m, σ) = (ρ, m/ρ, σ/ρ).  The term s_x/ρ survives, so (ρ, u, s) are NOT flat coordinates;
# and the symbol g of J3 is degenerate, so no pointwise change of variables can make the
# bracket constant and nondegenerate.  (That none can make it constant and degenerate either is
# the frozen-coordinate argument in II.1.md: a constant degenerate bracket freezes its kernel
# coordinate pointwise, whereas s is advected.)
let A = a, B = b, C = cf(x)
    a3, b3, c3 = A - u / ρ * B - s / ρ * C, B / ρ, C / ρ      # Dψᵀ (A, B, C)
    r1, r2, r3 = J3(a3, b3, c3)
    Jρ, Ju, Js = r1, -u / ρ * r1 + r2 / ρ, -s / ρ * r1 + r3 / ρ   # Dψ (r1, r2, r3)
    check(
        "in (ρ,u,s) the entropy bracket is [0 ∂ 0; ∂ 0 -s_x/ρ; 0 s_x/ρ 0] -- not constant",
        simplify(Jρ - ∂(B)) == 0 && simplify(Ju - (∂(A) - ∂(s) / ρ * C)) == 0 &&
            simplify(Js - ∂(s) / ρ * B) == 0)
    # Read the symbol off J3 rather than typing it: the claim is about the bracket defined above,
    # and a hand-entered matrix would make the rank statement one about the literal instead.
    r3 = J3(A, B, C)
    g3 = [expand(r3[i]).coeff(∂(z)) for i in 1:3, z in (A, B, C)]
    check("the symbol of the entropy bracket is [0 ρ 0; ρ 2m σ; 0 σ 0]",
        all(iszero, simplify.(g3 - [Sym(0) ρ Sym(0); ρ 2m σ; Sym(0) σ Sym(0)])),
        "g = " * string(g3))
    check(
        "the symbol g = [0 ρ 0; ρ 2m σ; 0 σ 0] of the entropy bracket has rank 2: g (σ, 0, -ρ) = 0",
        all(==(0), simplify.(g3 * [σ, Sym(0), -ρ])) && simplify(g3[1, 2] * g3[2, 1]) != 0)
end

# One advected quantity a of weight μ (Vect(S¹) ⋉ F_μ, the variable a ∈ F_μ).  Its test functions
# b = δF/δa have weight 1 - μ, on which ξ∂ acts by ξ b_x + (1 - μ) ξ_x b, so the coupling
# ∫ a (F_m · G_a) gives  J_ma b = a b_x - (1 - μ) ∂(a b) = μ a b_x - (1 - μ) a_x b  and
# J_am c = -(J_ma)† c = μ a c_x + a_x c.  Symbol entry μ a: degenerate iff μ = 0 (advected scalar).
# Claim (Corollary 4): ρ = a^{1/μ} is a 1-density and (ρ, m) carries the (ρ, m) bracket, i.e. the
# push-forward of J(ρ, m) under (ρ, m) ↦ (a, m) = (ρ^μ, m) is the Vect ⋉ F_μ bracket.  Checked for
# symbolic μ.
let
    @syms μ::positive
    # the Lie-Poisson coupling for an independent field a of weight μ, applied to test functions
    lp_ma(a, b) = μ * a * ∂(b) - (1 - μ) * ∂(a) * b       # J_ma b
    lp_am(a, c) = μ * a * ∂(c) + ∂(a) * c                 # J_am c = -(J_ma)† c
    check(
        "Vect ⋉ F_μ: J_am = -(J_ma)† (the coupling is antisymmetric): c J_ma b + b J_am c is a divergence",
        simplify(a * lp_ma(σ, b) + b * lp_am(σ, a) - ∂(μ * σ * a * b)) == 0)
    check("Vect ⋉ F_μ: the symbol entry of the coupling is μ a (coefficient of b_x)",
        simplify(lp_ma(σ, b).coeff(∂(b)) - μ * σ) == 0)
    check(
        "Vect ⋉ F_0 (advected scalar): the coupling J_ms = -s_x has no derivative, symbol zero",
        simplify(lp_ma(σ, b).subs(μ, 0) + ∂(σ) * b) == 0)
    # push-forward of J(ρ,m) under Ψ = (ρ^μ, m): J^Ψ = DΨ J DΨᵀ with DΨ = diag(Ja, 1), Ja = ∂a/∂ρ:
    #   J^Ψ_am c = Ja · J_ρm c = Ja ∂(ρ c);   J^Ψ_ma b = J_mρ (Ja b) = ρ ∂(Ja b)
    aμ = ρ^μ
    Ja = μ * ρ^(μ - 1)
    check(
        "Vect ⋉ F_μ, symbolic μ: push-forward of J(ρ,m) under a = ρ^μ reproduces J_am = μ a ∂ + a_x",
        simplify(expand(Ja * ∂(ρ * a) - lp_am(aμ, a))) == 0)
    check(
        "Vect ⋉ F_μ, symbolic μ: push-forward of J(ρ,m) under a = ρ^μ reproduces J_ma = μ a ∂ - (1-μ) a_x",
        simplify(expand(ρ * ∂(Ja * b) - lp_ma(aμ, b))) == 0)
end

# =============================================================================================

header("1. Jacobi identity, exactly over Q, N = 6..10; controls must fail")

for N in 6:10
    K = rnd_mat(rng, N, :antisymmetric)
    ρq = positive_vec(rng, N)
    uq = rnd_vec(rng, N)
    mq = ρq .* uq
    P, dP = flat_bracket(K, ρq, mq)
    check("N = $N: J = Dφ K2 Dφᵀ is antisymmetric", P == -transpose(P))
    check("N = $N: Jacobiator of J(ρ,m) vanishes exactly", iszero(raw_residual(P, dP)))

    Pd, dPd = direct_bracket(K, ρq, mq)
    rd = jacobi_residual(Pd, dPd)
    check("N = $N: CONTROL direct collocation diag(m)K + Kdiag(m) violates Jacobi",
        rd > 1 // 50,
        "normalised residual = $(fmt(rd)) ≈ $(@sprintf("%.3f", float(rd)))")

    sq = rnd_vec(rng, N)
    P3, dP3 = entropy_candidate(K, ρq, mq, ρq .* sq)
    r3 = jacobi_residual(P3, dP3)
    check(
        "N = $N: CONTROL three-component entropy candidate violates Jacobi", r3 > 1 // 50,
        "normalised residual ≈ $(@sprintf("%.3f", float(r3)))")
end

# cross-check the hand-coded derivative tensor against central differences (Float64)
let N = 7
    K = float.(rnd_mat(rng, N, :antisymmetric))
    ρ0 = float.(positive_vec(rng, N))
    m0 = ρ0 .* float.(rnd_vec(rng, N))
    _, dP = flat_bracket(K, ρ0, m0)
    v = randn(rng, 2N)
    ε = 1e-6
    Pp, _ = flat_bracket(K, ρ0 + ε * v[1:N], m0 + ε * v[(N + 1):2N])
    Pm, _ = flat_bracket(K, ρ0 - ε * v[1:N], m0 - ε * v[(N + 1):2N])
    fd = (Pp - Pm) / (2ε)
    an = sum(v[l] * dP[l, :, :] for l in 1:2N)
    err = maximum(abs, fd - an) / maximum(abs, an)
    check("hand-coded dP agrees with a central difference", err < 1e-6,
        @sprintf("rel. err = %.1e", err))
end

# ... and exactly, by symbolic differentiation at N = 3: this is the exact route for dP.  The
# entries of P are rational functions of (ρ, m); `flat_bracket` differentiates them by hand, and
# SymPy differentiates the assembled matrix.  The two must agree identically in the symbols.
let N = 3
    ρs = [Sym("ρ$i") for i in 1:N]
    ms = [Sym("m$i") for i in 1:N]
    Ks = [Sym("k$(min(i, j))$(max(i, j))") * (i < j ? 1 : i > j ? -1 : 0)
          for i in 1:N, j in 1:N]
    Ps, dPs = flat_bracket(Ks, Ks, ρs, ms)
    zs = [ρs; ms]
    ok = true
    for l in 1:(2N), i in 1:(2N), j in 1:(2N)
        ok &= simplify(dPs[l, i, j] - diff(Ps[i, j], zs[l])) == 0
    end
    check("hand-coded dP equals the symbolic derivative of P entry by entry (N = 3, exact)", ok)
    check("symbolic P is antisymmetric", all(simplify(Ps[i, j] + Ps[j, i]) == 0
    for i in 1:(2N), j in 1:(2N)))
end

# the same statement with the actual P1 finite element K (rational on [0,1], periodic)
for ne in (7, 8)
    _, M, S, _ = lagrange_matrices(Q, 1, ne; periodic = true, L = 1)
    K = skew_derivative(M, S)
    ρq = positive_vec(rng, ne)
    mq = ρq .* rnd_vec(rng, ne)
    P, dP = flat_bracket(K, ρq, mq)
    check("P1 periodic FE, N = $ne: Jacobiator vanishes exactly", iszero(raw_residual(P, dP)))
end

# =============================================================================================

header("1b. The general theorem: any pointwise flat chart η with n = 2 components per node; controls: non-closed factors")

"""
    factored_bracket(K2, L, H, z; inverse) -> (P, dP)

`P = G K2 Gᵀ` with its exact derivative tensor, where `A(z)_ip = L_ip + Σ_q H_ipq z_q` and either
`G = A⁻¹` (`inverse = true`) or `G = A` (`inverse = false`).

With `H_ipq` symmetric in `(p, q)`, `A` is the Jacobian of the quadratic map
`η_i = L_ip z_p + ½ H_ipq z_p z_q`.  The theorem needs `G⁻¹ = Dη`, i.e. the flat chart is
`z̄ = η(z)` and `P = Dη⁻¹ K2 Dη⁻ᵀ` is the constant bracket written in the `z` coordinates; that is
`inverse = true`.  `inverse = false` puts the Jacobian in the wrong slot -- `G = Dη(z)` itself --
which is Poisson only when `Dη⁻¹` happens to be a Jacobian too (always in one component per node,
not in general).  A non-symmetric `H` makes `A` the Jacobian of nothing.
"""
function factored_bracket(K2, L, H, z; inverse::Bool)
    n = length(z)
    A = [L[i, p] + sum(H[i, p, q] * z[q] for q in 1:n) for i in 1:n, p in 1:n]
    dA = [H[:, :, l] for l in 1:n]                       # ∂_l A_ip = H_ipl
    if inverse
        G = inv(A)
        dG = [-G * dA[l] * G for l in 1:n]                # ∂_l A⁻¹ = -A⁻¹ (∂_l A) A⁻¹
    else
        G, dG = A, dA
    end
    P = G * K2 * transpose(G)
    dP = zeros(eltype(P), n, n, n)
    for l in 1:n
        dP[l, :, :] = dG[l] * K2 * transpose(G) + G * K2 * transpose(dG[l])
    end
    return P, dP
end

let N = 5, n = 2N
    node(k) = (k + 1) ÷ 2
    nodedist(i, j) = min(mod(node(i) - node(j), N), mod(node(j) - node(i), N))
    # K2: antisymmetric, nearest-neighbour in the nodes (periodic), like a nodal derivative
    K2 = rnd_mat(rng, n, :antisymmetric)
    for i in 1:n, j in 1:n

        nodedist(i, j) ≤ 1 || (K2[i, j] = 0)
    end
    z = rnd_vec(rng, n)
    L = zeros(Q, n, n)
    H = zeros(Q, n, n, n)                    # pointwise, symmetric: a Jacobian
    Hbad = zeros(Q, n, n, n)                 # pointwise, NOT symmetric: not a Jacobian
    for i in 1:n, p in 1:n

        node(i) == node(p) || continue
        L[i, p] = rnd(rng)
        for q in 1:n
            node(q) == node(i) || continue
            Hbad[i, p, q] = rnd(rng)
            p ≤ q && (H[i, p, q] = H[i, q, p] = rnd(rng))
        end
    end
    Lfull = rnd_mat(rng, n)                  # not pointwise, but still a Jacobian
    Hfull = zeros(Q, n, n, n)
    for i in 1:n, p in 1:n, q in p:n
        Hfull[i, p, q] = Hfull[i, q, p] = rnd(rng)
    end

    P, dP = factored_bracket(K2, L, H, z; inverse = true)
    check(
        "pointwise quadratic chart η, n = 2 per node, P = Dη⁻¹ K2 Dη⁻ᵀ: Jacobiator vanishes exactly",
        iszero(raw_residual(P, dP)))
    check("pointwise η: P inherits the node-bandwidth of K2 (locality)",
        all(iszero(P[i, j]) for i in 1:n, j in 1:n if nodedist(i, j) > 1))
    P, dP = factored_bracket(K2, Lfull, Hfull, z; inverse = true)
    check(
        "non-pointwise η: Jacobiator still vanishes exactly (Jacobi needs a chart, not locality)",
        iszero(raw_residual(P, dP)))
    check("non-pointwise η: locality is lost",
        any(!iszero(P[i, j]) for i in 1:n, j in 1:n if nodedist(i, j) > 1))
    P, dP = factored_bracket(K2, L, Hbad, z; inverse = true)
    rb = jacobi_residual(P, dP)
    # exact arithmetic: nonzero is the verdict; the normalised value is small only because inv(A)
    # of random rationals produces a few large entries that dominate the normalisation
    check("CONTROL pointwise G(z) with G⁻¹ not a Jacobian (H_ipq ≠ H_iqp): Jacobi fails",
        !iszero(rb),
        "normalised residual ≈ $(@sprintf("%.3f", float(rb))) (exactly nonzero)")
    P, dP = factored_bracket(K2, L, H, z; inverse = false)
    rb = jacobi_residual(P, dP)
    check(
        "CONTROL the Jacobian in the wrong slot, P = Dη K2 Dηᵀ with Dη at z: Jacobi fails for n = 2 per node",
        rb > 1 // 50, "normalised residual ≈ $(@sprintf("%.3f", float(rb)))")
    # ... and the one-component-per-node case, where every diagonal G is a Jacobian and so is G⁻¹
    K1 = rnd_mat(rng, N, :antisymmetric)
    z1 = rnd_vec(rng, N)
    L1 = zeros(Q, N, N)
    H1 = zeros(Q, N, N, N)
    for i in 1:N
        L1[i, i] = rnd(rng; lo = 1, hi = 6)
        H1[i, i, i] = rnd(rng)
    end
    P, dP = factored_bracket(K1, L1, H1, z1; inverse = false)
    check(
        "one component per node: the wrong slot is harmless (Theorem thm:discrete-poisson-bracket)",
        iszero(raw_residual(P, dP)))
end

# =============================================================================================

header("2. Casimirs: Σ n_i ρ_i and Σ n_i m_i/ρ_i for n ∈ ker K, exactly; rank J = 2 rank K")

for (label, K) in (("random antisymmetric K, N = 7", rnd_mat(rng, 7, :antisymmetric)),
    ("P1 periodic FE, N = 7",
    skew_derivative(lagrange_matrices(Q, 1, 7; periodic = true, L = 1)[2:3]...)),
    ("P1 periodic FE, N = 8",
    skew_derivative(lagrange_matrices(Q, 1, 8; periodic = true, L = 1)[2:3]...)))
    N = size(K, 1)
    ρq = positive_vec(rng, N)
    mq = ρq .* rnd_vec(rng, N)
    P, _ = flat_bracket(K, ρq, mq)
    ker = kernel(K)
    check("$label: dim ker K = $(size(ker, 2))", size(ker, 2) ≥ 1)
    for idx in axes(ker, 2)
        n = ker[:, idx]
        gC1 = [n; zeros(Q, N)]                          # ∇ Σ n_i ρ_i
        gC2 = [-n .* mq ./ ρq .^ 2; n ./ ρq]            # ∇ Σ n_i m_i/ρ_i
        check("$label: Σ n_i ρ_i is an exact Casimir (kernel vector $idx)", all(iszero, P *
                                                                                        gC1))
        check("$label: Σ n_i m_i/ρ_i is an exact Casimir (kernel vector $idx)", all(iszero, P *
                                                                                            gC2))
    end
    check("$label: rank J = 2 rank K", exact_rank(P) == 2 * exact_rank(K),
        "rank J = $(exact_rank(P)), rank K = $(exact_rank(K))")
    if occursin("FE", label)
        _, M, _, _ = lagrange_matrices(Q, 1, N; periodic = true, L = 1)
        w = M * ones(Q, N)
        check(
            "$label: the mass weights w = M 1 lie in ker K (discrete mass and circulation)",
            all(iszero, K * w))
    end
end

# =============================================================================================

header("3. Consistency of the blocks with ∂ρ, ρ∂, m∂+∂m")

ρx(x) = 2.0 + sin(x) + 0.3 * cos(2x)
dρx(x) = cos(x) - 0.6 * sin(2x)
ux(x) = 1.0 + 0.5 * cos(x) + 0.2 * sin(3x)
dux(x) = -0.5 * sin(x) + 0.6 * cos(3x)
ax(x) = cos(2x) + 0.3 * sin(x)
dax(x) = -2 * sin(2x) + 0.3 * cos(x)
bx(x) = sin(x) + 0.5 * cos(3x)
dbx(x) = cos(x) - 1.5 * sin(3x)

"Exact continuum J (a, b) at the nodes: (∂(ρ b), ρ a' + 2 m b' + m' b)."
function exact_rows(x)
    ρ, dρ, u, du = ρx.(x), dρx.(x), ux.(x), dux.(x)
    da, b, db = dax.(x), bx.(x), dbx.(x)
    m, dm = ρ .* u, dρ .* u .+ ρ .* du
    return dρ .* b .+ ρ .* db, ρ .* da .+ 2 .* m .* db .+ dm .* b
end

"Relative max error of J [M a; M b] against the exact rows, for K acting on coefficient gradients."
function block_error(x, M, K)
    P = flat_matrix(K, ρx.(x), ρx.(x) .* ux.(x))
    v = P * [M * ax.(x); M * bx.(x)]
    eρ, em = exact_rows(x)
    N = length(x)
    return maximum(abs, v[1:N] - eρ) / maximum(abs, eρ),
    maximum(abs, v[(N + 1):2N] - em) / maximum(abs, em)
end

function rates(errs, hs)
    [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]
end

function build_lagrange(p, ne)
    x, M, S, _ = lagrange_matrices(Float64, p, ne; periodic = true, L = 2π)
    return x, M, skew_derivative(M, S)
end

"Diagonal mass h·I and K = D/h, so that K acting on the coefficient gradient h·a returns D a."
function build_cd(order, N)
    h = 2π / N
    return [(i - 1) * h for i in 1:N], h * Matrix{Float64}(I, N, N),
    central_difference(Float64, N, h, order) / h
end

for (label, order, sizes, build) in (
    ("P1 Lagrange, consistent mass", 2, (16, 32, 64, 128), ne -> build_lagrange(1, ne)),
# the P2 Galerkin derivative M⁻¹S w is only second order at the nodes and carries a large
# constant (the P1 one is fourth order), so P2 needs finer meshes to reach its asymptotic rate
    (
    "P2 Lagrange, consistent mass", 2, (16, 32, 64, 128, 256), ne -> build_lagrange(2, ne)),
    ("central differences, order 2", 2, (16, 32, 64, 128), N -> build_cd(2, N)),
    ("central differences, order 4", 4, (16, 32, 64, 128), N -> build_cd(4, N)))
    eρs, ems, hs = Float64[], Float64[], Float64[]
    for n in sizes
        xn, M, K = build(n)
        eρ, em = block_error(xn, Matrix(M), K)
        push!(eρs, eρ)
        push!(ems, em)
        push!(hs, xn[2] - xn[1])
    end
    rρ, rm = rates(eρs, hs), rates(ems, hs)
    println("   $label: ρ-row errors ", join([@sprintf("%.2e", e) for e in eρs], " "),
        "   rates ", join([@sprintf("%.2f", r) for r in rρ], " "))
    println("   $label: m-row errors ", join([@sprintf("%.2e", e) for e in ems], " "),
        "   rates ", join([@sprintf("%.2f", r) for r in rm], " "))
    check(
        "$label: ρ-row (K diag ρ ≈ ∂ρ) converges at order ≥ $order", rρ[end] > order - 0.2,
        @sprintf("finest rate = %.2f", rρ[end]))
    check("$label: m-row (ρK ≈ ρ∂, uKρ + ρKu ≈ m∂+∂m) converges at order ≥ $order",
        rm[end] > order - 0.2, @sprintf("finest rate = %.2f", rm[end]))
end

# sanity: the hand-assembled P2 matrices agree with the package's LagrangeSpace
let ne = 12
    s = LagrangeSpace(2, ne)
    _, M, S, _ = lagrange_matrices(Float64, 2, ne; periodic = true, L = 2π)
    dM = maximum(abs, M - Matrix(mass_matrix(s)))
    dS = maximum(abs, S - Matrix(derivative_matrix(s)))
    check("hand-assembled P2 matrices agree with LagrangeSpace(2, $ne)",
        dM < 1e-12 && dS < 1e-12,
        @sprintf("max|ΔM| = %.1e, max|ΔS| = %.1e", dM, dS))
end

# =============================================================================================

header("4. The direct control under refinement, smooth fields: how the normalised Jacobiator scales")

# Direct minus push-forward is E_ij = K_ij (ρ_i - ρ_j)(u_i - u_j) in the mm block, which for smooth
# fields and a nearest-neighbour K is O(h) against the O(1/h) of J; its derivative ∂E ~ K (u_i - u_j)
# is O(1) against ∂J ~ K ~ 1/h.  So the normalised Jacobiator of the direct bracket should decay
# like h for smooth fields (and is O(1) for the rough, random data of section 1).  Measured here.
let res_d = Float64[], res_f = Float64[], hs = Float64[]
    for ne in (9, 17, 33, 65)
        x, M, S, _ = lagrange_matrices(Float64, 1, ne; periodic = true, L = 2π)
        K = skew_derivative(M, S)
        ρ0, m0 = ρx.(x), ρx.(x) .* ux.(x)
        Pd, dPd = direct_bracket(K, ρ0, m0)
        Pf, dPf = flat_bracket(K, ρ0, m0)
        push!(res_d, jacobi_residual(Pd, dPd))
        push!(res_f, jacobi_residual(Pf, dPf))
        push!(hs, 2π / ne)
    end
    rd = rates(res_d, hs)
    println("   direct:       ", join([@sprintf("%.3f", r) for r in res_d], "  "),
        "   rates ", join([@sprintf("%.2f", r) for r in rd], " "))
    println("   push-forward: ", join([@sprintf("%.1e", r) for r in res_f], "  "))
    check("direct collocation violates Jacobi at every resolution (residual ≫ round-off)",
        minimum(res_d) > 1e-3, @sprintf("range %.3f .. %.3f", minimum(res_d),
            maximum(res_d)))
    check(
        "direct collocation, smooth fields: the normalised residual decays at first order, not faster",
        0.7 < rd[end] < 1.3, @sprintf("finest rate = %.2f", rd[end]))
    check("push-forward: normalised residual at round-off for the FE K",
        maximum(res_f) < 1e-11,
        @sprintf("max = %.1e", maximum(res_f)))
end

# =============================================================================================

header("5. Boundaries: Ω = [0, 1], nodal basis, two antisymmetric closures of K2")

for (label, p, nes) in (("P1", 1, (6, 7)), ("P2", 2, (3, 4)))
    for ne in nes
        _, M, S, B = lagrange_matrices(Q, p, ne; periodic = false, L = 1)
        N = size(M, 1)
        Minv = inv(M)
        check("$label, N = $N: S + Sᵀ = B = diag(-1, 0, …, 0, 1) (summation by parts)",
            B == Diagonal([i == 1 ? -1 : i == N ? 1 : 0 for i in 1:N]))
        ρq = positive_vec(rng, N)
        mq = ρq .* rnd_vec(rng, N)
        w = M * ones(Q, N)

        # (a) skew closure: the same K = ½ M⁻¹(S - Sᵀ)M⁻¹ in both blocks
        Ks = skew_derivative(M, S)
        P, dP = flat_bracket(Ks, ρq, mq)
        check("$label, N = $N, skew closure: Jacobiator vanishes exactly", iszero(raw_residual(P, dP)))
        ker = kernel(Ks)
        check(
            "$label, N = $N, skew closure: dim ker K = $(size(ker, 2)) ($(isodd(N) ? "odd" : "even") N)",
            size(ker, 2) == (isodd(N) ? 1 : 0))
        check("$label, N = $N, skew closure: the mass Σ w_i ρ_i is NOT a Casimir", !all(iszero, Ks *
                                                                                                w))
        if size(ker, 2) == 1
            c = Minv * ker[:, 1]
            c = c / c[findfirst(!iszero, c)]
            println("   $label, N = $N, skew closure: kernel vector M⁻¹n = ", join(fmt.(c), " "))
        end

        # (b) wall closure: K_ρm = M⁻¹(S - B)M⁻¹ (conservative, zero flux), K_mρ = M⁻¹SM⁻¹
        Ka = Minv * (S - B) * Minv
        Kb = Minv * S * Minv
        P, dP = flat_bracket(Ka, Kb, ρq, mq)
        # Kb = -Kaᵀ holds by construction here (B is returned as S + Sᵀ), so on its own it is not a
        # test; it is stated alongside the Jacobiator, which is.
        check(
            "$label, N = $N, wall closure: K2 antisymmetric and Jacobiator vanishes exactly",
            Kb == -transpose(Ka) && iszero(raw_residual(P, dP)))
        gC1 = [w; zeros(Q, N)]
        gC2 = [-w .* mq ./ ρq .^ 2; w ./ ρq]
        check("$label, N = $N, wall closure: mass Σ w_i ρ_i is an exact Casimir", all(iszero, P *
                                                                                              gC1))
        check("$label, N = $N, wall closure: circulation Σ w_i m_i/ρ_i is NOT a Casimir",
            !all(iszero, P * gC2))
        kb, ka = kernel(Kb), kernel(Ka)
        check("$label, N = $N, wall closure: ker K_mρ = span(M 1), the mass",
            size(kb, 2) == 1 && all(iszero, Kb * w))
        check("$label, N = $N, wall closure: ker K_ρm has dimension $(size(ka, 2))", size(ka, 2) ≥
                                                                                     1)
        for idx in axes(ka, 2)
            c = Minv * ka[:, idx]
            c = c / c[findfirst(!iszero, c)]
            println("   $label, N = $N, wall closure: u-block kernel vector M⁻¹n = ", join(fmt.(c), " "))
        end
        # K_ρm = -K_mρᵀ forces dim ker K_ρm = dim ker K_mρ, so the mass Casimir comes with a second,
        # u-linear Casimir for EVERY N -- one with no physical counterpart on an interval with
        # walls.  That equality of dimensions is not asserted: dim ker A = dim ker Aᵀ holds for any
        # square A, so it would pass whatever the assembly did.  The content is the dimension
        # itself, checked above, and the kernel vector printed below.
        if p == 1
            saw = M * [Q((-1)^i) for i in 1:N]
            check(
                "$label, N = $N, wall closure: Σ (M s)_i m_i/ρ_i with s = (-1)^i is an exact spurious Casimir",
                all(iszero, P * [-saw .* mq ./ ρq .^ 2; saw ./ ρq]))
        end
        check("$label, N = $N, wall closure: rank J = 2N - 1 - dim ker K_ρm",
            exact_rank(P) == 2N - 1 - size(ka, 2), "rank J = $(exact_rank(P))")
    end
end

# Which continuous bracket each closure discretises -- an exact identity on the finite element
# space.  With nodal gradient values f = (f_ρ, f_u), g = (g_ρ, g_u) and coefficient gradients M f,
# M g, the discrete bracket is (M f_ρ)ᵀ K_ρu (M g_u) + (M f_u)ᵀ K_uρ (M g_ρ), and
#     fᵀ S g = Σ_kl f_k g_l ∫ φ_k φ_l' = ∫ f_h g_h'
# is the Galerkin integral.  So the wall closure is  ∫ (F_u ∂G_ρ - G_u ∂F_ρ)  exactly, the skew
# closure the mean of that and  ∫ (F_ρ ∂G_u - G_ρ ∂F_u), and the two forms differ by the boundary
# term  [F_ρ G_u - G_ρ F_u]_0^L = f_ρᵀ B g_u - g_ρᵀ B f_u, which vanishes iff the u-gradients
# vanish at the walls.
for p in (1, 2)
    x, M, S, B = lagrange_matrices(Q, p, 5; periodic = false, L = 1)
    N = size(M, 1)
    Minv = inv(M)
    Ka, Kb, Ks = Minv * (S - B) * Minv, Minv * S * Minv, skew_derivative(M, S)
    fρ, fu, gρ, gu = (rnd_vec(rng, N) for _ in 1:4)
    # The four checks below all reduce to S + Sᵀ = B, which holds by construction; what they are
    # really about is that S is the Galerkin derivative in the first place.  Verify that against
    # two integrands the space represents exactly, for which the answer is known in closed form:
    # (f_h, g_h) = (1, x) gives ∫ 1 dx = L, and (x, x) gives ∫ x dx = L²/2, with L = 1 here.
    check("P$p interval: S is the Galerkin derivative -- <1, x'> = L",
        transpose(ones(Q, N)) * S * x == 1, fmt(transpose(ones(Q, N)) * S * x))
    check("P$p interval: S is the Galerkin derivative -- <x, x'> = L^2/2",
        transpose(x) * S * x == 1 // 2, fmt(transpose(x) * S * x))
    function bracket(Kρu, Kuρ)
        transpose(M * fρ) * Kρu * (M * gu) + transpose(M * fu) * Kuρ * (M * gρ)
    end
    form_u = transpose(fu) * S * gρ - transpose(gu) * S * fρ       # ∫ (F_u ∂G_ρ - G_u ∂F_ρ)
    form_ρ = transpose(fρ) * S * gu - transpose(gρ) * S * fu       # ∫ (F_ρ ∂G_u - G_ρ ∂F_u)
    check(
        "P$p interval: the wall closure is exactly ∫(F_u ∂G_ρ - G_u ∂F_ρ) on the FE space",
        bracket(Ka, Kb) == form_u)
    check(
        "P$p interval: the skew closure is exactly the mean of the two integrated-by-parts forms",
        bracket(Ks, Ks) == (form_u + form_ρ) / 2)
    check("P$p interval: the two forms differ by the boundary term [F_ρ G_u - G_ρ F_u]",
        form_ρ - form_u == transpose(fρ) * B * gu - transpose(gρ) * B * fu)
    fu0, gu0 = copy(fu), copy(gu)
    fu0[1] = fu0[N] = gu0[1] = gu0[N] = 0
    check("P$p interval: the two forms coincide when δF/δu = δG/δu = 0 at the walls",
        transpose(fρ) * S * gu0 - transpose(gρ) * S * fu0 ==
        transpose(fu0) * S * gρ - transpose(gu0) * S * fρ)
end

# SBP finite differences on the interval, wall closure, exact
for N in (7, 8)
    _, W, Qm, B = sbp2(Q, N, 1)
    Winv = inv(W)
    Ka = Winv * (Qm - B) * Winv
    Kb = Winv * Qm * Winv
    ρq = positive_vec(rng, N)
    mq = ρq .* rnd_vec(rng, N)
    P, dP = flat_bracket(Ka, Kb, ρq, mq)
    w = diag(W)
    check("SBP2, N = $N, wall closure: antisymmetric and Jacobiator vanishes exactly",
        Kb == -transpose(Ka) && iszero(raw_residual(P, dP)))
    check("SBP2, N = $N, wall closure: mass Σ w_i ρ_i is an exact Casimir",
        all(iszero, P * [w; zeros(Q, N)]))
    saw = [Q((-1)^i) for i in 1:N]
    check("SBP2, N = $N, wall closure: ker K_ρm = sawtooth W(-1)^i for every N",
        size(kernel(Ka), 2) == 1 && all(iszero, Ka * (W * saw)))
end

# Which boundary condition the wall closure encodes.  Ka (ρ ∘ M b) = -M⁻¹Sᵀ(ρb) + O(h²) is the
# Galerkin form of ∂(ρb) with the boundary term [φ_k ρ b] dropped, i.e. the zero-flux condition
# (ρ b)(0) = (ρ b)(L) = 0 -- the momentum m = ρ δH/δm vanishes at the wall -- imposed weakly.
# Expected: far from the walls second order whatever the data; within a few nodes of a wall the
# Galerkin derivative on an interval is first order when the data satisfy the wall condition, and
# the dropped boundary term is an O(1/h) residual (decaying into the interior like the entries of
# M⁻¹) when they do not.
let
    b0x(x) = sin(2π * x)                     # vanishes at 0 and 1 (wall-compatible)
    db0x(x) = 2π * cos(2π * x)
    L = 1.0
    ρI(x) = 2.0 + sin(2π * x) * 0.5
    dρI(x) = π * cos(2π * x)
    uI(x) = 0.7 + 0.3 * cos(2π * x)
    duI(x) = -0.6π * sin(2π * x)
    aI(x) = cos(2π * x) + 0.4
    daI(x) = -2π * sin(2π * x)
    for (bl, bf, dbf, compatible) in (("b(0) = b(L) = 0", b0x, db0x, true),
        ("b(0) = b(L) ≠ 0", x -> 0.5 + cos(2π * x), x -> -2π * sin(2π * x), false))
        efar, enear, hs = Float64[], Float64[], Float64[]
        for ne in (16, 32, 64, 128, 256)
            x, M, S, B = lagrange_matrices(Float64, 1, ne; periodic = false, L = L)
            N = length(x)
            Minv = inv(M)
            Ka = Minv * (S - B) * Minv
            Kb = Minv * S * Minv
            ρ0, u0 = ρI.(x), uI.(x)
            P = flat_matrix(Ka, Kb, ρ0, ρ0 .* u0)
            v = P * [M * aI.(x); M * bf.(x)]
            m0, dm0 = ρ0 .* u0, dρI.(x) .* u0 .+ ρ0 .* duI.(x)
            eρ = dρI.(x) .* bf.(x) .+ ρ0 .* dbf.(x)
            em = ρ0 .* daI.(x) .+ 2 .* m0 .* dbf.(x) .+ dm0 .* bf.(x)
            err = abs.([v[1:N] - eρ; v[(N + 1):2N] - em]) / maximum(abs, [eρ; em])
            far = [i for i in 1:N if L / 4 ≤ x[i] ≤ 3L / 4]        # fixed physical distance
            near = [1, 2, N - 1, N]                                # within one element of a wall
            push!(efar, maximum(err[[far; far .+ N]]))
            push!(enear, maximum(err[[near; near .+ N]]))
            push!(hs, L / ne)
        end
        rf, rn = rates(efar, hs), rates(enear, hs)
        println("   wall closure, $bl: errors on [L/4, 3L/4] ",
            join([@sprintf("%.2e", e) for e in efar], " "),
            "   rates ", join([@sprintf("%.2f", r) for r in rf], " "))
        println("   wall closure, $bl: errors at the wall nodes ",
            join([@sprintf("%.2e", e) for e in enear], " "),
            "   rates ", join([@sprintf("%.2f", r) for r in rn], " "))
        check("wall closure, $bl: rows on [L/4, 3L/4] converge at order 2", rf[end] > 1.8,
            @sprintf("rate = %.2f", rf[end]))
        if compatible
            check(
                "wall closure, $bl: wall rows converge at first order (interval Galerkin derivative)",
                0.8 < rn[end] < 1.5, @sprintf("rate = %.2f", rn[end]))
        else
            check(
                "wall closure, $bl: wall rows carry the dropped boundary term, O(1/h) (weak m = 0 violated)",
                -1.4 < rn[end] < -0.7, @sprintf("rate = %.2f", rn[end]))
        end
    end
end

summary("II1_semidirect_flat.jl")
