#!/usr/bin/env julia
#
# I.1 -- Vlasov--Poisson on a bounded x-domain as an exact Lie-Poisson system on u(N)*.
#
#     julia --project=scripts scripts/fable/I1_vlasov_uN.jl
#
# Quantise over the CONFIGURATION domain only: V_N ⊂ L²(Ω_x) spanned by N Fourier modes
# (periodic, Ω_x = S¹ of length 2π) or N sine modes (Dirichlet, Ω_x = [0,π]).  The Lie algebra
# is u(N) with bracket (X,Y) ↦ [X,Y]/(iℏ); its dual carries the density matrix ρ and the
# Lie-Poisson bracket  {F,G}(ρ) = (1/iℏ) tr(ρ [F_ρ, G_ρ]).  A Vlasov observable a(x,p) is
# carried to u(N) by the compressed Weyl quantisation T_N(a) = P_N Op(a) P_N, which for
# a = Σ_m p^m χ_m(x) is the midpoint rule in the momentum representation,
#
#     [T_N a]_{kl} = Σ_m ((p_k + p_l)/2)^m  χ̂_m(k − l),        p_k = ℏ k .
#
# The homomorphism defect  D(a,b) = (1/iℏ)[T_N a, T_N b] − T_N{a,b}  is the whole
# consistency question: {F_a,F_b}_{u(N)}(ρ) − "∫ f {a,b}" = tr(ρ D(a,b)).
#
#   1. Jacobi: exact zero of the Jacobiator of (1/iℏ) tr(ρ[·,·]) on gl(N,Q), N = 3,4,5, and of
#      the structure-constant residual; control = perturbed bracket [X,Y] + ε(XYᵀ − YXᵀ), which
#      is antisymmetric and bilinear but not Lie (residual O(1));
#   2. Casimirs: {tr ρ^k, G} = 0 exactly for all k and every G; control = tr(ρ D ρ), not a
#      spectral function, and tr ρ² under the perturbed bracket;
#   3. dictionary on the circle (exact arithmetic): the defect D(a,b) vanishes EXACTLY on the
#      inner block |k|,|l| ≤ K − B for symbols of degree ≤ 1 in p of x-band B, and for
#      (a, h = p²/2 + φ) with deg_p a ≤ 2; for deg_p a = 3 it equals +(ℏ²/4) T(χ φ''') — the
#      Wigner-equation correction (ℏ²/24) φ''' ∂_p³ with its exact coefficient; the edge block is
#      O(1); control = standard (non-Weyl) ordering, whose defect is O(ℏ) and nonzero inside;
#   4. Dirichlet box [0,π]: T^Dir(a) equals the odd block A_{mn} − A_{m,−n} of the doubled-circle
#      quantisation for reflection-even symbols a(−x,−p) = a(x,p) (checked against direct
#      sine-basis matrix elements by quadrature); the inner-block exactness is inherited; the
#      canonical pair (x, p) is NOT reflection-compatible and its box commutator misses the CCR
#      on the inner block m,n ≤ N/2, by an amount that does not decay as N grows (an edge effect
#      would); the kink of the even extension of a grounded-wall potential x(π−x) gives a
#      mode-tail leak at every N, a zero-field-wall potential none;
#   5. dynamics (floating point, refinement in N at fixed v_max/σ): Weyl symbol of the
#      von Neumann–Poisson field (1/iℏ)[T h_ρ, ρ] against the Vlasov field φ'∂_p f − p ∂_x f for
#      a Gaussian-in-p, cosine-in-x state; the difference scales as ℏ² ∝ K⁻² and matches
#      (ℏ²/24) φ''' ∂_p³ f; tr ρ^k against (2πℏ)^{k−1} ∫∫ f^k (exact for k = 1, 2, O(ℏ²) for k = 3).
#
# Pairing convention (as in III.2): dF = tr(F_ρ dρ), so for F = tr(Xρ), F_ρ = X.  Coordinates
# ρ_{ab} ↦ l = a + (b−1)N; the coordinate function ρ_l has gradient E_lᵀ.  Then
# ρ̇ = {ρ, H} = (1/iℏ)[H_ρ, ρ], the von Neumann equation with Ĥ = H_ρ, and for the classical
# side ḟ = {H_f, f}_{xp} = φ' ∂_p f − p ∂_x f, the Vlasov equation with force −φ'.

using LinearAlgebra
using Random
using Printf
using GeometricBrackets: jacobi_residual, structure_constant_residual

include(joinpath(@__DIR__, "..", "check.jl"))
using .Checks: header, check, check_exact, summary, fmt

const Q = Rational{BigInt}
const CQ = Complex{Q}
const rng = MersenneTwister(2026_09_03)

randq(n, m = n) = [Q(rand(rng, -3:3), rand(rng, 1:3)) for _ in 1:n, _ in 1:m]

# ---------------------------------------------------------------------------------------------
# 1. the Lie-Poisson bracket on u(N)*, from an arbitrary bilinear product
# ---------------------------------------------------------------------------------------------

"Elementary matrix E_l, l = a + (b−1)N ↔ (a,b)."
function elementary(::Type{T}, N, l) where {T}
    E = zeros(T, N, N)
    E[l] = one(T)
    return E
end

"Structure matrix P_ij(ρ) = tr(ρ · prod(E_iᵀ, E_jᵀ)) of the bracket {F,G} = tr(ρ prod(F_ρ,G_ρ))."
function structure_matrix(prod, ρ::AbstractMatrix{T}) where {T}
    N = size(ρ, 1)
    P = zeros(T, N^2, N^2)
    for j in 1:(N ^ 2), i in 1:(N ^ 2)

        P[i, j] = tr(ρ *
                     prod(transpose(elementary(T, N, i)), transpose(elementary(T, N, j))))
    end
    return P
end

"dP[l,i,j] = ∂P_ij/∂ρ_l = tr(E_l prod(E_iᵀ,E_jᵀ)): the structure constants c_ij^l."
function structure_tensor(prod, ::Type{T}, N) where {T}
    dP = zeros(T, N^2, N^2, N^2)
    for l in 1:(N ^ 2)
        dP[l, :, :] = structure_matrix(prod, elementary(T, N, l))
    end
    return dP
end

commutator(X, Y) = X * Y - Y * X
perturbed(ε) = (X, Y) -> X * Y - Y * X + ε * (X * transpose(Y) - Y * transpose(X))

header("1. the u(N) Lie-Poisson bracket (1/iℏ) tr(ρ[F_ρ,G_ρ]) satisfies Jacobi exactly")
println("  (the scalar 1/iℏ does not enter the Jacobiator; the real form gl(N,Q) is tested,")
println("   its complexification gl(N,C) ⊃ u(N) — a polynomial identity over Q holds on u(N))")
for N in (3, 4, 5)
    ρ = randq(N)
    P = structure_matrix(commutator, ρ)
    dP = structure_tensor(commutator, Q, N)
    check("N = $N: P(ρ) is antisymmetric", iszero(P + transpose(P)))
    check("N = $N: P(ρ) is linear in ρ, P = Σ_l ρ_l dP[l]",
        P == sum(ρ[l] * dP[l, :, :] for l in 1:(N ^ 2)))
    res = jacobi_residual(P, dP)
    check_exact("N = $N: Jacobiator of the commutator bracket", res; atol = 0)
    # dP[m,i,j] = c_ij^m is already the index order structure_constant_residual wants
    check_exact("N = $N: structure-constant condition of u(N)", structure_constant_residual(dP); atol = 0)
end
println("  control: [X,Y]_ε = XY − YX + ε(XYᵀ − YXᵀ), antisymmetric, bilinear, NOT a Lie bracket")
for (N, ε) in ((3, Q(1)), (3, Q(1, 10)), (4, Q(1)))
    ρ = randq(N)
    P = structure_matrix(perturbed(ε), ρ)
    dP = structure_tensor(perturbed(ε), Q, N)
    check("N = $N, ε = $(fmt(ε)): control bracket is antisymmetric", iszero(P +
                                                                            transpose(P)))
    res = jacobi_residual(P, dP)
    check("N = $N, ε = $(fmt(ε)): control Jacobiator is NONZERO", res > 0,
        @sprintf("%.3e", Float64(res)))
    check("N = $N, ε = $(fmt(ε)): control structure-constant residual is NONZERO",
        structure_constant_residual(dP) > 0, @sprintf("%.3e",
            Float64(structure_constant_residual(dP))))
end

# ---------------------------------------------------------------------------------------------
# 2. Casimirs tr ρ^k
# ---------------------------------------------------------------------------------------------

header("2. tr ρ^k are Casimirs: {tr ρ^k, G} = (1/iℏ) tr(ρ [k ρ^{k−1}, G_ρ]) = 0 exactly")
"Bracket of two functions given by their gradient matrices (dF = tr(F_ρ dρ)), without the 1/iℏ."
lp_bracket(ρ, Fρ, Gρ) = tr(ρ * commutator(Fρ, Gρ))
for N in (3, 4, 6)
    ρ = randq(N)
    Gρ = randq(N)
    # From k = 2.  At k = 1 the gradient is the identity, which commutes with everything, so the
    # bracket vanishes for any ρ, any G and any commutator-shaped bracket at all -- no test.
    for k in 2:N
        Ck = k * ρ^(k - 1)          # gradient of tr ρ^k
        check_exact("N = $N: {tr ρ^$k, G} for random G", lp_bracket(ρ, Ck, Gρ); atol = 0)
    end
    # the same statement on the coordinate side: P(ρ) ∇C = 0
    P = structure_matrix(commutator, ρ)
    ∇C2 = vec(transpose(2ρ))
    check_exact("N = $N: P(ρ) ∇(tr ρ²) = 0", maximum(abs, P * ∇C2); atol = 0)
    # controls
    D = Diagonal(Q.(1:N))
    Cρ = D * ρ + ρ * D              # gradient of tr(ρ D ρ), not a spectral function
    check("N = $N: control tr(ρDρ) is NOT a Casimir", !iszero(lp_bracket(ρ, Cρ, Gρ)),
        fmt(lp_bracket(ρ, Cρ, Gρ)))
    Pε = structure_matrix(perturbed(Q(1)), ρ)
    check("N = $N: control tr ρ² is NOT a Casimir of the perturbed bracket",
        !iszero(Pε * ∇C2), @sprintf("%.3e", Float64(maximum(abs, Pε * ∇C2))))
end

# ---------------------------------------------------------------------------------------------
# 3. the dictionary on the circle: symbols, Weyl quantisation, the defect D(a,b)
# ---------------------------------------------------------------------------------------------
#
# A symbol is a Dict (m, j) => c  meaning  Σ c p^m e^{ijx}  on the circle of length 2π; then
# ∂_x (p^m e_j) = ij p^m e_j,  ∂_p (p^m e_j) = m p^{m−1} e_j,  and
# {p^m e_j, p^n e_l} = i (j n − m l) p^{m+n−1} e_{j+l}.

const Symbol_ = Dict{Tuple{Int, Int}, CQ}

function sym_add!(a::Symbol_, key, c)
    v = get(a, key, zero(CQ)) + c
    iszero(v) ? delete!(a, key) : (a[key] = v)
    return a
end
sym(pairs...) = (a = Symbol_(); for (k, c) in pairs
        sym_add!(a, k, CQ(c))
    end; a)
sym_scale(a, s) = (b = Symbol_(); for (k, c) in a
        sym_add!(b, k, s * c)
    end; b)
sym_sum(a, b) = (c = copy(a); for (k, v) in b
        sym_add!(c, k, v)
    end; c)
function sym_mul(a, b)
    c = Symbol_()
    for ((m, j), u) in a, ((n, l), v) in b

        sym_add!(c, (m + n, j + l), u * v)
    end
    return c
end
function sym_bracket(a, b)
    c = Symbol_()
    for ((m, j), u) in a, ((n, l), v) in b

        coef = j * n - m * l
        iszero(coef) && continue
        sym_add!(c, (m + n - 1, j + l), im * coef * u * v)
    end
    return c
end
"∂_x^r of a symbol."
function sym_dx(a, r = 1)
    b = Symbol_()
    for ((m, j), u) in a
        sym_add!(b, (m, j), (im * j)^r * u)
    end
    return b
end
"Real trigonometric polynomials as symbols: cos(jx) p^m and sin(jx) p^m."
cosx(j, m = 0) = j == 0 ? sym(((m, 0), 1)) : sym(((m, j), Q(1, 2)), ((m, -j), Q(1, 2)))
sinx(j, m = 0) = sym(((m, j), -im * Q(1, 2)), ((m, -j), im * Q(1, 2)))
pmono(m) = sym(((m, 0), 1))
xband(a) = maximum(abs(j) for (m, j) in keys(a); init = 0)
pdeg(a) = maximum(m for (m, j) in keys(a); init = 0)

"Compressed Weyl quantisation on modes k = −K..K (index k + K + 1), p_k = ℏ k: midpoint rule."
function weyl(a::Symbol_, K, ℏ)
    N = 2K + 1
    A = zeros(CQ, N, N)
    for l in (-K):K, k in (-K):K

        pm = ℏ * (k + l) // 2
        v = zero(CQ)
        for ((m, j), u) in a
            j == k - l || continue
            v += u * pm^m
        end
        A[k + K + 1, l + K + 1] = v
    end
    return A
end
"Standard-ordered control: p^m χ(x) ↦ p̂^m χ, i.e. [·]_{kl} = p_k^m χ̂(k−l)."
function standard_ordered(a::Symbol_, K, ℏ)
    N = 2K + 1
    A = zeros(CQ, N, N)
    for l in (-K):K, k in (-K):K

        v = zero(CQ)
        for ((m, j), u) in a
            j == k - l || continue
            v += u * (ℏ * k)^m
        end
        A[k + K + 1, l + K + 1] = v
    end
    return A
end

"The homomorphism defect (1/iℏ)[T a, T b] − T{a,b} for a quantisation map T."
function defect(T, a, b, K, ℏ)
    commutator(T(a, K, ℏ), T(b, K, ℏ)) / (im * ℏ) - T(sym_bracket(a, b), K, ℏ)
end

"Largest |entry| on the inner block |k|,|l| ≤ K − B and on its complement."
function inner_edge(D, K, B)
    isinner = [abs(k) <= K - B for k in (-K):K]
    mi = maximum((isinner[k] && isinner[l]) ? abs(D[k, l]) : zero(real(eltype(D)))
    for k in axes(D, 1), l in axes(D, 2))
    me = maximum((isinner[k] && isinner[l]) ? zero(real(eltype(D))) : abs(D[k, l])
    for k in axes(D, 1), l in axes(D, 2))
    return mi, me
end
f64(x) = Float64(Rational{BigInt}(x))

header("3. dictionary on the circle: the defect D(a,b) = (1/iℏ)[T a, T b] − T{a,b}")
let ℏ = Q(2, 3), K = 6
    # Hermiticity of the quantisation of real symbols
    for (name, a) in (("cos 2x", cosx(2)), ("p sin x", sinx(1, 1)), (
        "p² cos 3x", cosx(3, 2)))
        A = weyl(a, K, ℏ)
        check("T($name) is Hermitian", A == A')
    end
    # 3a. first-order symbols: exact on the inner block
    a = sym_sum(cosx(2), sinx(1, 1))                    # φ_a = cos 2x, ψ_a = sin x
    b = sym_sum(sinx(1), sym_sum(cosx(2, 1), pmono(1)))  # φ_b = sin x, ψ_b = 1 + cos 2x
    B = max(xband(a), xband(b))
    D = defect(weyl, a, b, K, ℏ)
    scale = maximum(abs, commutator(weyl(a, K, ℏ), weyl(b, K, ℏ)) / (im * ℏ))
    mi, me = inner_edge(D, K, B)
    check_exact("deg_p ≤ 1, band B = $B: D vanishes on the inner block |k|,|l| ≤ K − B", mi; atol = 0)
    check(
        "deg_p ≤ 1: D is O(1) on the edge (so the inner statement is not vacuous)", me > 0,
        @sprintf("max edge |D| / max |[Ta,Tb]/iℏ| = %.3f", f64(me) / f64(scale)))
    # the uncompressed identity, seen through a larger K: the inner block of D at K' ⊃ inner at K
    D2 = defect(weyl, a, b, K + B, ℏ)
    mi2, _ = inner_edge(D2, K + B, B)
    check_exact("deg_p ≤ 1: the same with K → K + B (inner block grows with K)", mi2; atol = 0)
    # 3b. control: standard ordering is a homomorphism for deg ≤ 1 as well (vector-field rep.) ...
    Ds = defect(standard_ordered, a, b, K, ℏ)
    mis, _ = inner_edge(Ds, K, B)
    check_exact(
        "deg_p ≤ 1: standard ordering ALSO exact (it is the Lie-derivative representation)",
        mis; atol = 0)
    # ... but not for degree 2, where Weyl is:
    a2 = cosx(1, 2)                                      # p² cos x
    φ = cosx(2)                                          # potential cos 2x
    Dw = defect(weyl, a2, φ, K, ℏ)
    Ds2 = defect(standard_ordered, a2, φ, K, ℏ)
    B2 = xband(a2) + xband(φ)
    check_exact("D(p² cos x, cos 2x) Weyl: zero on the inner block", inner_edge(Dw, K, B2)[1]; atol = 0)
    mis2, _ = inner_edge(Ds2, K, B2)
    check("D(p² cos x, cos 2x) standard ordering: NONZERO inside (control)", mis2 > 0,
        @sprintf("max inner |D| = %.3f, = O(ℏ): predicted −iℏ χφ'' term", f64(mis2)))
    # the standard-ordering defect is exactly −iℏ T_std(χ φ''):  (1/iℏ)[p̂²χ, φ] = −2p̂(χφ') − iℏχφ''
    pred = standard_ordered(sym_scale(sym_mul(cosx(1), sym_dx(φ, 2)), -im * ℏ), K, ℏ)
    check_exact(
        "   standard-ordering defect = −iℏ·T_std(χ φ'') on the inner block, exactly",
        inner_edge(Ds2 - pred, K, B2)[1]; atol = 0)
    # 3c. the Vlasov--Poisson Hamiltonian h = p²/2 + φ against observables of degree 0, 1, 2, 3
    h = sym_sum(sym_scale(pmono(2), Q(1, 2)), sym_sum(cosx(1), sym_scale(sinx(2), Q(1, 3))))
    for (m, name) in ((0, "cos 2x"), (1, "p cos 2x"), (2, "p² cos 2x"))
        a_m = cosx(2, m)
        Dm = defect(weyl, a_m, h, K, ℏ)
        Bm = xband(a_m) + xband(h)
        check_exact(
            "D($name, p²/2 + φ) = 0 on the inner block (Moyal terms vanish for deg_p ≤ 2)",
            inner_edge(Dm, K, Bm)[1]; atol = 0)
    end
    a3 = cosx(2, 3)                                      # p³ cos 2x
    D3 = defect(weyl, a3, h, K, ℏ)
    B3 = xband(a3) + xband(h)
    φ3 = sym_dx(sym_sum(cosx(1), sym_scale(sinx(2), Q(1, 3))), 3)   # φ'''
    # Moyal: {a,h}_M = {a,h} − (ℏ²/24) aΛ³h, and for h = p²/2 + φ only the term −∂_p³a φ''' of Λ³
    # survives, so {a,h}_M − {a,h} = +(ℏ²/24) ∂_p³a φ''' = (ℏ²/4) χ φ''' for a = p³χ.
    pred3 = weyl(sym_scale(sym_mul(cosx(2), φ3), ℏ^2 // 4), K, ℏ)  # +(ℏ²/4) T(χ φ''')
    check("D(p³ cos 2x, h) is NONZERO on the inner block (the ℏ² Wigner term)",
        inner_edge(D3, K, B3)[1] > 0, @sprintf("max inner |D| = %.4f",
            f64(inner_edge(D3, K, B3)[1])))
    check_exact(
        "D(p³ χ, h) = +(ℏ²/4) T(χ φ''') exactly on the inner block  [= (ℏ²/24) φ''' ∂_p³ (p³χ)]",
        inner_edge(D3 - pred3, K, B3)[1]; atol = 0)
    # general degree-2 pair: Moyal ℏ² term  −(ℏ²/2) p (χ'η'' − χ''η')
    χ, η = cosx(1), sinx(2)
    a22, b22 = cosx(1, 2), sinx(2, 2)
    D22 = defect(weyl, a22, b22, K, ℏ)
    corr = sym_scale(
        sym_mul(pmono(1),
            sym_sum(sym_mul(sym_dx(χ), sym_dx(η, 2)), sym_scale(sym_mul(sym_dx(χ, 2), sym_dx(η)), -1))),
        -ℏ^2 // 2)
    check_exact(
        "D(p² cos x, p² sin 2x) = −(ℏ²/2) T(p(χ'η'' − χ''η')) exactly on the inner block",
        inner_edge(D22 - weyl(corr, K, ℏ), K, xband(a22) + xband(b22))[1]; atol = 0)
    # The same identity at ℏ/2, so that the ℏ² prefactor is pinned rather than absorbed: a defect
    # of the wrong order in ℏ would match +(ℏ²/4) T(χφ''') at one ℏ and not at both.
    D3h = defect(weyl, a3, h, K, ℏ // 2)
    pred3h = weyl(sym_scale(sym_mul(cosx(2), φ3), (ℏ // 2)^2 // 4), K, ℏ // 2)
    check_exact("   the same at ℏ/2", inner_edge(D3h - pred3h, K, B3)[1]; atol = 0)
end

# ---------------------------------------------------------------------------------------------
# 4. the Dirichlet box [0,π] as the reflection-odd sector of the circle [−π,π]
# ---------------------------------------------------------------------------------------------
#
# e_n(x) = √(2/π) sin(nx), n = 1..N, on [0,π] is the restriction of √2 · o_n, o_n = (E_n − E_{−n})/(i√2)
# odd on the circle.  For a symbol with a(−x,−p) = a(x,p), Op(a) commutes with the reflection R and
# ⟨e_m|Op(a)|e_n⟩_{[0,π]} = A_{m,n} − A_{m,−n}  with  A = weyl(a, K ≥ N, ℏ)  on the doubled circle.

function dirichlet_block(A, K, N)
    idx(k) = k + K + 1
    return [A[idx(m), idx(n)] - A[idx(m), idx(-n)] for m in 1:N, n in 1:N]
end
weyl_dir(a, N, ℏ) = dirichlet_block(weyl(a, N, ℏ), N, N)
"Defect on the Dirichlet block."
function defect_dir(a, b, N, ℏ)
    commutator(weyl_dir(a, N, ℏ), weyl_dir(b, N, ℏ)) / (im * ℏ) -
    weyl_dir(sym_bracket(a, b), N, ℏ)
end
function inner_edge_dir(D, N, B)
    isinner = [n <= N - B for n in 1:N]
    mi = maximum((isinner[k] && isinner[l]) ? abs(D[k, l]) : zero(real(eltype(D)))
    for k in 1:N, l in 1:N)
    me = maximum((isinner[k] && isinner[l]) ? zero(real(eltype(D))) : abs(D[k, l])
    for k in 1:N, l in 1:N)
    return mi, me
end

# Gauss--Legendre on [0,π] (Golub--Welsch), for the direct sine-basis matrix elements.
function gauss_legendre(n, a, b)
    β = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    J = SymTridiagonal(zeros(n), β)
    ev = eigen(J)
    x = ev.values
    w = 2 .* ev.vectors[1, :] .^ 2
    return (b - a) / 2 .* x .+ (a + b) / 2, (b - a) / 2 .* w
end
"Values of the real trigonometric symbol coefficient function χ_m(x) = Σ_j c_{mj} e^{ijx} and derivatives."
function coeff_fun(a::Symbol_, m, r, x)
    v = zero(ComplexF64)
    for ((mm, j), u) in a
        mm == m || continue
        v += ComplexF64(u) * (im * j)^r * exp(im * j * x)
    end
    return v
end
"Direct Dirichlet matrix of Weyl(p^m χ_m(x)), m ≤ 2, by quadrature on [0,π]."
function weyl_dir_direct(a::Symbol_, N, ℏ; nq = 256)
    x, w = gauss_legendre(nq, 0.0, Float64(π))
    e(n) = sqrt(2 / π) .* sin.(n .* x)
    de(n) = sqrt(2 / π) .* n .* cos.(n .* x)
    d2e(n) = -n^2 .* e(n)
    A = zeros(ComplexF64, N, N)
    ħ = Float64(ℏ)
    for n in 1:N
        u, du, d2u = e(n), de(n), d2e(n)
        Au = zeros(ComplexF64, nq)
        for m in 0:pdeg(a)
            χ = coeff_fun.(Ref(a), m, 0, x)
            χ1 = coeff_fun.(Ref(a), m, 1, x)
            χ2 = coeff_fun.(Ref(a), m, 2, x)
            if m == 0
                Au .+= χ .* u
            elseif m == 1                 # (p̂χ + χp̂)/2 = −iħ(χ ∂ + χ'/2)
                Au .+= -im * ħ .* (χ .* du .+ χ1 .* u ./ 2)
            elseif m == 2                 # −ħ²(χ ∂² + χ' ∂ + χ''/4)
                Au .+= -ħ^2 .* (χ .* d2u .+ χ1 .* du .+ χ2 .* u ./ 4)
            else
                error("degree > 2 not implemented in the direct route")
            end
        end
        for mm in 1:N
            A[mm, n] = sum(w .* e(mm) .* Au)
        end
    end
    return A
end

header("4. the Dirichlet box [0,π]: odd sector of the doubled circle")
let ℏ = Q(2, 3), N = 8
    # 4a. block formula against direct sine-basis matrix elements for reflection-even symbols
    for (name, a) in (("cos 3x", cosx(3)), ("p sin 2x", sinx(2, 1)), (
        "p² cos x", cosx(1, 2)),
        ("p² (1 + cos 2x)", sym_sum(pmono(2), cosx(2, 2))))
        Ablock = ComplexF64.(weyl_dir(a, N, ℏ))
        Adir = weyl_dir_direct(a, N, ℏ)
        check_exact("T^Dir($name): odd-block formula = direct sine-basis quadrature",
            maximum(abs, Ablock - Adir); atol = 1e-10)
    end
    # a reflection-ODD symbol: the block formula does not apply (Op does not preserve the odd sector)
    aodd = pmono(1)   # p itself: a(−x,−p) = −a
    Ablock = ComplexF64.(weyl_dir(aodd, N, ℏ))
    Adir = weyl_dir_direct(aodd, N, ℏ)
    check("T^Dir(p): block formula ≠ direct (p is reflection-odd) — control",
        maximum(abs, Ablock - Adir) > 1e-3,
        @sprintf("max diff %.3f; block formula gives %s", maximum(abs, Ablock - Adir),
            iszero(Ablock) ? "zero" : "nonzero"))
    # 4b. inherited inner exactness for reflection-even band-limited symbols
    a = sym_sum(cosx(2), sinx(1, 1))                 # φ_a even, ψ_a odd  → a(−x,−p) = a
    b = sym_sum(cosx(1), sym_sum(sinx(2, 1), pmono(2)))
    B = xband(a) + xband(b)
    D = defect_dir(a, b, N, ℏ)
    mi, me = inner_edge_dir(D, N, B)
    check_exact("Dirichlet, reflection-even deg ≤ 2 vs deg ≤ 2: D = 0 for m,n ≤ N − B", mi; atol = 0)
    check("Dirichlet: D ≠ 0 at the edge", me > 0, @sprintf("max edge |D| = %.3f", f64(me)))
    hD = sym_sum(sym_scale(pmono(2), Q(1, 2)), cosx(1))
    for (m, name) in ((0, "cos 2x"), (1, "p sin 2x"), (2, "p² cos 2x"))
        am = m == 1 ? sinx(2, 1) : cosx(2, m)
        Dm = defect_dir(am, hD, N, ℏ)
        check_exact("Dirichlet: D($name, p²/2 + cos x) = 0 for m,n ≤ N − B",
            inner_edge_dir(Dm, N, xband(am) + 1)[1]; atol = 0)
    end
end
# The box observables x and −iℏ∂ are the odd blocks of Weyl(|x|) and Weyl(p sign x), whose Fourier
# coefficients are irrational (4/π): this part is in floating point with the closed forms.
let ℏ = 2 / 3
    "Weyl matrix on the doubled circle in Float64 for a symbol given as (m, j) => ComplexF64."
    function weylf(a::Dict{Tuple{Int, Int}, ComplexF64}, K, ℏ)
        local N = 2K + 1      # `local`: an inner function assigning to `N` would rebind the let-block's N
        A = zeros(ComplexF64, N, N)
        for l in (-K):K, k in (-K):K

            pm = ℏ * (k + l) / 2
            v = zero(ComplexF64)
            for ((m, j), u) in a
                j == k - l || continue
                v += u * pm^m
            end
            A[k + K + 1, l + K + 1] = v
        end
        return A
    end
    dirblock(A, K, N) = [A[m + K + 1, n + K + 1] - A[m + K + 1, -n + K + 1]
                         for m in 1:N, n in 1:N]
    "Closed-form sine-basis matrices on [0,π]: position x and momentum −iℏ∂."
    function xbox(N)
        X = zeros(N, N)
        for n in 1:N, m in 1:N

            if m == n
                X[m, n] = π / 2
            elseif isodd(m + n)
                X[m, n] = -8m * n / (π * (m^2 - n^2)^2)
            end
        end
        return X
    end
    function pbox(N, ℏ)
        P = zeros(ComplexF64, N, N)
        for n in 1:N, m in 1:N

            if m != n && isodd(m + n)
                P[m, n] = -im * ℏ * 4m * n / (π * (m^2 - n^2))
            end
        end
        return P
    end
    # check the closed forms by quadrature
    x, w = gauss_legendre(400, 0.0, Float64(π))
    N = 8
    e(n) = sqrt(2 / π) .* sin.(n .* x)
    Xq = [sum(w .* e(m) .* x .* e(n)) for m in 1:N, n in 1:N]
    Pq = [sum(w .* e(m) .* (-im * ℏ * sqrt(2 / π) * n .* cos.(n .* x)))
          for m in 1:N, n in 1:N]
    check_exact("closed-form ⟨e_m|x|e_n⟩ on [0,π] = quadrature", maximum(abs, Xq - xbox(N)); atol = 1e-10)
    check_exact(
        "closed-form ⟨e_m|−iℏ∂|e_n⟩ on [0,π] = quadrature", maximum(abs, Pq -
                                                                         pbox(N, ℏ)); atol = 1e-10)
    # x on the box is the odd block of Weyl(|x|) with |x| = π/2 − (4/π) Σ_{j odd} cos(jx)/j²  (truncated band J)
    J = 4001
    absx = Dict{Tuple{Int, Int}, ComplexF64}((0, 0) => π / 2)
    for j in 1:2:J
        absx[(0, j)] = -2 / (π * j^2)
        absx[(0, -j)] = -2 / (π * j^2)
    end
    Xblk = dirblock(weylf(absx, N, ℏ), N, N)
    check_exact(
        "x on the box = odd block of Weyl(|x|)",
        maximum(abs, Xblk - xbox(N)); atol = 1e-12)
    # p·sign(x) is reflection-even; its odd block is the box momentum matrix
    signp = Dict{Tuple{Int, Int}, ComplexF64}()
    for j in 1:2:J          # sign(x) = (4/π) Σ_{j odd} sin(jx)/j
        signp[(1, j)] = -im * 2 / (π * j)
        signp[(1, -j)] = im * 2 / (π * j)
    end
    Pblk = dirblock(weylf(signp, N, ℏ), N, N)
    # Both identities are exact -- the Fourier tail beyond |k − l| = 2N never enters the N×N block,
    # so the truncation at J does not loosen them.  Measured residuals are at machine epsilon;
    # atol 1e-6 and 1e-3 admitted a 0.1 % error in every entry and were 10 to 13 orders too loose.
    check_exact("−iℏ∂ on the box = odd block of Weyl(p·sign x)",
        maximum(abs, Pblk - pbox(N, ℏ)); atol = 1e-12)
    # 4c. the CCR in a box: (1/iℏ)[X,P] − I.  {|x|, p sign x} = 1, but neither symbol is band-limited,
    # so the defect is not confined to the edge.
    #
    # tr((1/iℏ)[X,P] − I) = −N is NOT checked here: the trace of a commutator vanishes for any pair
    # of square matrices, so that identity holds whatever X and P are and distinguishes nothing.
    # What distinguishes is the inner block m,n ≤ N/2, which recedes from the edge as N grows.  The
    # defect there is O(1/N) -- an algebraic tail, which is what a symbol that is not band-limited
    # leaves behind; a defect genuinely confined to the boundary would fall off exponentially in
    # the distance to it and be invisible at m,n ≤ N/2.  The rate is what is asserted below.
    println("  4c. CCR in the box: (1/iℏ)[X_N, P_N] − I, inner block m,n ≤ N/2")
    prev = NaN
    inners = Float64[]
    for N in (8, 16, 32)
        C = commutator(xbox(N), pbox(N, ℏ)) / (im * ℏ) - I
        h = N ÷ 2
        inner = maximum(abs, C[1:h, 1:h])
        diag_inner = maximum(abs, [C[n, n] for n in 1:h])
        push!(inners, inner)
        check("N = $N: the CCR defect is O(1) on the inner block, not at roundoff",
            inner > 1e-8,
            @sprintf("max |C| inner = %.3e", inner))
        @printf("    N = %2d: max |C| inner = %.3e   max |C_nn| inner = %.3e   ratio to previous %.2f\n",
            N, inner, diag_inner, isnan(prev) ? NaN : prev / inner)
        prev = inner
    end
    check(
        "the inner-block defect decays only as 1/N: ratio over N = 8 -> 32 is 4, not exponential",
        3.2 < inners[1] / inners[end] < 4.8,
        @sprintf("%.3e -> %.3e, ratio %.2f (1/N predicts 4)", inners[1], inners[end],
            inners[1] / inners[end]))
    # 4d. mode-tail leak of a potential: zero-field walls (smooth even extension) vs grounded walls (kink)
    println("  4d. leak Q_N φ P_N ρ of the potential term for a state on modes n ≤ N/2 (σ_p = 1, v_max = 6)")
    for (name, φfun, noleak) in ((
        "φ = cos 2x   (even extension smooth; zero-field-compatible)", x -> cos(2x), true),
        ("φ = x(π − x) (grounded walls: even extension has a kink)",
        x -> x * (π - x), false),
        ("φ = x(π − x) + cos x  (same kink, different smooth part)",
        x -> x * (π - x) + cos(x), false))
        prevleak = NaN
        leaks = Float64[]
        for N in (8, 16, 32)
            N2 = 2N
            xq, wq = gauss_legendre(600, 0.0, Float64(π))
            es(n) = sqrt(2 / π) .* sin.(n .* xq)
            Φ = [sum(wq .* es(m) .* φfun.(xq) .* es(n)) for m in 1:N2, n in 1:N2]
            # a state: Gaussian in the mode momentum p_n = ℏ_N n with ℏ_N = 6/N, band-limited in x
            ℏN = 6 / N
            ρ = zeros(ComplexF64, N2, N2)
            for n in 1:(N ÷ 2), m in 1:(N ÷ 2)

                abs(m - n) <= 2 || continue
                ρ[m, n] = exp(-(ℏN * (m + n) / 2)^2 / 2) * (m == n ? 1.0 : 0.25)
            end
            full = commutator(Φ, ρ) / (im * ℏN)
            trunc = copy(full)
            trunc[(N + 1):N2, :] .= 0
            trunc[:, (N + 1):N2] .= 0
            leak = norm(full - trunc) / norm(full)
            push!(leaks, leak)
            @printf("    %-64s N = %2d: relative leak %.3e   ratio %.2f\n", name, N, leak,
                isnan(prevleak) ? NaN : prevleak / leak)
            prevleak = leak
        end
        if noleak
            check(
                "a zero-field-compatible φ does not leak at any N (state on n ≤ N/2 stays inside)",
                maximum(leaks) < 1e-12, @sprintf("max leak %.3e", maximum(leaks)))
        else
            check(
                "the kink of $(strip(name)) leaks at every N (the sine tail of a non-compatible φ)",
                minimum(leaks) > 1e-8, @sprintf("min leak %.3e", minimum(leaks)))
        end
    end
    println("    (the state occupies n ≤ N/2; cos 2x maps it to n ≤ N/2 + 2 < N, so nothing leaves the")
    println("     block; the kink potential leaks algebraically — the sine tail of a non-compatible φ)")
end

# ---------------------------------------------------------------------------------------------
# 5. dynamics: Weyl symbol of the von Neumann–Poisson field against the Vlasov field
# ---------------------------------------------------------------------------------------------
#
# Periodic circle, f(x,p) = g(x) exp(−p²/2σ²), g = 1 + cos(x)/2, σ = 1, v_max = 6 = ℏK.
# ρ = T(2πℏ f): [ρ]_{kl} = 2πℏ ĝ_{k−l} exp(−p_mid²/2σ²), tr ρ ≈ ∫∫ f.  The Weyl symbol of a
# matrix M is read off entrywise: M_{kl} is the x-Fourier coefficient of frequency j = k − l of
# the symbol at momentum p = ℏ(k+l)/2 (on the cylinder the pointwise Wigner function is a
# checkerboard on the half-integer lattice — even j at integer p/ℏ, odd j at half-integer — so the
# comparison is made coefficient-wise, never pointwise).
# Self-consistent potential:  n̂_j = (1/2π) Σ_k ρ_{k+j,k},  φ̂_j = n̂_j / j²  (−φ'' = n − n̄).

header("5. vector-field consistency: symbol of (1/iℏ)[T h_ρ, ρ] vs φ'∂_p f − p ∂_x f, refinement in K")
let σ = 1.0, vmax = 6.0, ε = 0.5
    ĝ = Dict(0 => 1.0 + 0im, 1 => ε / 2 + 0im, -1 => ε / 2 + 0im)   # g(x) = 1 + ε cos x
    results = Float64[]
    resid = Float64[]
    tr3 = Float64[]
    hs = Float64[]
    # K ≥ 7 keeps the Poisson-summation aliasing 2exp(−(2/3)π²σ²/ℏ²) of tr ρ³ below 3e-4, well under
    # its O(ℏ²) semiclassical defect; N = 2K + 1 ≤ 31.
    for K in (7, 10, 15)
        ℏ = vmax / K
        N = 2K + 1
        idx(k) = k + K + 1
        pk(k) = ℏ * k
        ρ = zeros(ComplexF64, N, N)
        for l in (-K):K, k in (-K):K

            haskey(ĝ, k - l) || continue
            ρ[idx(k), idx(l)] = 2π * ℏ * ĝ[k - l] * exp(-(ℏ * (k + l) / 2)^2 / (2σ^2))
        end
        # density and potential
        n̂ = Dict{Int, ComplexF64}()
        for j in -2:2
            n̂[j] = sum(ρ[idx(k + j), idx(k)] for k in (-K):K if abs(k + j) <= K) / (2π)
        end
        φ̂ = Dict(j => (j == 0 ? 0.0im : n̂[j] / j^2) for j in -2:2)
        H = zeros(ComplexF64, N, N)
        for l in (-K):K, k in (-K):K

            k == l && (H[idx(k), idx(l)] += pk(k)^2 / 2)
            haskey(φ̂, k - l) && (H[idx(k), idx(l)] += φ̂[k - l])
        end
        Xρ = commutator(H, ρ) / (im * ℏ)
        # coefficient-wise comparison away from the momentum edge (|p_mid| ≤ vmax/2)
        err = 0.0
        scl = 0.0
        predmax = 0.0
        err2 = 0.0
        for l in (-K):K, k in (-K):K

            abs(k + l) <= K || continue
            j = k - l
            abs(j) <= 3 || continue
            p = ℏ * (k + l) / 2
            ex = exp(-p^2 / (2σ^2))
            # x-Fourier coefficient j at momentum p of V = φ'∂_p f − p ∂_x f, and of the Wigner
            # correction W = −(ℏ²/24) φ''' ∂_p³ f,  ∂_p³ exp(−p²/2σ²) = (3p/σ⁴ − p³/σ⁶) exp(−p²/2σ²)
            V = -p * (im * j) * get(ĝ, j, 0.0im) * ex
            W = 0.0im
            for (j1, c) in φ̂
                haskey(ĝ, j - j1) || continue
                V += (im * j1) * c * ĝ[j - j1] * (-(p / σ^2)) * ex
                W += -ℏ^2 / 24 * (im * j1)^3 * c * ĝ[j - j1] * (3p / σ^4 - p^3 / σ^6) * ex
            end
            X = Xρ[idx(k), idx(l)] / (2π * ℏ)
            err = max(err, abs(X - V))
            err2 = max(err2, abs(X - V - W))
            scl = max(scl, abs(V))
            predmax = max(predmax, abs(W))
        end
        push!(results, err / scl)
        push!(resid, err2 / scl)
        push!(hs, ℏ)
        # Casimirs vs ∫∫ f^k:  ∫g^k dx = 2π(1, 1+ε²/2, 1+3ε²/2), ∫exp(−k p²/2σ²) dp = σ√(2π/k).
        # k = 1, 2 hold up to the Poisson-summation aliasing 2exp(−2π²σ²/ℏ²), 2exp(−π²σ²/ℏ²).
        gk = (2π, 2π * (1 + ε^2 / 2), 2π * (1 + 3ε^2 / 2))
        for k in 1:3
            lhs = real(tr(ρ^k))
            rhs = (2π * ℏ)^(k - 1) * gk[k] * σ * sqrt(2π / k)
            rel = abs(lhs - rhs) / abs(rhs)
            k == 3 && push!(tr3, rel)
            alias = 2 * exp(-(2 / k) * π^2 * σ^2 / ℏ^2)
            k <= 2 && check("K = $K: tr ρ^$k = (2πℏ)^$(k-1) ∫∫ f^$k up to aliasing",
                rel <= 1e-8 + 10alias,
                @sprintf("%.2e  (aliasing bound %.1e)", rel, alias))
            k == 3 &&
                @printf("    K = %2d: tr ρ³ vs (2πℏ)² ∫∫ f³: relative difference %.3e\n", K,
                    rel)
        end
        @printf("    K = %2d, ℏ = %.3f: max rel. |σ(Xρ) − 2πℏ·Vlasov| = %.3e   (ℏ²/24)φ'''∂_p³f term = %.3e   after subtracting it: %.3e\n",
            K, ℏ, err / scl, predmax / scl, err2 / scl)
    end
    for i in 2:length(results)
        rate = results[i - 1] / results[i]
        expected = (hs[i - 1] / hs[i])^2
        check(
            "refinement step $(i - 1): error ratio ≈ (ℏ_coarse/ℏ_fine)² = $(round(expected, digits = 2))",
            abs(rate / expected - 1) < 0.25, @sprintf("ratio %.2f", rate))
        # Both steps, not only the last: the first used to be printed with an excuse rather than
        # checked, and it meets the same window.  Note the window is wide enough that it does not
        # separate ℏ³ from ℏ⁴ -- the ℏ⁶ terms are still visible at these K.
        rate2 = resid[i - 1] / resid[i]
        check(
            "refinement step $(i - 1): residual after the ℏ² term ≈ (ℏ_coarse/ℏ_fine)⁴ = $(round(expected^2, digits = 2))",
            abs(rate2 / expected^2 - 1) < 0.35,
            @sprintf("ratio %.2f  (ℏ³ would predict %.2f)", rate2, expected^1.5))
        rate3 = tr3[i - 1] / tr3[i]
        check("refinement step $(i - 1): tr ρ³ defect ratio ≈ (ℏ_coarse/ℏ_fine)²",
            abs(rate3 / expected - 1) < 0.25,
            @sprintf("ratio %.2f", rate3))
    end
    check("the error IS the −(ℏ²/24) φ''' ∂_p³ f term (residual ≪ error at the finest K)",
        resid[end] / results[end] < 0.05, @sprintf("residual / error = %.4f",
            resid[end] / results[end]))
end

summary("I1_vlasov_uN.jl")
