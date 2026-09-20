#!/usr/bin/env julia
#
# III.2 -- the matrix analogue of the c-bracket:  {F,G}_c(W) = tr( c(W) [F_W, G_W] )  on gl(N).
#
#     julia --project=scripts scripts/fable/III2_matrix_structure_function.jl
#
# The continuum theorem (four-bracket paper, thm:jacobi) says {A,B}_c = ∫ c(u)[A_u,B_u] is
# Poisson for EVERY smooth c.  Zeitlin's sine algebra quantises the same Lie algebra, and the
# natural finite-N analogue is tr(c(W)[F_W,G_W]) with c a spectral (matrix) function.
#
#   1. exact Jacobiator over Q for polynomial c, N = 2, 3, 4  (affine c must vanish; W^2 must not
#      for N >= 3; N = 2 is degenerate and passes for every c -- Cayley-Hamilton);
#   2. the closed form  Jac(X,Y,Z) = Σ_cyc tr( Dc(W)^2 [[Z,W]] · [X,Y] ), checked exactly against
#      the coordinate Jacobiator, together with the chain-rule identity [Z,c(W)] = Dc(W)[[Z,W]];
#   3. the eigenbasis form  Jac = Σ_{a,b,c distinct} Φ_abc X_ab (Y_bc Z_ca − Z_bc Y_ca)  with
#      Φ_abc = Σ_cyc κ_ab (w_b − w_a),  κ_ab = c^[1](w_a,w_b)^2:  Jacobi at W  ⇔  the antisymmetric
#      edge weight κ_ab (w_b − w_a) is a cocycle on the complete graph of the eigenvalues, i.e.
#      exact.  Demanding it for all W forces c affine (base-point identity, checked exactly);
#   4. Casimirs: tr κ(W) is a Casimir of every c-bracket, Poisson or not; the flow is Lax,
#      Ẇ = [H_W, c(W)] = [Dc(W)[H_W], W];
#   5. semiclassical scaling on Zeitlin's su(N): the normalised Jacobiator of tr(c(W)[·,·]) on
#      band-limited data is O(N^{-2}) as N → ∞, for c = W^2, W^3, truncated exp, W^{3/2};
#   6. the pushforward route: the Lie-Poisson bracket pushed forward under V = W^2 is exactly
#      Poisson, equals  tr(W[WX+XW, WY+YW])  (W = V^{1/2}), and converges to (4/3) tr(V^{3/2}[X,Y]);
#   7. spectral-kernel brackets Σ θ(v_a,v_b) X_ab Y_ba: the general Jacobiator
#      Ψ_abc = ω_ab ω_bc (v_a − v_c) + cyc,  ω = θ/(v_a − v_b),  and the classification
#      Poisson ⇔ ω = 1/ψ^[1] (pushforward of Lie-Poisson under V = ψ^{-1}(W)).  Exact: the cubic
#      bracket θ = ab(a−b) is Poisson, the linear-cubic pencil is not; the log-mean kernel is the
#      exactly Poisson counterpart of tr(V^2[X,Y]) and agrees with it to O(N^{-2}).
#
# Pairing convention: dF = tr(F_W dW), so (F_W)_{ba} = ∂F/∂W_{ab}; for F = tr(X W), F_W = X.
# Coordinates W_{ab} ↦ l = a + (b−1)N (column-major vec), so the gradient vector of F is
# vec(F_W'), and the structure matrix of tr(C[·,·]) is P_{(ab),(cd)} = C_{cb} δ_{ad} − C_{ad} δ_{cb}.
#
# Real form: the Jacobiator is a polynomial identity with real coefficients, so vanishing on
# gl(N,Q) is equivalent to vanishing on u(N) (its complexification is gl(N,C)).  The exact tests
# therefore use general rational W and general X, Y, Z.

using LinearAlgebra
using Random
using Printf
using GeometricBrackets: jacobi_residual, exact_rank

include(joinpath(@__DIR__, "..", "check.jl"))
using .Checks: header, check, summary, fmt

const Q = Rational{BigInt}
const rng = MersenneTwister(2026_09_03)

# ---------------------------------------------------------------------------------------------
# generic machinery
# ---------------------------------------------------------------------------------------------

"Random rational matrix with entries p/q, |p| ≤ 3, 1 ≤ q ≤ 3."
randq(n, m = n) = [Q(rand(rng, -3:3), rand(rng, 1:3)) for _ in 1:n, _ in 1:m]

"Elementary matrix E_l with l = a + (b−1)N ↔ (a,b)."
function elementary(::Type{T}, N, l) where {T}
    E = zeros(T, N, N)
    E[l] = one(T)
    return E
end

"Gradient vector of F = tr(X W) in the coordinates W_l: vec(X')."
gradvec(X) = vec(transpose(X))

"P_{(ab),(cd)} = C_{cb} δ_{ad} − C_{ad} δ_{cb}: the structure matrix of (X,Y) ↦ tr(C [X,Y])."
function bracket_matrix(C::AbstractMatrix{T}) where {T}
    N = size(C, 1)
    P = zeros(T, N^2, N^2)
    for d in 1:N, c in 1:N, b in 1:N, a in 1:N
        i = a + (b - 1) * N
        j = c + (d - 1) * N
        v = zero(T)
        a == d && (v += C[c, b])
        c == b && (v -= C[a, d])
        P[i, j] = v
    end
    return P
end

"dP[l,i,j] = ∂P_ij/∂W_l for P = bracket_matrix(c(W)), given E ↦ Dc(W)[E]."
function bracket_derivative(Dc, N, ::Type{T}) where {T}
    dP = zeros(T, N^2, N^2, N^2)
    for l in 1:(N ^ 2)
        dP[l, :, :] = bracket_matrix(Dc(elementary(T, N, l)))
    end
    return dP
end

# A polynomial spectral function c(W) = Σ_k a[k+1] W^k with its Fréchet derivative
#     Dc(W)[E] = Σ_k a[k+1] Σ_{r=0}^{k−1} W^r E W^{k−1−r} .
struct Poly{T}
    a::Vector{T}
end
(c::Poly)(W) = sum(c.a[k + 1] * W^k for k in 0:(length(c.a) - 1))
(c::Poly)(w::Number) = sum(c.a[k + 1] * w^k for k in 0:(length(c.a) - 1))
function fderiv(c::Poly, W)
    N = size(W, 1)
    pw = [W^r for r in 0:(length(c.a) - 1)]      # pw[r+1] = W^r
    return E -> begin
        out = zeros(promote_type(eltype(W), eltype(E)), N, N)
        for k in 1:(length(c.a) - 1)
            iszero(c.a[k + 1]) && continue
            for r in 0:(k - 1)
                out += c.a[k + 1] * pw[r + 1] * E * pw[k - r]
            end
        end
        out
    end
end

# A general spectral function of a Hermitian matrix through its eigen-decomposition, with the
# Daleckii–Krein formula for the Fréchet derivative:  Dc(W)[E] = U (K .* (U' E U)) U',
# K_ij = (c(w_i) − c(w_j))/(w_i − w_j),  K_ii = c'(w_i).
struct Spectral{F, G}
    f::F
    df::G
end
function (c::Spectral)(W)
    w, U = eigen(Hermitian(W))
    return U * Diagonal(c.f.(w)) * U'
end
function fderiv(c::Spectral, W)
    w, U = eigen(Hermitian(W))
    n = length(w)
    # divided differences, with the derivative on (numerically) coincident eigenvalues: the symbol
    # matrices have a highly degenerate spectrum and the naive quotient is 0/0 there
    K = [abs(w[i] - w[j]) <= 1e-9 * (abs(w[i]) + abs(w[j])) ? c.df((w[i] + w[j]) / 2) :
         (c.f(w[i]) - c.f(w[j])) / (w[i] - w[j]) for i in 1:n, j in 1:n]
    return E -> U * (K .* (U' * E * U)) * U'
end

"The structure matrix and its derivative tensor of tr(c(W)[·,·]) at W."
function tensors(c, W)
    Dc = fderiv(c, W)
    return bracket_matrix(c(W)), bracket_derivative(Dc, size(W, 1), eltype(c(W)))
end

"Σ_cyc {{F,G},H} for linear F = tr(XW), G = tr(YW), H = tr(ZW), from the coordinate tensors."
function jacobiator_coord(P, dP, X, Y, Z)
    N2 = size(P, 1)
    term(x, y, z) = begin
        g = [sum(dP[l, i, j] * x[i] * y[j] for i in 1:N2, j in 1:N2) for l in 1:N2]   # ∇{F,G}
        transpose(g) * P * z                                                            # {{F,G},H}
    end
    x, y, z = gradvec(X), gradvec(Y), gradvec(Z)
    return term(x, y, z) + term(y, z, x) + term(z, x, y)
end

"The closed form  Σ_cyc tr( Dc(W)[ [Z, c(W)] ] · [X,Y] ) = Σ_cyc tr( Dc(W)^2 [[Z,W]] · [X,Y] )."
function jacobiator_closed(c, W, X, Y, Z)
    Dc = fderiv(c, W)
    C = c(W)
    term(X, Y, Z) = tr(Dc(Z * C - C * Z) * (X * Y - Y * X))
    return term(X, Y, Z) + term(Y, Z, X) + term(Z, X, Y)
end

comm(A, B) = A * B - B * A

"Divided difference c^[1](a,b) of a scalar function, c'(a) on the diagonal."
divdiff(c, dc, a, b) = a == b ? dc(a) : (c(a) - c(b)) / (a - b)

"""
The eigenbasis Jacobiator of a spectral-kernel bracket  B(X,Y) = Σ_ab θ(v_a,v_b) X_ab Y_ba
at V = diag(v), for gradient matrices X, Y, Z given in the eigenbasis:

    J = Σ_{a,b,c distinct} Ψ_abc X_ab (Y_bc Z_ca − Z_bc Y_ca),
    Ψ_abc = ω_ab ω_bc (v_a − v_c) + ω_bc ω_ca (v_b − v_a) + ω_ca ω_ab (v_c − v_b),   ω_ab = θ_ab/(v_a − v_b).

`ω(a, b)` is passed directly (for the c-bracket ω = c^[1], for a pushforward ω = 1/ψ^[1]).
"""
function jacobiator_kernel(ω, v, X, Y, Z)
    N = length(v)
    J = zero(promote_type(eltype(v), eltype(X)))
    for a in 1:N, b in 1:N, c in 1:N
        (a == b || b == c || a == c) && continue
        Ψ = ω(v[a], v[b]) * ω(v[b], v[c]) * (v[a] - v[c]) +
            ω(v[b], v[c]) * ω(v[c], v[a]) * (v[b] - v[a]) +
            ω(v[c], v[a]) * ω(v[a], v[b]) * (v[c] - v[b])
        J += Ψ * X[a, b] * (Y[b, c] * Z[c, a] - Z[b, c] * Y[c, a])
    end
    return J
end

"Ψ_abc for the three eigenvalues (a, b, c) and the symmetric slope function ω."
function Ψ(ω, a, b, c)
    ω(a, b) * ω(b, c) * (a - c) + ω(b, c) * ω(c, a) * (b - a) + ω(c, a) * ω(a, b) * (c - b)
end

# ---------------------------------------------------------------------------------------------
header("1. exact Jacobiator over Q, N = 2, 3, 4: which polynomial c give a Poisson bracket")
# ---------------------------------------------------------------------------------------------
println("   residual = jacobi_residual(P, dP), normalised by the largest term formed; exact.")

function exact_case(label, c, N; expect_zero)
    W = randq(N)
    P, dP = tensors(c, W)
    r = jacobi_residual(P, dP)
    ok = expect_zero ? iszero(r) : !iszero(r)
    check(
        @sprintf("N=%d  %-34s %s", N, label,
            expect_zero ? "Jacobiator == 0" : "Jacobiator != 0"),
        ok, "residual = " * fmt(r))
    return r
end

for N in (2, 3, 4)
    Z0 = randq(N)                                     # a frozen constant shift
    exact_case("c = W  (Lie-Poisson)", Poly([Q(0), Q(1)]), N; expect_zero = true)
    exact_case("c = 3W − 2·1  (affine)", Poly([Q(-2), Q(3)]), N; expect_zero = true)
    # c(W) = W + Z0: Dc = identity -- a Poly cannot hold the shift, so build it by hand
    let W = randq(N)
        P = bracket_matrix(W + Z0)
        dP = bracket_derivative(identity, N, Q)
        check(
            @sprintf("N=%d  %-34s Jacobiator == 0", N, "c = W + Z  (frozen shift, pencil)"),
            iszero(jacobi_residual(P, dP)), "residual = " * fmt(jacobi_residual(P, dP)))
    end
    # Casimir-rescaled: c(W) = tr(W^2) W, Dc[E] = 2 tr(W E) W + tr(W^2) E
    let W = randq(N)
        C = tr(W^2) * W
        Dc = E -> 2 * tr(W * E) * W + tr(W^2) * E
        P, dP = bracket_matrix(C), bracket_derivative(Dc, N, Q)
        check(
            @sprintf("N=%d  %-34s Jacobiator == 0", N, "c = tr(W^2) W  (Casimir-rescaled)"),
            iszero(jacobi_residual(P, dP)), "residual = " * fmt(jacobi_residual(P, dP)))
    end
    quad = N == 2                                     # N = 2 is degenerate: every c passes
    exact_case("c = W^2", Poly([Q(0), Q(0), Q(1)]), N; expect_zero = quad)
    exact_case("c = 1 + 2W + W^2", Poly([Q(1), Q(2), Q(1)]), N; expect_zero = quad)
    exact_case("c = W^3", Poly([Q(0), Q(0), Q(0), Q(1)]), N; expect_zero = quad)
    exact_case("c = 1 + W + W^2/2 + W^3/6  (exp)", Poly([Q(1), Q(1), Q(1, 2), Q(1, 6)]), N;
        expect_zero = quad)
end
println("   N = 2 passes for every c: by Cayley-Hamilton c(W) = α(tr W, det W) W + β(tr W, det W) 1,")
println("   so tr(c(W)[X,Y]) = α(Casimirs) · tr(W[X,Y]) is a Casimir rescaling of Lie-Poisson.")
println("   N = 2 is therefore NOT a valid control here (the analogue of so(3)).")

# ---------------------------------------------------------------------------------------------
header("2. the closed-form Jacobiator, exactly")
# ---------------------------------------------------------------------------------------------
for N in (3, 4),
    (label, c) in (("W^2", Poly([Q(0), Q(0), Q(1)])), (
        "W^3", Poly([Q(0), Q(0), Q(0), Q(1)])),
        ("exp-trunc", Poly([Q(1), Q(1), Q(1, 2), Q(1, 6)])))

    W = randq(N)
    X, Y, Z = randq(N), randq(N), randq(N)
    Dc = fderiv(c, W)
    C = c(W)
    # chain rule: d/dt c(e^{tZ} W e^{-tZ}) at t = 0 in two ways
    check(@sprintf("N=%d  c=%-9s [Z, c(W)] == Dc(W)[[Z, W]]", N, label),
        iszero(comm(Z, C) - Dc(comm(Z, W))))
    P, dP = tensors(c, W)
    jc = jacobiator_coord(P, dP, X, Y, Z)
    jf = jacobiator_closed(c, W, X, Y, Z)
    check(
        @sprintf("N=%d  c=%-9s coordinate Jacobiator == closed form", N, label), jc == jf,
        "value = " * fmt(jf))
    check(@sprintf("N=%d  c=%-9s (and it is nonzero on random X,Y,Z)", N, label), !iszero(jf))
end

# ---------------------------------------------------------------------------------------------
header("3. eigenbasis form: Jac = Σ_{abc} Φ_abc X_ab (Y_bc Z_ca − Z_bc Y_ca),  Φ = δ(κ δw)")
# ---------------------------------------------------------------------------------------------
println("   κ_ab = c^[1](w_a,w_b)^2.  Φ_abc = κ_ab(w_b−w_a) + κ_bc(w_c−w_b) + κ_ca(w_a−w_c) is the")
println("   coboundary of the antisymmetric edge weight θ_ab = κ_ab (w_b − w_a) on the complete graph")
println("   K_N of the eigenvalues; Jac ≡ 0 at W  ⇔  θ is a cocycle  ⇔  θ_ab = f_b − f_a (exact).")

"S_ij(X,Y,Z) = Σ_cyc Z_ij [X,Y]_ji as an N×N matrix in (i,j)."
function Smat(X, Y, Z)
    t(X, Y, Z) = Z .* transpose(comm(X, Y))
    return t(X, Y, Z) + t(Y, Z, X) + t(Z, X, Y)
end

"Distinct random rational eigenvalues."
function distinct_spectrum(N)
    w = Q[rand(rng, 1:20) for _ in 1:N]
    while length(unique(w)) < N
        w = Q[rand(rng, 1:20) for _ in 1:N]
    end
    return w
end

for N in (3, 4)
    w = distinct_spectrum(N)
    W = Matrix(Diagonal(w))
    X, Y, Z = randq(N), randq(N), randq(N)
    for (label, c) in (("W^2", Poly([Q(0), Q(0), Q(1)])), (
        "W^3", Poly([Q(0), Q(0), Q(0), Q(1)])))
        dc = a -> sum(k * c.a[k + 1] * a^(k - 1) for k in 1:(length(c.a) - 1))
        ω = (a, b) -> divdiff(c, dc, a, b)                     # ω = c^[1]  (so ω^2 = κ)
        jeig = jacobiator_kernel(ω, w, X, Y, Z)
        jf = jacobiator_closed(c, W, X, Y, Z)
        # !iszero(jf) as in section 2: two implementations both returning zero would agree.
        check(
            @sprintf("N=%d  c=%-4s eigenbasis Φ-formula == closed form, and nonzero", N,
                label),
            jeig == jf && !iszero(jf),
            "value = " * fmt(jf))
    end

    # the edge forms T_ab := (w_b − w_a)(S_ab − S_ba), a<b, as trilinear forms on basis triples;
    # Jac = Σ_{a<b} κ_ab T_ab.  Their span is the image of the coboundary δ¹ on K_N, of rank
    # M − (N − 1) = (N−1)(N−2)/2, not M − 1: the relations are the exact 1-forms, dimension N − 1.
    pairs = [(i, j) for i in 1:N for j in (i + 1):N]
    M = length(pairs)
    Tm = zeros(Q, M, N^6)
    col = 0
    for r in 1:(N ^ 2), q in 1:(N ^ 2), p in 1:(N ^ 2)
        col += 1
        S = Smat(elementary(Q, N, p), elementary(Q, N, q), elementary(Q, N, r))
        for (k, (i, j)) in enumerate(pairs)
            Tm[k, col] = (w[j] - w[i]) * (S[i, j] - S[j, i])
        end
    end
    rk = exact_rank(Tm)
    check(
        @sprintf("N=%d  rank{T_ab : a<b} == (N−1)(N−2)/2 = %d  (rank of δ¹ on K_N)", N,
            (N - 1) * (N - 2) ÷ 2),
        rk == (N - 1) * (N - 2) ÷ 2, "rank = $rk")
    check(@sprintf("N=%d  κ ≡ const  ⇒  Σ_{a<b} T_ab == 0  (Lie-Poisson Jacobi)", N),
        iszero(sum(Tm; dims = 1)))
    # a NON-constant κ that is Poisson at this W: κ_ab (w_b − w_a) = f_b − f_a for random f
    f = Q[rand(rng, -9:9) for _ in 1:N]
    κexact = [(f[j] - f[i]) / (w[j] - w[i]) for (i, j) in pairs]
    check(
        @sprintf("N=%d  κ_ab = (f_b−f_a)/(w_b−w_a), f random  ⇒  Σ κ_ab T_ab == 0  (exact weight)",
            N),
        iszero(transpose(κexact) * Tm) && length(unique(κexact)) > 1)
    # a random symmetric κ is not
    κrand = Q[rand(rng, 1:9) for _ in pairs]
    check(@sprintf("N=%d  κ random  ⇒  Σ κ_ab T_ab != 0  (control)", N), !iszero(transpose(κrand) *
                                                                                 Tm))
end

println("   For all W: Φ_{a,x,y} with base point a has a squared numerator (checked exactly below),")
println("   so Φ ≡ 0 on every triple  ⇔  (c(x)−c(a))/(x−a) independent of x  ⇔  c affine.")
for (label, c) in (("W^2", Poly([Q(0), Q(0), Q(1)])), (
    "W^3", Poly([Q(0), Q(0), Q(0), Q(1)])),
    ("exp-trunc", Poly([Q(1), Q(1), Q(1, 2), Q(1, 6)])), ("3W − 2", Poly([Q(-2), Q(3)])))
    a, x, y = distinct_spectrum(3)
    κ(p, q) = ((c(p) - c(q)) / (p - q))^2
    Φ = κ(a, x) * (x - a) + κ(x, y) * (y - x) + κ(y, a) * (a - y)
    num = ((y - a) * (c(x) - c(a)) - (x - a) * (c(y) - c(a)))^2
    rhs = num / ((x - a) * (y - a) * (y - x))
    check(
        @sprintf("c=%-9s Φ_{a,x,y} == ((y−a)(c_x−c_a) − (x−a)(c_y−c_a))^2 / ((x−a)(y−a)(y−x))",
            label),
        Φ == rhs, "Φ = " * fmt(Φ))
end

# ---------------------------------------------------------------------------------------------
header("4. Casimirs and the Lax form, for non-Poisson c")
# ---------------------------------------------------------------------------------------------
for N in (3, 4),
    (label, c) in (("W^2", Poly([Q(0), Q(0), Q(1)])), (
        "W^3", Poly([Q(0), Q(0), Q(0), Q(1)])))

    W = randq(N)
    P = bracket_matrix(c(W))
    allzero = true
    for k in 1:4                                  # tr W^k, gradient k W^{k−1}
        g = gradvec(k * W^(k - 1))
        allzero &= iszero(P * g)
    end
    check(
        @sprintf("N=%d  c=%-4s P ∇tr(W^k) == 0, k = 1..4  (Casimirs, though not Poisson)",
            N, label),
        allzero)
    Z0 = randq(N)
    check(
        @sprintf("N=%d  c=%-4s P ∇tr(Z0 W) != 0  (control: a non-spectral functional)", N,
            label),
        !iszero(P * gradvec(Z0)))
    H = randq(N)
    Dc = fderiv(c, W)
    check(
        @sprintf("N=%d  c=%-4s [H_W, c(W)] == [Dc(W)[H_W], W]  (Lax form, isospectral)", N,
            label),
        iszero(comm(H, c(W)) - comm(Dc(H), W)))
end

# ---------------------------------------------------------------------------------------------
header("5. semiclassical scaling on Zeitlin's su(N): band-limited data, N → ∞")
# ---------------------------------------------------------------------------------------------

"Clock-and-shift generators T_m of `sine_algebra(N)`, as matrices, for the modes in `modes`."
function zeitlin_generators(N, modes)
    ζ = cis(4π / N)
    g = Matrix(Diagonal([ζ^a for a in 0:(N - 1)]))
    h = zeros(ComplexF64, N, N)
    for i in 1:N
        h[i, mod1(i + 1, N)] = 1
    end
    return Dict(m => cis(2π * m[1] * m[2] / N) * g^mod(m[1], N) * h^mod(m[2], N)
    for m in modes)
end

"Hermitian 'cos/sin' basis on the window |m|_∞ ≤ K (one representative per ±m)."
function hermitian_window(N, K)
    half = [(m1, m2) for m1 in (-K):K for m2 in (-K):K if (m1, m2) > (0, 0)]
    T = zeitlin_generators(N, half)
    B = Matrix{ComplexF64}[]
    for m in half
        push!(B, (T[m] + T[m]') / 2)
        push!(B, (T[m] - T[m]') / (2im))
    end
    return B
end

"W = Op(u) for a fixed real band-limited symbol; the coefficients are fixed across N."
function symbol_matrix(N, K, coeffs, shift)
    B = hermitian_window(N, K)
    W = Matrix{ComplexF64}(shift * I, N, N)
    for (b, a) in zip(B, coeffs)
        W += a * b
    end
    return Hermitian(W)
end

"""
Normalised Jacobiator of tr(c(W)[·,·]) over all triples of the Hermitian window basis:
max |Σ_cyc term| / max |term|,  term(X,Y,Z) = tr( Dc(W)[[Z,c(W)]] · [X,Y] ).
"""
function semiclassical_residual(c, W, B)
    Dc = fderiv(c, Matrix(W))
    C = c(Matrix(W))
    Qz = [transpose(Dc(comm(Z, C))) for Z in B]           # so that tr(Q A) = sum(Qᵀ .* A)
    n = length(B)
    term = zeros(ComplexF64, n, n, n)
    for i in 1:n, j in (i + 1):n

        A = comm(B[i], B[j])
        for k in 1:n
            term[i, j, k] = sum(Qz[k] .* A)
            term[j, i, k] = -term[i, j, k]
        end
    end
    res = 0.0
    for k in 1:n, j in 1:n, i in 1:n
        res = max(res, abs(term[i, j, k] + term[j, k, i] + term[k, i, j]))
    end
    return res / maximum(abs, term)
end

"Least-squares slope of log r against log N."
function rate(Ns, rs)
    x, y = log.(Ns), log.(rs)
    return sum((x .- sum(x) / length(x)) .* (y .- sum(y) / length(y))) /
           sum((x .- sum(x) / length(x)) .^ 2)
end

# The window has to be narrow for the asymptotics to be visible: the Moyal corrections are
# (2π m×n/N)^2 at fixed modes, and the terms formed here are products of six band-limited
# factors, so with |m|_∞ ≤ 2 the regime 2π|m×n|/N ≪ 1 is not reached before N ~ 10^3.  With
# |m|_∞ ≤ 1 the N^{-2} plateau is clean from N ≈ 65 on.
const K_TEST = 1                                           # window of the test functions
const K_SYM = 1                                            # band limit of the symbol u
const NSYM = 2 * length([(m1, m2) for m1 in (-K_SYM):K_SYM
                     for m2 in (-K_SYM):K_SYM if (m1, m2) > (0, 0)])
const COEFFS = 0.1 * (2 * rand(rng, NSYM) .- 1)            # |a| ≤ 0.1: ‖W − 3·1‖ ≤ 0.8, W > 0
const SHIFT = 3.0
const Ns = [5, 9, 17, 33, 65, 129, 257, 513]
const NTAIL = 4                                            # slope over the last four N

cases = (
    ("c = W        (control, exact)", Poly([0.0, 1.0])),
    ("c = W^2", Poly([0.0, 0.0, 1.0])),
    ("c = W^3", Poly([0.0, 0.0, 0.0, 1.0])),
    ("c = exp-trunc", Poly([1.0, 1.0, 0.5, 1 / 6])),
    ("c = W^{3/2}  (spectral)", Spectral(x -> x^1.5, x -> 1.5 * sqrt(x)))
)
println("   symbol u: fixed random coefficients on |m|_∞ ≤ $K_SYM, W = 3·1 + Op(u) > 0; test functions:")
println("   the Hermitian window |m|_∞ ≤ $K_TEST ($NSYM matrices, all triples).  Table: r, then N^2 r.")
println()

function print_row(label, Ns, rs)
    @printf("   %-30s", label)
    for r in rs
        @printf(" %8.1e", r)
    end
    @printf("   %6.2f\n", rate(Ns[(end - NTAIL + 1):end], rs[(end - NTAIL + 1):end]))
    @printf("   %-30s", "    N^2 r")
    for (N, r) in zip(Ns, rs)
        @printf(" %8.1e", N^2 * r)
    end
    println()
end

@printf("   %-30s", "N")
for N in Ns
    @printf(" %8d", N)
end
println("     slope")

# the symbol matrices and windows, once per N
const SYMBOLS = Dict(N => symbol_matrix(N, K_SYM, COEFFS, SHIFT) for N in Ns)
const WINDOWS = Dict(N => hermitian_window(N, K_TEST) for N in Ns)

results = Dict{String, Vector{Float64}}()
for (label, c) in cases
    rs = [semiclassical_residual(c, SYMBOLS[N], WINDOWS[N]) for N in Ns]
    results[label] = rs
    print_row(label, Ns, rs)
end
println()
check("c = W: residual at roundoff for every N",
    maximum(results["c = W        (control, exact)"]) < 1e-12,
    @sprintf("max %.1e", maximum(results["c = W        (control, exact)"])))
for label in ("c = W^2", "c = W^3", "c = exp-trunc", "c = W^{3/2}  (spectral)")
    rs = results[label]
    sl = rate(Ns[(end - NTAIL + 1):end], rs[(end - NTAIL + 1):end])
    check("$label: residual far above roundoff at N = 5  (not Poisson at finite N)",
        rs[1] > 1e-6,
        @sprintf("r(5) = %.2e", rs[1]))
    check(
        "$label: residual decays at rate N^{-2}  (slope over last $NTAIL in [-2.2, -1.8])",
        -2.2 <= sl <= -1.8, @sprintf("slope = %.2f, N^2 r(%d) = %.3e", sl, Ns[end],
            Ns[end]^2 * rs[end]))
end
println("   O(N^{-2}) = O(ℏ^2): the O(ℏ) Moyal terms of Dc(W)[E] cancel (symmetrised products), so the")
println("   finite-N bracket is the continuum c-bracket up to relative O(ℏ^2), whose Jacobiator vanishes.")

# ---------------------------------------------------------------------------------------------
header("6. the pushforward route: Lie-Poisson under V = W^2, compared with (4/3) tr(V^{3/2}[X,Y])")
# ---------------------------------------------------------------------------------------------
println("   {F,G}_φ(V) := tr( W [DφX, DφY] ),  W = V^{1/2},  Dφ(W)X = WX + XW  (exactly Poisson by")
println("   construction).  Continuum limit: c'(v) = φ'(φ⁻¹(v)) = 2√v, i.e. c(v) = (4/3) v^{3/2}.")

pushforward(W, X, Y) = tr(W * comm(W * X + X * W, W * Y + Y * W))
cform(W, X, Y) = 4 / 3 * tr(W^3 * comm(X, Y))

# 6a. finite-difference Jacobiator on a generic SPD V at N = 3: pushforward ≈ 0, c-form O(1)
function fd_tensors(bilinear, V; h = 1e-5)
    N = size(V, 1)
    Xs = [transpose(elementary(Float64, N, l)) for l in 1:(N ^ 2)]   # gradvec(Xs[l]) = e_l
    Pof(V) = [bilinear(V, Xs[i], Xs[j]) for i in 1:(N ^ 2), j in 1:(N ^ 2)]
    P = Pof(V)
    dP = zeros(N^2, N^2, N^2)
    for l in 1:(N ^ 2)
        E = elementary(Float64, N, l)
        dP[l, :, :] = (Pof(V + h * E) - Pof(V - h * E)) / (2h)
    end
    return P, dP
end
const V3 = let A = randn(rng, 3, 3)
    A * A' + 2I                                          # SPD, generic, N = 3
end
let V = V3
    r_lp = jacobi_residual(fd_tensors((V, X, Y) -> tr(V * comm(X, Y)), V)...)
    r_pf = jacobi_residual(fd_tensors((V, X, Y) -> pushforward(sqrt(V), X, Y), V)...)
    r_cf = jacobi_residual(fd_tensors((V, X, Y) -> cform(sqrt(V), X, Y), V)...)
    @printf("   FD Jacobiator, N = 3:  Lie-Poisson %.1e   pushforward %.1e   (4/3)V^{3/2}-form %.1e\n",
        r_lp, r_pf, r_cf)
    # r_lp is the exactly-Poisson control that calibrates the 1e-6 floor below.  Asserted, not just
    # printed: if the finite-difference machinery were wrong, r_lp would be large and the floor
    # would mean nothing.
    check("Lie-Poisson: FD Jacobiator at the FD noise floor (calibrates the threshold)",
        r_lp < 1e-6, @sprintf("%.1e", r_lp))
    check("pushforward bracket: FD Jacobiator at the FD noise floor", r_pf < 1e-6, @sprintf("%.1e",
        r_pf))
    check("(4/3) tr(V^{3/2}[X,Y]): FD Jacobiator far above the floor  (control)",
        r_cf > 1e-4, @sprintf("%.1e", r_cf))
    # Casimirs of the pushforward: tr V^k
    P = first(fd_tensors((V, X, Y) -> pushforward(sqrt(V), X, Y), V))
    cas = maximum(norm(P * gradvec(k * V^(k - 1))) for k in 1:3)
    check("pushforward bracket: P ∇tr(V^k) == 0, k = 1..3  (FD floor)", cas < 1e-10, @sprintf("%.1e",
        cas))
end

# 6b. semiclassical difference on Zeitlin's su(N)
println()
@printf("   %-30s", "N")
for N in Ns
    @printf(" %8d", N)
end
println("     slope")
diffs = map(Ns) do N
    W = sqrt(Matrix(SYMBOLS[N]))
    B = WINDOWS[N]
    num = 0.0
    den = 0.0
    for X in B, Y in B

        p = pushforward(W, X, Y)
        q = cform(W, X, Y)
        num = max(num, abs(p - q))
        den = max(den, abs(q))
    end
    num / den
end
print_row("|pushforward − c-form|/|c-form|", Ns, diffs)
let sl = rate(Ns[(end - NTAIL + 1):end], diffs[(end - NTAIL + 1):end])
    check(
        "pushforward → (4/3) tr(V^{3/2}[X,Y]) as N → ∞ at rate N^{-2}  (slope in [-2.2, -1.8])",
        -2.2 <= sl <= -1.8, @sprintf("slope = %.2f", sl))
end

# ---------------------------------------------------------------------------------------------
header("7. spectral-kernel brackets  B_θ(X,Y) = Σ_ab θ(v_a,v_b) X_ab Y_ba:  Jacobiator and classification")
# ---------------------------------------------------------------------------------------------
println("   c-bracket: θ = c(v_a) − c(v_b);  Lie-Poisson: θ = v_a − v_b;  pushforward of Lie-Poisson under")
println("   V = ψ⁻¹(W): θ = (v_a − v_b)^2/(ψ(v_a) − ψ(v_b)).  With ω = θ/(v_a − v_b):")
println("   Jac = Σ_{abc} Ψ_abc X_ab (Y_bc Z_ca − Z_bc Y_ca),  Ψ_abc = ω_ab ω_bc (v_a − v_c) + cyc,")
println("   and (all ω ≠ 0)  Ψ ≡ 0  ⇔  (v_a − v_b)/ω_ab is a cocycle  ⇔  ω = 1/ψ^[1]  ⇔  pushforward.")

# 7a. exact: polynomial kernels θ(a,b) = Σ_kl t[k+1,l+1] a^k b^l give polynomial brackets
#     B_θ(X,Y) = Σ_kl t_kl tr(V^k X V^l Y), whose tensors are exact over Q.
struct Kernel{T}
    t::Matrix{T}
end
function (θ::Kernel)(a, b)
    sum(θ.t[k + 1, l + 1] * a^k * b^l
    for k in 0:(size(θ.t, 1) - 1), l in 0:(size(θ.t, 2) - 1))
end

function kernel_tensors(θ::Kernel, V::AbstractMatrix{T}) where {T}
    N = size(V, 1)
    kmax = size(θ.t, 1) - 1
    pw = [V^r for r in 0:kmax]                                           # pw[r+1] = V^r
    dpw(E, k) = k == 0 ? zeros(T, N, N) : sum(pw[r + 1] * E * pw[k - r] for r in 0:(k - 1))
    terms = [(k, l, θ.t[k + 1, l + 1])
             for k in 0:kmax, l in 0:kmax if !iszero(θ.t[k + 1, l + 1])]
    Xs = [transpose(elementary(T, N, l)) for l in 1:(N ^ 2)]               # gradient matrix of V_l
    B(X, Y) = sum(t * tr(pw[k + 1] * X * pw[l + 1] * Y) for (k, l, t) in terms)
    P = [B(Xs[i], Xs[j]) for i in 1:(N ^ 2), j in 1:(N ^ 2)]
    dP = zeros(T, N^2, N^2, N^2)
    for l in 1:(N ^ 2)
        E = elementary(T, N, l)
        dE = [dpw(E, k) for k in 0:kmax]
        dB(X, Y) = sum(t * (tr(dE[k + 1] * X * pw[l′ + 1] * Y) +
                        tr(pw[k + 1] * X * dE[l′ + 1] * Y))
        for (k, l′, t) in terms)
        for j in 1:(N ^ 2), i in 1:(N ^ 2)

            dP[l, i, j] = dB(Xs[i], Xs[j])
        end
    end
    return P, dP
end

# θ(a,b) as a coefficient matrix t[k+1, l+1] ↔ a^k b^l
const θ_lin = Kernel(Q[0 -1; 1 0])                    # a − b                    Lie-Poisson
const θ_sq = Kernel(Q[0 0 -1; 0 0 0; 1 0 0])           # a² − b²                  c = W²
const θ_cub = Kernel(Q[0 0 0; 0 0 -1; 0 1 0])          # a²b − ab² = ab(a−b)      pushforward, ψ = −1/v
const θ_pencil = Kernel(Q[0 -1 0; 1 0 -1; 0 1 0])      # (a−b)(1 + ab)            linear + cubic
const θ_ctrl = Kernel(Q[0 -1 0 -1; 1 0 0 0; 0 0 0 0; 1 0 0 0])   # a−b+a³−b³  control

let N = 3, V = randq(N)
    P, _ = kernel_tensors(θ_lin, V)
    check("N=3  kernel θ = a−b reproduces bracket_matrix(V) exactly", P ==
                                                                      bracket_matrix(V))
    P2, dP2 = kernel_tensors(θ_sq, V)
    P1, dP1 = tensors(Poly([Q(0), Q(0), Q(1)]), V)
    check("N=3  kernel θ = a²−b² reproduces the tensors of c = W² exactly", P2 == P1 &&
        dP2 == dP1)
end
for N in (3, 4)
    V = randq(N)
    for (label, θ, expect_zero) in (("a − b  (Lie-Poisson)", θ_lin, true),
        ("a² − b²  (c = W²)", θ_sq, false),
        ("ab(a − b)  (cubic; pushforward ψ = −1/v)", θ_cub, true),
        ("(a−b)(1 + ab)  (linear + cubic pencil)", θ_pencil, false),
        ("(a−b)(1 + a² + ab + b²)  (control)", θ_ctrl, false))
        r = jacobi_residual(kernel_tensors(θ, V)...)
        check(
            @sprintf("N=%d  θ = %-42s Jacobiator %s", N, label,
                expect_zero ? "== 0" : "!= 0"),
            expect_zero ? iszero(r) : !iszero(r), "residual = " * fmt(r))
    end
end
println("   The cubic bracket is  tr(Y [V, V X V]) = tr(V²XVY) − tr(VXV²Y): Lie-Poisson pushed forward under")
println("   V = −W⁻¹.  Linear and cubic are NOT compatible: ρ = (a−b)/(1+ab) = tan(arctan a − arctan b) is")
println("   not exact.  (Λ = a−b, Λ·ab are the only Poisson kernels with ω a symmetric polynomial of degree ≤ 2.)")

# the Ψ-formula, exactly, against the coordinate Jacobiator at a diagonal V
for N in (3, 4)
    v = distinct_spectrum(N)
    V = Matrix(Diagonal(v))
    X, Y, Z = randq(N), randq(N), randq(N)
    for (label, θ) in (("a² − b²", θ_sq), ("ab(a − b)", θ_cub),
        ("(a−b)(1 + ab)", θ_pencil), ("(a−b)(1 + a² + ab + b²)", θ_ctrl))
        ω = (a, b) -> θ(a, b) / (a - b)
        jc = jacobiator_coord(kernel_tensors(θ, V)..., X, Y, Z)
        jk = jacobiator_kernel(ω, v, X, Y, Z)
        check(@sprintf("N=%d  θ = %-16s coordinate Jacobiator == Ψ-formula", N, label),
            jc == jk,
            "value = " * fmt(jk))
    end
end

# 7b. the classification, pointwise and exactly: at a fixed spectrum, ω = 1/ψ^[1] for a random
#     potential ψ on the eigenvalues gives Ψ ≡ 0 on every triple; a random symmetric ω does not.
let N = 5
    v = distinct_spectrum(N)
    ψ = distinct_spectrum(N)                              # a potential with distinct values on the vertices
    idx = Dict(v[i] => i for i in 1:N)
    ω_pf = (a, b) -> (a - b) / (ψ[idx[a]] - ψ[idx[b]])
    triples = [(v[a], v[b], v[c]) for a in 1:N for b in (a + 1):N for c in (b + 1):N]
    check("N=5  ω = 1/ψ^[1], ψ random on the spectrum  ⇒  Ψ_abc == 0 on all 10 triples",
        all(iszero(Ψ(ω_pf, t...)) for t in triples))
    Ω = Q[rand(rng, 1:9) for _ in 1:N, _ in 1:N]
    Ω = Ω + transpose(Ω)
    ω_rand = (a, b) -> Ω[idx[a], idx[b]]
    check("N=5  ω random symmetric  ⇒  some Ψ_abc != 0  (control)",
        any(!iszero(Ψ(ω_rand, t...)) for t in triples))
    # and the c-bracket weight ω = c^[1] for c = v² is not of pushforward type: Ψ = (a−b)(b−c)(c−a);
    # the linear-cubic pencil ω = 1 + ab has Ψ = −(a−b)(b−c)(c−a), the negative (cf. the values in 7a)
    ω_sq = (a, b) -> a + b
    ω_pc = (a, b) -> 1 + a * b
    check("N=5  ω = a + b  (c = v²):  Ψ_abc == (a−b)(b−c)(c−a)  on all triples",
        all(Ψ(ω_sq, t...) == (t[1] - t[2]) * (t[2] - t[3]) * (t[3] - t[1])
        for t in triples))
    check("N=5  ω = 1 + ab  (pencil):  Ψ_abc == −(a−b)(b−c)(c−a)  on all triples",
        all(Ψ(ω_pc, t...) == -(t[1] - t[2]) * (t[2] - t[3]) * (t[3] - t[1])
        for t in triples))
end

# 7c. floating point, FD Jacobiator at N = 3 for non-polynomial kernels
# The FD tensor perturbs V by every elementary matrix, including non-symmetric ones, so the bracket
# has to be evaluated on general (non-Hermitian) V: V = U diag(v) U⁻¹, X̃ = U⁻¹ X U.  Wrapping V in
# Hermitian() would silently drop the lower-triangle perturbations and ruin the derivative.
function kernel_bracket(θ, V, X, Y)
    F = eigen(V)
    # A general eigenproblem may return a conjugate pair.  The spectra here are real and the
    # perturbations are O(h), so the imaginary parts are round-off -- but say so loudly rather than
    # dropping whatever turns up, which is how a broken perturbation would pass unnoticed.
    scaleV = maximum(abs, F.values)
    maximum(abs, imag.(F.values)) <= 1e-8 * max(scaleV, 1) ||
        error("kernel_bracket: eigenvalues are not real, max |imag| = " *
              string(maximum(abs, imag.(F.values))))
    v = real.(F.values)
    U = F.vectors
    Ui = inv(U)
    Xe = Ui * X * U
    Ye = Ui * Y * U
    s = zero(ComplexF64)
    n = length(v)
    for i in 1:n, j in 1:n

        i == j && continue
        s += θ(v[i], v[j]) * Xe[i, j] * Ye[j, i]
    end
    abs(imag(s)) <= 1e-8 * max(abs(real(s)), 1) ||
        error("kernel_bracket: the bracket is not real, imag = " * string(imag(s)))
    return real(s)
end
θ_arith(a, b) = (a - b) * (a + b)                                             # c = v²: 2·(a−b)·A(a,b)
θ_log(a, b) = (d = a - b; abs(d) < 1e-8 * abs(b) ? d * (a + b) : 2 * d^2 / log(a / b))  # 2·(a−b)·L(a,b)
θ_geo(a, b) = (a - b) * sqrt(a * b)                                           # (a−b)·G(a,b): not a pushforward
θ_cube(a, b) = (d = a - b; d^2 / (a^3 - b^3 + (d == 0)))                       # ψ = v³
θ_sqrt(a, b) = (a - b) * (sqrt(a) + sqrt(b))                                  # ψ = √v
println()
println("   c = v² has ω = A(a,b) (arithmetic mean); its exactly Poisson counterpart with the same continuum")
println("   limit has ω = L(a,b) = (a−b)/ln(a/b) (logarithmic mean): ψ = ½ ln v, i.e. V = exp(2W).")
let V = V3
    for (label, θ, expect_zero) in ((
        "(a−b)(a+b)   c = v², arithmetic mean", θ_arith, false),
        ("2(a−b)²/ln(a/b)   logarithmic mean, ψ = ½ln v", θ_log, true),
        ("(a−b)√(ab)   geometric mean", θ_geo, false),
        ("(a−b)²/(a³−b³)   ψ = v³", θ_cube, true),
        ("(a−b)(√a+√b)   ψ = √v", θ_sqrt, true))
        r = jacobi_residual(fd_tensors((V, X, Y) -> kernel_bracket(θ, V, X, Y), V)...)
        check(
            @sprintf("N=3  θ = %-46s FD Jacobiator %s", label,
                expect_zero ? "≈ 0" : "≫ floor"),
            expect_zero ? r < 1e-6 : r > 1e-4, @sprintf("%.1e", r))
    end
    P = first(fd_tensors((V, X, Y) -> kernel_bracket(θ_log, V, X, Y), V))
    cas = maximum(norm(P * gradvec(k * V^(k - 1))) for k in 1:3)
    check(
        "N=3  log-mean bracket: P ∇tr(V^k) == 0, k = 1..3  (same Casimirs as tr(V²[·,·]))",
        cas < 1e-10,
        @sprintf("%.1e", cas))
end

# 7d. semiclassics: the log-mean bracket vs tr(V²[X,Y]) on Zeitlin data, N → ∞
println()
@printf("   %-30s", "N")
for N in Ns
    @printf(" %8d", N)
end
println("     slope")
diffsL = map(Ns) do N
    V = Matrix(SYMBOLS[N])
    v, U = eigen(Hermitian(V))
    B = [U' * X * U for X in WINDOWS[N]]
    num = 0.0
    den = 0.0
    n = length(v)
    ΘA = [θ_arith(v[i], v[j]) for i in 1:n, j in 1:n]
    ΘL = [θ_log(v[i], v[j]) for i in 1:n, j in 1:n]
    for X in B, Y in B

        M = X .* transpose(Y)                             # M_ij = X_ij Y_ji
        a = sum(ΘA .* M)
        l = sum(ΘL .* M)
        num = max(num, abs(a - l))
        den = max(den, abs(a))
    end
    num / den
end
print_row("|log-mean − tr(V²[X,Y])|/|·|", Ns, diffsL)
let sl = rate(Ns[(end - NTAIL + 1):end], diffsL[(end - NTAIL + 1):end])
    check(
        "log-mean bracket → tr(V²[X,Y]) as N → ∞ at rate N^{-2}  (slope in [-2.2, -1.8])",
        -2.2 <= sl <= -1.8, @sprintf("slope = %.2f", sl))
end

summary("III2_matrix_structure_function.jl")
