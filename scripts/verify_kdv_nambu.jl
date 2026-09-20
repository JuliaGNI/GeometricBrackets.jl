#!/usr/bin/env julia
#
# Does KdV admit a fully anti-symmetric three-bracket?
#
#     julia --project=scripts scripts/verify_kdv_nambu.jl
#
#   1.  the no-go for (H1, H2): both gradients vanish at u = 0, so the bracket has no term of
#       first order in u while -u_xxx does. Shown symbolically and on a finite-dimensional
#       model; the Casimir and Galilean freedoms do not help;
#   2.  THE VANDERMONDE LEMMA. Anti-symmetry forces the symbol to be the Vandermonde times a
#       symmetric factor, so at (k,-k,0) it is k^3 times an EVEN polynomial: a local bracket
#       gives d_x^3, d_x^5, ... in the Casimir slot and never d_x;
#   3.  the repaired triple (H1, C0), with two explicit brackets that agree where the
#       consistency condition pins them;
#   4.  the discrete wedge tensor S_ijk = (P1_ij + P1_jk + P1_ki)/L on the spline basis --
#       and what it does not do: its flow IS the P1 flow, so it repackages Section 4;
#   5.  THE ONE GENUINELY NEW BRACKET, {u,H2,C0}. Its dispersive part is local -- half the
#       Wronskian -- and its nonlinear part provably is not. It carries a zero-mode anomaly:
#       the weight at the mean of u must be 3/2 times the weight at every other mode;
#   6.  THE FUNDAMENTAL IDENTITY is out of reach by theorem, not by effort. A Nambu-Poisson
#       tensor of order >= 3 satisfying it must be decomposable (Gautheron; Alekseevsky-Guha),
#       hence of rank three;
#   7.  NAMBU'S OWN CRITERION, which is neither of the above but LIOUVILLE'S THEOREM. A
#       constant anti-symmetric tensor gives a divergence-free flow for any pair of slot
#       functions, which covers the wedge tensor and nothing more. For {u,H2,C0}, where S
#       really does depend on u, the flow is divergence-free all the same -- and not for
#       that reason: the trace sum_i dS_ijk/duhat_i is a nonzero matrix whose CASIMIR
#       column alone vanishes, because contracting the field index against the first slot
#       is a Fourier trace, and it lands on triples where two slot modes coincide -- where
#       the Vandermonde vanishes, the symbol's denominator vanishes, and slot anti-symmetry
#       gives zero, all three independently of the weight.
#
# Needs SymPy.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random
using SymPyPythonCall

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary
include(joinpath(@__DIR__, "kdvtools.jl"));
using .KdVTools

const TOL = 1e-9
# the six permutations of three slots, with their signs
const SIG = ((1, 2, 3) => 1, (2, 3, 1) => 1, (3, 1, 2) => 1,
    (3, 2, 1) => -1, (2, 1, 3) => -1, (1, 3, 2) => -1)

# --------------------------------------------------------------------------
# spectral machinery: brackets as symbols tau(k,l,m) on the plane k+l+m = 0
#
#   int a f dx = 2 pi sum_{k+l=0} a_k f_l ,
#   T(a,b,c)   = 2 pi sum_{k+l+m=0} tau(k,l,m) a_k b_l c_m ,
#
# so that T(a,b,1) = int a L b with the symbol of L read off tau(k,-k,0).
# --------------------------------------------------------------------------

V(k, l, m) = (k - l) * (l - m) * (m - k)
e2(k, l, m) = k * l + l * m + m * k
e3(k, l, m) = k * l * m

"T(a,b,c); `tau(k,l,m)` if `u` is nothing, else `tau(j,k,l,m)` summed against u_j."
function spectral(tau, A, B, C; u = nothing)
    s = 0.0im
    if u === nothing
        for (k, x) in A, (l, y) in B

            m = -(k + l)
            haskey(C, m) && (s += tau(k, l, m) * x * y * C[m])
        end
    else
        for (j, w) in u, (k, x) in A, (l, y) in B
            m = -(j + k + l)
            haskey(C, m) && (s += tau(j, k, l, m) * w * x * y * C[m])
        end
    end
    2π * s
end

"max over the six permutations of | T(sigma) - sgn(sigma) T(id) |."
function antisymmetry_defect(T, A, B, C)
    base = T(A, B, C)
    args = (A, B, C)
    maximum(abs(T(args[p[1]], args[p[2]], args[p[3]]) - sg * base) for (p, sg) in SIG)
end

"Fourier coefficients of a random real trigonometric polynomial."
function rand_real(rng, K; zero = true)
    d = zero ? Dict(0 => complex(randn(rng), 0.0)) : Dict{Int, ComplexF64}()
    for k in 1:K
        c = complex(randn(rng), randn(rng))
        d[k], d[-k] = c, conj(c)
    end
    d
end

function times(p, q)
    r = Dict{Int, ComplexF64}()
    for (k, x) in p, (l, y) in q

        r[k + l] = get(r, k + l, 0.0im) + x * y
    end
    r
end

"int a f dx."
function pair(A, f)
    2π * sum(ComplexF64[A[k] * f[-k] for k in keys(A) if haskey(f, -k)]; init = 0.0im)
end

"Fourier coefficients of 6 u u_x - u_xxx = 3(u^2)_x - u_xxx."
function kdv_rhs(u; nonlinear = true, dispersive = true)
    f = Dict{Int, ComplexF64}()
    if nonlinear
        for (k, v) in times(u, u)
            f[k] = get(f, k, 0.0im) + 3im * k * v
        end
    end
    if dispersive
        for (k, v) in u
            f[k] = get(f, k, 0.0im) + 1im * k^3 * v      # -u_xxx = i k^3 u_k
        end
    end
    f
end

"Random totally anti-symmetric S[i,j,k]."
function rand_antisym3(n, rng; integer = true)
    S = zeros(n, n, n)
    for i in 1:n, j in (i + 1):n, k in (j + 1):n
        v = integer ? float(rand(rng, -4:4)) : randn(rng)
        idx = (i, j, k)
        for (p, sg) in SIG
            S[idx[p[1]], idx[p[2]], idx[p[3]]] = sg * v
        end
    end
    S
end

# --------------------------------------------------------------------------
header("1. equation (69) cannot hold: both gradients vanish at u = 0")

@syms eps S0 S1 S2 Qs Rs Ps
# S(eps u) = S0 + eps S1 + ... for any S regular at u = 0;
# dH1/du (eps u) = 3 eps^2 Q - eps R  and  dH2/du (eps u) = eps P.
flow = expand((S0 + eps * S1 + eps^2 * S2) * (3 * eps^2 * Qs - eps * Rs) * (eps * Ps))
target_sym = 6 * eps^2 * Qs * Ps - eps * Rs               # schematic 6uu_x - u_xxx
check("the bracket has no term of first order in u, for any regular S",
    flow.coeff(eps, 1) == 0, "O(eps) coefficient = $(flow.coeff(eps, 1))")
check("the KdV right-hand side does: -u_xxx is linear",
    expand(target_sym).coeff(eps, 1) != 0)
check(
    "so no S regular at u = 0 can satisfy (69); only a component of u-degree " *
    "-1 could, which is singular at the trivial solution",
    simplify(expand(flow - target_sym).coeff(eps, 1)) != 0)

# the same statement on an exact finite-dimensional model, for a generic u-dependent totally
# anti-symmetric S
let n = 5, rng = MersenneTwister(0)
    Sc = rand_antisym3(n, rng)                                # constant part
    Sl = [rand_antisym3(n, rng) for _ in 1:n]                 # u-linear part
    uu = [Sym("u$(i-1)") for i in 1:n]
    A1 = [Sym(rand(rng, -3:3)) for _ in 1:n, _ in 1:n]
    A2 = [Sym(rand(rng, -3:3)) for _ in 1:n, _ in 1:n]
    g1 = A1 * (eps .* uu) + [sum((eps * uu[a]) * (eps * uu[b]) for a in 1:n) for b in 1:n]
    g2 = A2 * (eps .* uu)                                     # both vanish at u = 0
    Sfull(i, j, k) = Sym(Int(Sc[i, j, k])) +
                     sum(eps * uu[m] * Int(Sl[m][i, j, k]) for m in 1:n)
    comp = expand(sum(Sfull(1, j, k) * g1[j] * g2[k] for j in 1:n, k in 1:n))
    check(
        "finite-dimensional model: a generic u-dependent anti-symmetric S gives a " *
        "flow with no term linear in u either",
        expand(comp).coeff(eps, 1) == 0)
end

header("1b. the Casimir and Galilean freedoms do not help (u-independent S)")

let
    @syms ks ls al be la
    ms = -(ks + ls)
    # degree 1:  M(l) [ lambda (l^2 + beta) - alpha ] = i l^3
    M(x) = im * x^3 / (la * (x^2 + be) - al)
    # degree 2 (symmetrised over the two u factors):
    #   tau(k,l,m) (l^2 - m^2) + 6 lambda M(-k) = -6 i k ,  tau = V s
    num = -6 * im * ks - 6 * la * M(-ks)
    s_req = simplify(num / (V(ks, ls, ms) * (ls^2 - ms^2)))
    s_swap = s_req.subs(Dict(ks => ls, ls => ks); simultaneous = true)
    check(
        "the s the consistency conditions demand is not symmetric under k <-> l, " *
        "so it is not of the form s(e2,e3) and no anti-symmetric tau exists",
        simplify(s_req - s_swap) != 0)
    check(
        "and the degenerate branch lambda beta = alpha gives s = 0, which fails " *
        "the degree-2 condition outright",
        simplify(num.subs(al, la * be)) == 0 && simplify(-6 * im * ks) != 0)
end

# --------------------------------------------------------------------------
header("2. anti-symmetry forces the Vandermonde, hence at least three derivatives")

let
    @syms kk ll mm
    DEG = 6
    mons = [(a, b, c) for a in 0:DEG for b in 0:DEG for c in 0:DEG if a + b + c ≤ DEG]
    cs = [Sym("c$(i-1)") for i in eachindex(mons)]
    gen = sum(cs[i] * kk^mons[i][1] * ll^mons[i][2] * mm^mons[i][3]
    for i in eachindex(mons))
    vars = (kk, ll, mm)
    alt = expand(sum(sg * gen.subs(
                         Dict(vars[1] => vars[p[1]], vars[2] => vars[p[2]],
                             vars[3] => vars[p[3]]);
                         simultaneous = true)
    for (p, sg) in SIG) / 6)
    red = expand(alt.subs(Dict(ll => -kk, mm => Sym(0)); simultaneous = true))
    quot, rem_ = sympy.div(sympy.Poly(red, kk), sympy.Poly(kk^3, kk))
    check(
        "every anti-symmetric polynomial symbol of degree <= $DEG reduces at " *
        "(k,-k,0) to k^3 times a polynomial",
        simplify(rem_.as_expr()) == 0)
    check(
        "and that polynomial is even, so a local bracket gives d_x^3, d_x^5, ... " *
        "in the Casimir slot and never d_x",
        simplify(quot.as_expr() - quot.as_expr().subs(kk, -kk)) == 0)
    check(
        "the Vandermonde itself is anti-symmetric and cubic, so it exists -- the " *
        "rank-one mode lattice does carry an alternating TRILINEAR form",
        all(simplify(V(vars[p[1]], vars[p[2]], vars[p[3]]) - sg * V(kk, ll, mm)) == 0
        for (p, sg) in SIG))
end

# --------------------------------------------------------------------------
header("3. the third slot must be the Casimir C0 = int u dx")

function tau_smooth(a, b, c)
    (a * a + b * b + c * c) == 0 ? 0.0im :
    -1im * V(a, b, c) / (a * a + b * b + c * c)
end
function tau_wedge(a, b, c)
    (c == 0 ? 1im * b : 0.0im) + (a == 0 ? 1im * c : 0.0im) +
    (b == 0 ? 1im * a : 0.0im)
end

const rng3 = MersenneTwister(1)
const uu3 = rand_real(rng3, 3)
const one3 = Dict(0 => 1.0 + 0.0im)
const dH1sp = let d = Dict{Int, ComplexF64}(q => 3v for (q, v) in times(uu3, uu3))
    for (q, v) in uu3
        d[q] = get(d, q, 0.0im) + q * q * v                     # -u_xx  ->  +k^2 u_k
    end
    d
end
const target3 = kdv_rhs(uu3)

for (name, tau) in (("smooth  tau = -i V / (k^2+l^2+m^2)", tau_smooth),
    ("wedge   P1 ^ dC0", tau_wedge))
    Tb(A, B, C) = spectral(tau, A, B, C)
    a, b, c = rand_real(rng3, 4), rand_real(rng3, 4), rand_real(rng3, 4)
    d = antisymmetry_defect(Tb, a, b, c)
    check("$name: totally anti-symmetric", d < 1e-10,
        @sprintf("max defect over S3 = %.2e", d))
    a = rand_real(rng3, 6)
    err = abs(Tb(a, dH1sp, one3) - pair(a, target3))
    check("$name: {a,H1,C0} = int a (6uu_x - u_xxx)", err < 1e-9,
        @sprintf("error %.2e against |rhs| = %.2f", err, abs(pair(a, target3))))
    bb = rand_real(rng3, 6)
    err = abs(Tb(a, bb, one3) - pair(a, Dict(q => 1im * q * v for (q, v) in bb)))
    check(
        "$name: its Casimir-slot operator is exactly d_x -- which the lemma " *
        "of 2. forbids a local bracket",
        err < 1e-9,
        @sprintf("error %.2e", err))
end

# both are the same member of the family tau = V s(e2,e3), pinned on e3 = 0
let same = maximum(abs(tau_smooth(a, b, -a - b) - tau_wedge(a, b, -a - b))
    for a in -4:4, b in -4:4 if (b == 0 || a == 0 || a + b == 0))
    check(
        "the two agree on the slices e3 = 0, where the consistency condition pins " *
        "s = i/(2 e2), and differ only where they are free",
        same < 1e-12,
        @sprintf("max difference on e3 = 0 is %.2e", same))
end

# --------------------------------------------------------------------------
header("4. the discrete wedge tensor  S_ijk = ( P1_ij + P1_jk + P1_ki ) / L")

for uniform in (true, false)
    tag = uniform ? "uniform" : "arbitrary"
    s, _ = setup(16; p = 3, uniform)
    P1 = poisson_matrix(kdv_bracket_1(s), zeros(nbasis(s)))
    g = mass_grad(s)
    check("$tag: partition of unity, sum_i int phi_i = L", abs(sum(g) - L) < 1e-12,
        @sprintf("defect %.2e", abs(sum(g) - L)))
    check("$tag: C0 is an EXACT Casimir of P1", maximum(abs, P1 * g) < 1e-12,
        @sprintf("|P1 g| = %.2e", maximum(abs, P1 * g)))

    nb = nbasis(s)
    S = [(P1[i, j] + P1[j, k] + P1[k, i]) / L for i in 1:nb, j in 1:nb, k in 1:nb]
    d = max(maximum(abs, S + permutedims(S, (2, 1, 3))),
        maximum(abs, S + permutedims(S, (1, 3, 2))),
        maximum(abs, S + permutedims(S, (3, 2, 1))))
    check("$tag: S is totally anti-symmetric", d < 1e-12, @sprintf("max defect %.2e", d))
    Sg = [sum(S[i, j, k] * g[k] for k in 1:nb) for i in 1:nb, j in 1:nb]
    e = maximum(abs, Sg - P1)
    check("$tag: S(.,.,dC0) = P1", e < 1e-12, @sprintf("max defect %.2e", e))

    u_h = project(s, U0)
    d1, d2 = grad_H1(s, u_h), grad_H2(s, u_h)
    flow = [sum(S[i, j, k] * d1[j] * g[k] for j in 1:nb, k in 1:nb) for i in 1:nb]
    check(
        "$tag: the Nambu flow IS the P1 flow -- the construction repackages " *
        "Section 4",
        maximum(abs, flow - P1 * d1) < 1e-12,
        @sprintf("max difference %.2e", maximum(abs, flow - P1 * d1)))
    check("$tag: H1 conserved exactly, by anti-symmetry", abs(dot(d1, flow)) < 1e-11,
        @sprintf("dH1/dt = %.2e", abs(dot(d1, flow))))
    check("$tag: mass conserved exactly, by anti-symmetry", abs(dot(g, flow)) < 1e-11,
        @sprintf("dC0/dt = %.2e", abs(dot(g, flow))))
    @printf("      but H2 is NOT in a slot: dH2/dt = %.2e  (%s)\n", abs(dot(d2, flow)),
        uniform ? "exact on a uniform mesh" : "the Section 4 gap survives")
end

# --------------------------------------------------------------------------
header("5. {u,H2,C0}: local dispersion, provably nonlocal nonlinearity")

# 5a. the dispersive part is half the Wronskian
let
    @syms kk ll mm
    W3 = Sym[1 1 1; im*kk im*ll im*mm; -kk^2 -ll^2 -mm^2]
    check("the Wronskian density det[[a,b,c],[a',b',c'],[a'',b'',c'']] has symbol -i V",
        simplify(expand(det(W3) - (-im * V(kk, ll, mm)))) == 0)
end
tau_wronski(a, b, c) = -0.5im * V(a, b, c)
const asp = rand_real(rng3, 6)
let err = abs(spectral(tau_wronski, asp, uu3, one3) -
              pair(asp, Dict(q => 1im * q^3 * v for (q, v) in uu3)))
    check("half the Wronskian gives T(a,u,1) = -int a u_xxx exactly, and it is local",
        err < 1e-9, @sprintf("error %.2e", err))
end

# 5b. no polynomial (local) u-linear part can supply 6 u u_x
let
    @syms jj kk ll
    for DEG2 in (5, 7)
        mons = [(x, y, z) for x in 0:DEG2 for y in 0:DEG2
                for z in 0:DEG2
                if x + y + z ≤ DEG2]
        co = [Sym("d$(i-1)") for i in eachindex(mons)]
        # P(j,k,l) = sigma(j; k, l, -(j+k+l)) as a general polynomial
        Pg = sum(co[i] * jj^mons[i][1] * kk^mons[i][2] * ll^mons[i][3]
        for i in eachindex(mons))
        eqs = Sym[]
        for expr in (Pg + Pg.subs(Dict(kk => ll, ll => kk); simultaneous = true),
            Pg + Pg.subs(Dict(ll => -(jj + kk + ll)); simultaneous = true),
            Pg.subs(Dict(ll => -(jj + kk)); simultaneous = true) +
            Pg.subs(Dict(jj => -(jj + kk), ll => jj); simultaneous = true) +
            6 * im * kk)
            pol = sympy.Poly(expand(expr), jj, kk, ll)
            append!(eqs, collect(pol.coeffs()))
        end
        sol = solve(eqs, co; dict = true)
        check(
            "no polynomial symbol of degree <= $DEG2 is anti-symmetric AND consistent " *
            "with 6 u u_x",
            isempty(sol),
            "$(length(eqs)) equations, solutions: $(length(sol))")
    end
end

# 5c. an explicit nonlocal solution, with its zero-mode anomaly
function make_sigma(g0, gn)
    (j, a, b, c) -> begin
        E2, E3 = e2(a, b, c), e3(a, b, c)
        den = E2 * E2 + E3 * E3
        den == 0 ? 0.0im : (j == 0 ? g0 : gn) * V(a, b, c) * E2 / den
    end
end

function h2_bracket(g0, gn)
    (A, B, C) -> spectral(tau_wronski, A, B, C) +
                 spectral(make_sigma(g0, gn), A, B, C; u = uu3)
end

let err = abs(h2_bracket(3im, 2im)(asp, uu3, one3) - pair(asp, target3))
    check("with g(0) = 3i and g(j) = 2i: {a,H2,C0} = int a (6uu_x - u_xxx)", err < 1e-9,
        @sprintf("error %.2e against |rhs| = %.2f", err, abs(pair(asp, target3))))
end
for (g0, gn, why) in ((2im, 2im, "no zero-mode anomaly"), (
    3im, 3im, "anomalous everywhere"))
    e = abs(h2_bracket(g0, gn)(asp, uu3, one3) - pair(asp, target3))
    check("a uniform weight fails ($why), so the anomaly is not a convention", e > 1.0,
        @sprintf("error %.2e", e))
end
let Tb = h2_bracket(3im, 2im)
    d = antisymmetry_defect(Tb, rand_real(rng3, 4), rand_real(rng3, 4), rand_real(rng3, 4))
    check("the full u-dependent bracket is totally anti-symmetric", d < 1e-9,
        @sprintf("max defect over S3 = %.2e", d))
end
println("      the anomaly is forced: at l = 0 the b- and c-slots coincide, so anti-symmetry")
println("      kills that term and the mean of u must be carried by the weight instead")

# --------------------------------------------------------------------------
header("6. the fundamental identity forces rank three")

"""
    fi_residual(S, rng; trials)

max |FI| over random quadratic test functions, and its natural scale.

    FI:  {f1,f2,{g1,g2,g3}} = {{f1,f2,g1},g2,g3} + {g1,{f1,f2,g2},g3} + {g1,g2,{f1,f2,g3}}

A quadratic f has gradient A x + b and constant Hessian A.
"""
function fi_residual(S, rng; trials = 4)
    n = size(S, 1)
    worst, scale = 0.0, 0.0
    for _ in 1:trials
        F = [(A = randn(rng, n, n); (A + transpose(A)) / 2) for _ in 1:5]
        v = [randn(rng, n) for _ in 1:5]
        x = randn(rng, n)
        gr = [F[i] * x + v[i] for i in 1:5]
        # gradient of the function {p,q,r}
        grad3(p, q, r) = [sum(S[a, b, c] * (F[p][a, d] * gr[q][b] * gr[r][c] +
                               gr[p][a] * F[q][b, d] * gr[r][c] +
                               gr[p][a] * gr[q][b] * F[r][c, d])
                          for a in 1:n, b in 1:n, c in 1:n) for d in 1:n]
        tri(x1, x2, x3) = sum(S[a, b, c] * x1[a] * x2[b] * x3[c]
        for a in 1:n, b in 1:n, c in 1:n)
        f1, f2, g1, g2, g3 = 1, 2, 3, 4, 5
        lhs = tri(gr[f1], gr[f2], grad3(g1, g2, g3))
        rhs = tri(grad3(f1, f2, g1), gr[g2], gr[g3]) +
              tri(gr[g1], grad3(f1, f2, g2), gr[g3]) +
              tri(gr[g1], gr[g2], grad3(f1, f2, g3))
        worst = max(worst, abs(lhs - rhs))
        scale = max(scale, abs(lhs), abs(rhs))
    end
    worst, scale
end

"Dimension of the smallest U with S in Lambda^3 U; 3 iff S is decomposable."
support_dim(S; tol = 1e-9) = rank(reshape(S, size(S, 1), :); atol = tol)

let rng = MersenneTwister(4)
    dec = zeros(5, 5, 5)
    for (p, sg) in SIG
        dec[p...] = float(sg)                               # e1 ∧ e2 ∧ e3 inside R^5
    end
    r, sc = fi_residual(dec, rng)
    check("control: a decomposable e1 ^ e2 ^ e3 in dimension 5 satisfies the FI",
        r / max(sc, 1.0) < 1e-9, @sprintf("normalised residual %.2e", r / max(sc, 1.0)))
    check("control: its support has dimension 3", support_dim(dec) == 3)

    r4, s4 = fi_residual(rand_antisym3(4, rng; integer = false), rng)
    check(
        "TRAP: a RANDOM anti-symmetric 3-tensor in dimension 4 also satisfies the " *
        "FI -- dim Lambda^3 R^4 = 4 and every 3-vector there is decomposable, so " *
        "dimension <= 4 is useless as a test case",
        r4 / max(s4, 1.0) < 1e-9, @sprintf("normalised residual %.2e", r4 / max(s4, 1.0)))

    S5 = rand_antisym3(5, rng; integer = false)
    r5, s5 = fi_residual(S5, rng)
    check("negative control: a random anti-symmetric 3-tensor in dimension 5 does NOT",
        r5 / max(s5, 1.0) > 1e-3, @sprintf("normalised residual %.2e", r5 / max(s5, 1.0)))
    check("and its support is all of R^5, so it is not decomposable", support_dim(S5) == 5,
        "support dimension $(support_dim(S5))")

    s, _ = setup(8; p = 3)
    P1 = poisson_matrix(kdv_bracket_1(s), zeros(nbasis(s)))
    nb = nbasis(s)
    Sw = [(P1[i, j] + P1[j, k] + P1[k, i]) / L for i in 1:nb, j in 1:nb, k in 1:nb]
    rw, sw = fi_residual(Sw ./ maximum(abs, Sw), rng)
    check("the discrete wedge tensor violates the FI, as it must",
        rw / max(sw, 1e-30) > 1e-3, @sprintf("normalised residual %.2f",
            rw / max(sw, 1e-30)))
    for nn in (7, 8, 9, 12)
        sq, _ = setup(nn; p = 3)
        Pq1 = poisson_matrix(kdv_bracket_1(sq), zeros(nbasis(sq)))
        Sq = [(Pq1[i, j] + Pq1[j, k] + Pq1[k, i]) / L for i in 1:nn, j in 1:nn, k in 1:nn]
        rk, sd = rank(Pq1), support_dim(Sq; tol = 1e-8)
        check("N = $nn: the wedge tensor has support $sd, not 3", sd == nn - (nn + 1) % 2,
            "rank P1 = $rk (N-2 for even N, the zero and Nyquist modes lying in its " *
            "kernel; N-1 for odd N), support of S = $sd")
    end
end
println("      a Nambu-Poisson tensor of order >= 3 must be decomposable (Gautheron;")
println("      Alekseevsky-Guha), hence of rank three: an N-dof discretisation satisfying")
println("      the FI would move on three-dimensional leaves.  The FI is not a hard goal")
println("      but an impossible one, and anti-symmetry alone is the usable structure.")

# --------------------------------------------------------------------------
header("7. Nambu's own criterion: is the flow divergence-free in coefficient space?")

# The guiding principle of Nambu (1973) is neither the Jacobi identity nor the fundamental
# identity but LIOUVILLE'S THEOREM: his eq. (3) is div(grad H x grad G) = 0.  The discrete
# analogue asks whether
#
#     udot_i = sum_jk S_ijk a_j b_k ,    a = dF/duhat ,  b = dG/duhat
#
# has vanishing divergence sum_i d(udot_i)/d(uhat_i) on the coefficient space.  Expanding,
#
#     div = sum_ijk [ (dS_ijk/duhat_i) a_j b_k  +  S_ijk F_ij b_k  +  S_ijk a_j G_ik ] ,
#
# and the last two terms die: S is anti-symmetric in (i,j) while the Hessian F_ij is
# symmetric, and anti-symmetric in (i,k) while G_ik is symmetric.  So the whole question is
# the first term, and for a CONSTANT S there is nothing left.

"""
sum_i d(udot_i)/d(uhat_i) by central differences on the flow itself.

Every flow in this section is a polynomial of degree at most two in `x`, so the central
difference carries no truncation error and `h` trades against cancellation alone: the residual
is one ULP of `|flow|` divided by `2h`, and a *larger* `h` is strictly better.  That is why
`1e-6` sits three orders of magnitude below the tightest threshold used here.  A genuinely
nonlinear flow reinstates the usual `h^2` term and the default stops being generous.
"""
function divergence_fd(flow, x; h = 1e-6)
    n = length(x)
    s = 0.0
    for i in 1:n
        e = zeros(n)
        e[i] = h
        s += (flow(x + e)[i] - flow(x - e)[i]) / (2h)
    end
    s
end

"random T[m,i,j,k], anti-symmetric in the three slots (i,j,k) and free in m"
function rand_slotantisym4(n, rng)
    T = zeros(n, n, n, n)
    for m in 1:n
        T[m, :, :, :] = rand_antisym3(n, rng; integer = false)
    end
    T
end

# 7a. the structural lemma: a constant totally anti-symmetric S is divergence-free for ANY
#     pair of functions, whatever they are
let rng = MersenneTwister(7), n = 6
    S = rand_antisym3(n, rng; integer = false)
    sym(A) = (A + transpose(A)) / 2
    Fh, Gh = sym(randn(rng, n, n)), sym(randn(rng, n, n))
    fv, gv, x = randn(rng, n), randn(rng, n), randn(rng, n)
    grF(y) = Fh * y + fv
    grG(y) = Gh * y + gv
    function flow(y)
        a, b = grF(y), grG(y)
        [sum(S[i, j, k] * a[j] * b[k] for j in 1:n, k in 1:n) for i in 1:n]
    end

    ax, bx = grF(x), grG(x)
    hess = sum(S[i, j, k] * (Fh[i, j] * bx[k] + ax[j] * Gh[i, k])
    for i in 1:n, j in 1:n, k in 1:n)
    check("the two Hessian terms cancel against anti-symmetry, identically",
        abs(hess) < 1e-10, @sprintf("|sum S (F_ij b_k + a_j G_ik)| = %.2e", abs(hess)))
    d = divergence_fd(flow, x)
    sc = maximum(abs, flow(x))
    check("so a CONSTANT anti-symmetric S gives a divergence-free flow, for any F and G",
        abs(d) / max(sc, 1.0) < 1e-8, @sprintf("div = %.2e against |flow| = %.2e", d, sc))
end

# 7b. the controls.  Anti-symmetry alone is NOT enough once S depends on the field, and one
#     of the two obvious controls cannot fail.
let rng = MersenneTwister(8), n = 6
    T = rand_slotantisym4(n, rng)
    S(y) = [sum(T[m, i, j, k] * y[m] for m in 1:n) for i in 1:n, j in 1:n, k in 1:n]
    a, b = randn(rng, n), randn(rng, n)
    x = randn(rng, n)
    function flow(y)
        Sy = S(y)
        [sum(Sy[i, j, k] * a[j] * b[k] for j in 1:n, k in 1:n) for i in 1:n]
    end
    d, sc = divergence_fd(flow, x), maximum(abs, flow(x))
    check("negative control: a FIELD-DEPENDENT anti-symmetric S is not divergence-free",
        abs(d) / sc > 1e-3, @sprintf("div = %.3f against |flow| = %.3f", d, sc))

    # the trap, in the spirit of the dimension-four warning in section 6
    T4 = zeros(n, n, n, n)
    for m in 1:n, i in 1:n, j in 1:n, k in 1:n
        # anti-symmetrise T over ALL FOUR indices
        T4[m, i, j, k] = (T[m, i, j, k] - T[i, m, j, k] - T[j, i, m, k] - T[k, i, j, m]) / 4
    end
    tr4 = maximum(abs, [sum(T4[i, i, j, k] for i in 1:n) for j in 1:n, k in 1:n])
    check(
        "TRAP: if the field index is anti-symmetric against the first slot too, the " *
        "trace sum_i dS_ijk/duhat_i vanishes identically and the control cannot fail",
        tr4 < 1e-10, @sprintf("max |sum_i T4[i,i,j,k]| = %.2e over a tensor of scale %.2e",
            tr4, maximum(abs, T4)))
end

# 7c. the discrete wedge tensor.  P1 does not depend on the field, so S is constant and 7a
#     applies: this is an instance of the lemma, not independent evidence.
for uniform in (true, false)
    tag = uniform ? "uniform" : "arbitrary"
    s, _ = setup(16; p = 3, uniform)
    P1 = poisson_matrix(kdv_bracket_1(s), zeros(nbasis(s)))
    nb = nbasis(s)
    g = mass_grad(s)
    S = [(P1[i, j] + P1[j, k] + P1[k, i]) / L for i in 1:nb, j in 1:nb, k in 1:nb]
    function flow(y)
        d1 = grad_H1(s, y)
        [sum(S[i, j, k] * d1[j] * g[k] for j in 1:nb, k in 1:nb) for i in 1:nb]
    end
    u_h = project(s, U0)
    d, sc = divergence_fd(flow, u_h), maximum(abs, flow(u_h))
    check("$tag: the wedge flow is divergence-free -- Nambu's Liouville criterion holds",
        abs(d) / max(sc, 1.0) < 1e-6, @sprintf("div = %.2e against |flow| = %.2e", d, sc))
end
println("      but that is 7a, not a fact about Nambu brackets: P1 is a constant matrix, so")
println("      the wedge tensor is a constant tensor and nothing else could have happened")

# 7d. {u,H2,C0}, where S genuinely depends on u.  Built on an ORTHONORMAL real trigonometric
#     basis, so the Gram matrix is the identity and dF/duhat_j is the coefficient vector of
#     delta F / delta u -- no mass matrix anywhere.  delta H2 / delta u = u, so a = uhat.
function real_basis(K)
    B = [Dict(0 => complex(1 / sqrt(2π)))]
    for k in 1:K
        c = 1 / (2 * sqrt(π))
        push!(B, Dict(k => complex(c), -k => complex(c)))            # cos kx / sqrt(pi)
        push!(B, Dict(k => complex(0, -c), -k => complex(0, c)))     # sin kx / sqrt(pi)
    end
    B
end
function tohat(f, B)
    [real(2π * sum(ComplexF64[get(f, -k, 0.0im) * v for (k, v) in ψ]; init = 0.0im))
     for ψ in B]
end
function tofield(uhat, B)
    d = Dict{Int, ComplexF64}()
    for (ψ, c) in zip(B, uhat), (k, v) in ψ

        d[k] = get(d, k, 0.0im) + c * v
    end
    d
end

"S0[i,j,k] and dS_ijk/duhat_m for the (H2,C0) bracket in the basis B"
function h2_tensors(B, g0, gn)
    n = length(B)
    sg = make_sigma(g0, gn)
    S0 = [real(spectral(tau_wronski, B[i], B[j], B[k])) for i in 1:n, j in 1:n, k in 1:n]
    dS = [real(spectral(sg, B[i], B[j], B[k]; u = B[m]))
          for m in 1:n, i in 1:n, j in 1:n, k in 1:n]
    S0, dS
end

let K = 3, B = real_basis(K), rng = MersenneTwister(9)
    n = length(B)
    S0, dS = h2_tensors(B, 3im, 2im)
    uhat = randn(rng, n)
    S(y) = S0 .+ [sum(dS[m, i, j, k] * y[m] for m in 1:n) for i in 1:n, j in 1:n, k in 1:n]
    gC0 = [i == 1 ? sqrt(2π) : 0.0 for i in 1:n]    # dC0/duhat: the constant mode only
    function flow(y)
        Sy = S(y)
        [sum(Sy[i, j, k] * y[j] * gC0[k] for j in 1:n, k in 1:n) for i in 1:n]
    end

    # the construction is validated by reproducing what section 5 already established
    fu, rhs = flow(uhat), tohat(kdv_rhs(tofield(uhat, B)), B)
    e = maximum(abs, fu - rhs)
    check("the tensor in the real basis reproduces the KdV right-hand side",
        e < 1e-9, @sprintf("max error %.2e against |rhs| = %.2f", e, maximum(abs, rhs)))
    Su = S(uhat)
    dd = maximum([maximum(abs, Su + permutedims(Su, (2, 1, 3))),
        maximum(abs, Su + permutedims(Su, (1, 3, 2))),
        maximum(abs, Su + permutedims(Su, (3, 2, 1)))])
    sc = maximum(abs, Su)
    check("and is totally anti-symmetric there", dd < 1e-10 && sc > 1.0,
        @sprintf("max defect %.2e at scale %.2f", dd, sc))

    d = divergence_fd(flow, uhat)
    check("{u,H2,C0} is divergence-free: Nambu's criterion holds for the new bracket too",
        abs(d) / maximum(abs, fu) < 1e-8,
        @sprintf("div = %.2e against |flow| = %.2f", d, maximum(abs, fu)))

    # and it is NOT 7a: here S really does depend on the field
    tr = [sum(dS[i, i, j, k] for i in 1:n) for j in 1:n, k in 1:n]
    check("this is not 7a: the trace sum_i dS_ijk/duhat_i is a NONZERO matrix",
        maximum(abs, tr) > 1.0,
        @sprintf("max |trace| = %.2f at tensor scale %.2f", maximum(abs, tr),
            maximum(abs, dS)))
    check("what vanishes is its CASIMIR column alone, the third slot being the constant",
        maximum(abs, tr[:, 1]) < 1e-10 && maximum(abs, tr[:, 2:end]) > 1.0,
        @sprintf("Casimir column %.2e, every other column up to %.2f",
            maximum(abs, tr[:, 1]), maximum(abs, tr[:, 2:end])))
    for k in 2:4
        dv = sum(dS[i, i, j, k] * uhat[j] for i in 1:n, j in 1:n)
        check("negative control: with psi_$k in the third slot instead, div is order one",
            abs(dv) > 1e-2, @sprintf("div = %+.4f", dv))
    end
end
for K in (2, 4, 5)
    B = real_basis(K)
    _, dS = h2_tensors(B, 3im, 2im)
    n = length(B)
    tr = [sum(dS[i, i, j, k] for i in 1:n) for j in 1:n, k in 1:n]
    check("K = $K: the Casimir column of the trace still vanishes exactly",
        maximum(abs, tr[:, 1]) < 1e-10 && maximum(abs, tr[:, 2:end]) > 1.0,
        @sprintf("Casimir column %.2e, others up to %.2f", maximum(abs, tr[:, 1]),
            maximum(abs, tr[:, 2:end])))
end
# the weight sweep.  Its second conjunct is a real check; its first cannot fail, and 7e says
# why -- the weight is not among the quantities the vanishing depends on.
let B = real_basis(3), n = length(B)
    for (a, b, why) in ((2im, 2im, "uniform 2i"), (3im, 3im, "uniform 3i"))
        _, dS = h2_tensors(B, a, b)
        tr = [sum(dS[i, i, j, k] for i in 1:n) for j in 1:n, k in 1:n]
        check(
            "and it does not depend on the zero-mode anomaly ($why is inconsistent " *
            "with KdV, and still gives a vanishing Casimir column)",
            maximum(abs, tr[:, 1]) < 1e-10 && maximum(abs, tr[:, 2:end]) > 1.0,
            @sprintf("Casimir column %.2e, others up to %.2f", maximum(abs, tr[:, 1]),
                maximum(abs, tr[:, 2:end])))
    end
end

# 7e. the mechanism, which IS identifiable, and is a Fourier trace landing on the degenerate
#     set where two slot modes coincide.  Writing dS[m,i,j,k] out over modes,
#
#         dS[m,i,j,k] = 2 pi sum_{p+a+b+c = 0} sigma(p,a,b,c) (B_m)_p (B_i)_a (B_j)_b (B_k)_c ,
#
#   (1)  the real basis is closed under conjugation, so sum_i (B_i)_p (B_i)_a = delta_{p,-a}/2pi
#        and contracting the field index against the FIRST SLOT forces p = -a;
#   (2)  the mode constraint p + a + b + c = 0 then leaves b + c = 0;
#   (3)  the Casimir column is c = 0, hence b = 0.  The surviving triples are (a,0,0), with the
#        second and third slot modes coincident -- and there the VANDERMONDE NUMERATOR vanishes,
#        V(a,0,0) = 0, at the same time as e2 = e3 = 0.  sigma is 0/0 and make_sigma's guard
#        resolves it to zero.
#
# So the vanishing does not rest on that guard being the right convention.  b = c = 0 also puts
# the constant basis function in BOTH remaining slots, and total anti-symmetry in the slots kills
# S_{i,1,1} on its own; three independent reasons agree, which is why no regularisation of the
# symbol off the e3 = 0 slice can disturb the result.
#
# The weight g appears in none of the three steps, which is why the sweep above could not have
# failed either, and which predicts something stronger than the sweep tests.
let K = 3, B = real_basis(K), ks = (-K):K
    d = maximum(abs(sum(get(ψ, p, 0.0im) * get(ψ, a, 0.0im) for ψ in B) -
                    (p == -a ? 1 / (2π) : 0.0)) for p in ks, a in ks)
    check(
        "(1) the real basis is closed under conjugation: sum_i (B_i)_p (B_i)_a = " *
        "delta_{p,-a} / 2pi",
        d < 1e-14,
        @sprintf("max defect %.2e", d))

    sg = make_sigma(3im, 2im)
    cas = [(a, b, c) for a in ks, b in ks, c in ks if b + c == 0 && c == 0]
    worst = maximum(abs(sg(-a, a, b, c)) for (a, b, c) in cas)
    check(
        "(3) so every term surviving the contraction in the Casimir column has sigma = 0",
        worst == 0.0, @sprintf("%d triples, max |sigma| = %.1e", length(cas), worst))
    # steps (1) and (2) pin b + c but leave a free, so a = 0 belongs to this set as much as
    # any other a, and it is where the floor sits: e2(0,1,-1) = -1 and e3(0,1,-1) = 0.
    # On b = -c the denominator is c^4 (1 + a^2), so the floor is 1 for every K >= 1 and the
    # equality below is not an artefact of K = 3.
    off = minimum(e2(a, b, c)^2 + e3(a, b, c)^2
    for a in ks, b in ks, c in ks if b + c == 0 && c != 0)
    check("while off that column the same denominator is bounded away from zero, by 1",
        off == 1, @sprintf("min |e2^2 + e3^2| = %.1f", off))

    # and it is not the guard's convention that does it: two further reasons agree
    vdm = maximum(abs(V(a, b, c)) for (a, b, c) in cas)
    check(
        "the Vandermonde NUMERATOR vanishes on every one of those triples too, two slot " *
        "modes being coincident, so sigma is 0/0 and any regularisation gives zero",
        vdm == 0.0,
        @sprintf("max |V(a,b,c)| = %.1e over the same %d triples", vdm, length(cas)))
    _, dS = h2_tensors(B, 3im, 2im)
    check(
        "and total anti-symmetry in the slots gives the same zero independently: b = c = 0 " *
        "puts the constant basis function in BOTH remaining slots",
        maximum(abs, dS[:, :, 1, 1]) < 1e-14,
        @sprintf("max |dS[:,:,1,1]| = %.1e at a tensor scale of %.2f",
            maximum(abs, dS[:, :, 1, 1]), maximum(abs, dS)))
end

# the prediction: ANY weight gives a vanishing Casimir column, not merely the two uniform ones.
# The weights here are imaginary because a purely real g sends the whole trace to zero -- the
# quantity behind `real` is imaginary -- which would test nothing at all.
let B = real_basis(3), n = length(B)
    for (g0, gn, why) in ((0.0im, 1.0im, "0 / i"), (2.5im, -7.3im, "2.5i / -7.3i"),
        (-7.3 + 0.5im, 11.1 - 4im, "-7.3+0.5i / 11.1-4i"))
        _, dS = h2_tensors(B, g0, gn)
        tr = [sum(dS[i, i, j, k] for i in 1:n) for j in 1:n, k in 1:n]
        check("the prediction: weight $why solves nothing, and the column vanishes anyway",
            maximum(abs, tr[:, 1]) < 1e-12 && maximum(abs, tr[:, 2:end]) > 1.0,
            @sprintf("Casimir column %.2e, others up to %.2f", maximum(abs, tr[:, 1]),
                maximum(abs, tr[:, 2:end])))
    end
end

# the control for step (1).  Complex exponentials are orthonormal but NOT closed under
# conjugation: there (B_i)_p (B_i)_a is supported on p = a rather than p = -a, so the step
# fails -- and with it the conclusion.  This is a different contraction over complex
# coordinates, not a claim that a divergence depends on the basis; it does not.
let K = 3, Bc = [Dict(k => complex(1 / sqrt(2π))) for k in (-K):K], n = 2K + 1
    sg = make_sigma(3im, 2im)
    dS = [spectral(sg, Bc[i], Bc[j], Bc[k]; u = Bc[m])
          for m in 1:n, i in 1:n, j in 1:n, k in 1:n]
    tr = [abs(sum(dS[i, i, j, k] for i in 1:n)) for j in 1:n, k in 1:n]
    check(
        "control: where (1) fails the constant-mode column does NOT vanish, so the " *
        "closure of the real basis under conjugation is load-bearing",
        maximum(abs, tr[:, K + 1]) > 1e-6,
        @sprintf("column %.3e against a matrix maximum of %.3e",
            maximum(abs, tr[:, K + 1]), maximum(abs, tr)))
end
println("      so Liouville is a property of the PAIR of slot functions here, not of the")
println("      bracket: it is the whole one-function family of section 5 that has a")
println("      vanishing Casimir column -- and the mechanism is a Fourier trace meeting a")
println("      vanishing denominator, neither of which ever looks at the weight")

summary("verify_kdv_nambu.jl")
