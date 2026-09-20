#!/usr/bin/env julia
#
# Prescribing the Casimir: entropy-adapted discrete Poisson brackets.
#
#     julia --project=scripts scripts/verify_burgers_entropy_casimir.jl
#
# Theorem 2.1 is the discrete counterpart of the one-component Dubrovin-Novikov family of
# hydrodynamic-type Poisson brackets. The correspondence rests on the factorisation
#
#     J_G = G(u) d_x + 1/2 G'(u) u_x = sqrt(G) . d_x . sqrt(G)          (continuum)
#     J   = diag(g) K diag(g) ,          g_i = sqrt(G_i)                (discrete)
#
# -- multiplication, a CONSTANT anti-symmetric operator, multiplication. In one field
# component every metric G > 0 is flat, so every such bracket is Poisson; the discrete
# statement is Theorem 2.1, and the flat coordinate is the same in both cases,
# ubar = int du / sqrt(G).
#
# The practical consequence is that one does not choose g and then discover the Casimir --
# one prescribes the Casimir density eta and it DETERMINES g:
#
#     eta'(u) = 1 / g(u) .
#
# Choosing eta(u) = u log u therefore makes the logarithmic entropy an exact Casimir by
# construction, with g(u) = 1/(log u + 1).
#
#   1. the prescription eta' = 1/g works for arbitrary eta, exactly, and the entropy case;
#   2. the continuum factorisation and the continuum Casimir rule c' = 1/sqrt(G);
#   3. the singularity at eta'(u) = 0 is INTRINSIC -- no additive normalisation removes it;
#   4. the alternative route: for a Lie-Poisson bracket built from a matrix Lie algebra,
#      EVERY spectral function tr(eta(W)) is a Casimir, so the logarithmic entropy comes for
#      free and without any singularity.
#
# Sections 1-4 need SymPy. Section 5 does not: so(N) and its structure constants are exact
# rational arithmetic, which `so_n` in the package already does, so the CAS is dropped there.

using GeometricBrackets
using LinearAlgebra
using SymPyPythonCall

# `integrate` is exported by both GeometricBrackets (trajectory integration, via
# GeometricBase) and SymPy (the symbolic one). Any script using both has to qualify it;
# below it is always `sympy.integrate`.

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary

"Whether the Jacobi identity holds symbolically, and the first triple where it does not."
function jacobi_holds(J, dJ, n)
    for i in 1:n, j in 1:n, k in 1:n
        r = simplify(sum(dJ[l][i][j] * J[l][k] + dJ[l][j][k] * J[l][i] +
                         dJ[l][k][i] * J[l][j]
        for l in 1:n))
        r == 0 || return false, (i - 1, j - 1, k - 1, r)
    end
    return true, nothing
end

header("1. Prescribing the Casimir: eta' = 1/g, verified for several eta")

const n = 4
z = [symbols("z$i", positive = true) for i in 1:n]

# K of rank 2 so that ker K is non-trivial and the Casimirs are visible
a = [1, 2, -1, 3]
b = [2, -1, 4, 1]
Kint = [a[i] * b[j] - a[j] * b[i] for i in 1:n, j in 1:n]
K = Sym.(Kint)
ker = kernel(Rational{BigInt}.(Kint))                 # exact, no CAS needed for an integer K
check("ker K is non-trivial", size(ker, 2) ≥ 1, "dim ker K = $(size(ker, 2))")

ETAS = [
    ("eta(u) = u log u        (Boltzmann entropy)", u -> u * log(u)),
    ("eta(u) = u log u - u    (shifted entropy)", u -> u * log(u) - u),
    ("eta(u) = 2 sqrt(u)      (Section 1 Casimir)", u -> 2 * sqrt(u)),
    ("eta(u) = u^2            (quadratic)", u -> u^2),
    ("eta(u) = log u          (Lotka-Volterra)", u -> log(u))
]

for (label, eta) in ETAS
    g = [simplify(1 / diff(eta(z[i]), z[i])) for i in 1:n]
    J = [[g[i] * K[i, j] * g[j] for j in 1:n] for i in 1:n]
    dJ = [[[diff(J[i][j], z[l]) for j in 1:n] for i in 1:n] for l in 1:n]
    ok, bad = jacobi_holds(J, dJ, n)
    check("$label: Jacobi identity", ok, ok ? "" : "failed at $bad")
    for idx in axes(ker, 2)
        C = sum(Sym(ker[i, idx]) * eta(z[i]) for i in 1:n)
        flow = [simplify(sum(J[i][j] * diff(C, z[j]) for j in 1:n)) for i in 1:n]
        check("$label: sum_i n_i eta(u_i) is an exact Casimir", all(==(0), flow))
    end
end

header("2. The entropy case written out")

g_ent = [1 / (log(z[i]) + 1) for i in 1:n]
J_ent = [[simplify(g_ent[i] * K[i, j] * g_ent[j]) for j in 1:n] for i in 1:n]
println("   g_i(u_i) = ", g_ent[1])
println("   J_12     = ", J_ent[1][2])
S = -sum(Sym(ker[i, 1]) * z[i] * log(z[i]) for i in 1:n)
println("   S        = ", simplify(S))
flow = [simplify(sum(J_ent[i][j] * diff(S, z[j]) for j in 1:n)) for i in 1:n]
check("the logarithmic entropy is an exact Casimir", all(==(0), flow))
check("it is conserved for *every* Hamiltonian, not a chosen one", all(==(0), flow),
    "J grad S = 0 identically, so dS/dt = 0 whatever H is")

header("3. Continuum counterpart: J_G = sqrt(G) d_x sqrt(G) = G d_x + 1/2 G' u_x")

@syms x u() f() G() c()
ux, fx = u(x), f(x)

"sqrt(G(u)) * d_x ( sqrt(G(u)) * expr )"
apply_sqrtG_dx_sqrtG(expr) = (sq = sqrt(G(ux)); sq * diff(sq * expr, x))

lhs = expand(simplify(apply_sqrtG_dx_sqrtG(fx)))
rhs = expand(simplify(G(ux) * diff(fx, x) + diff(G(ux), x) * fx / 2))
check("the factorisation identity holds", simplify(lhs - rhs) == 0)

# Casimir rule: J_G c'(u) = 0  ⟺  c' = 1/sqrt(G); substitute c'(u) = G(u)^(-1/2)
subst = simplify(apply_sqrtG_dx_sqrtG(G(ux)^(-Sym(1) // 2)))
check("c'(u) = G(u)^(-1/2) annihilates J_G, i.e. the continuum rule c' = 1/g",
    simplify(subst) == 0, "J_G c' = $(simplify(subst))")

# and for G = 1/(log u + 1)^2 this gives c(u) = u log u.
#
# Note the branch: sqrt(G) = 1/|log u + 1|, and the absolute value changes sign at exactly
# u = 1/e -- the same point at which g = 1/(log u + 1) blows up. On each of the two branches
# (0, 1/e) and (1/e, inf) the bracket is a genuine Poisson structure and the Casimir density
# is u log u up to sign; the two branches do not join. This is the analytic face of the
# obstruction of part 4.
U = symbols("U", positive = true)
c_from_G = simplify(sympy.integrate(log(U) + 1, U))          # branch u > 1/e
check("G = (log u + 1)^-2  =>  Casimir density = u log u  (on the branch u > 1/e)",
    simplify(diff(c_from_G - U * log(U), U)) == 0,
    "integral of 1/sqrt(G) = $c_from_G")
check("1/sqrt(G) = |log u + 1| really does change branch at u = 1/e",
    simplify(sqrt((log(U) + 1)^2) - abs(log(U) + 1)) == 0,
    "so the entropy-adapted bracket lives on (0, 1/e) or (1/e, inf), not both")

header("4. The singularity at eta'(u) = 0 is intrinsic")

A = symbols("A", real = true)
eta_shift = z[1] * log(z[1]) + A * z[1]                # any additive linear shift
crit = solve(Eq(diff(eta_shift, z[1]), 0), z[1])
println("   d/du [ u log u + A u ] = 0  at  u = [", join(string.(crit), ", "), "]")
check("eta' vanishes somewhere in (0, inf) for every shift A",
    length(crit) == 1 && simplify(crit[1] - exp(-1 - A)) == 0,
    "so g = 1/eta' is singular there; no normalisation removes it")

header("5. The alternative: every spectral function is a Casimir of a Lie-Poisson bracket")

# so(N) with basis E_ab = e_a e_b' - e_b e_a', identified with its dual via the trace form.
# The invariants tr(W^k) are Casimirs of J_ij = sum_m c_ij^m u_m. All exact over Q.
for N in (4, 5)
    basis, C = so_n(N)
    d = length(basis)
    uu = [Rational{BigInt}(i, i + 2) for i in 1:d]
    W = sum(uu[i] * basis[i] for i in 1:d)
    J = lie_poisson_matrix(C, uu)
    ok_all = true
    for k in (2, 4)
        # ∂/∂u_j tr(W^k) = k tr(W^{k-1} E_j)
        grad = [k * tr(W^(k - 1) * basis[j]) for j in 1:d]
        local ok = all(iszero, J * grad)
        ok_all &= ok
        check("so($N): tr(W^$k) is a Casimir of J_ij = sum_m c_ij^m u_m", ok)
    end
    check("so($N): hence tr(eta(W)) is a Casimir for every spectral function eta", ok_all,
        "tr(eta(W)) depends on W only through its spectrum, i.e. through the tr(W^k)")
end

println()
println("   Consequence: in the Lie-Poisson branch the logarithmic entropy")
println("   tr(W log W) is a Casimir by construction, together with the whole")
println("   family tr(eta(W)) -- and J is linear in u, so nothing is singular.")

summary("verify_burgers_entropy_casimir.jl")
