#!/usr/bin/env julia
#
# Where the square-root transformation constantifies a Lie-Poisson bracket.
#
#     julia --project=scripts scripts/verify_burgers_leibniz.jl
#
# The version of Appendix B in the notes expands
#
#     [f F, g G] = f g [F,G] + f G [F,g] + g F [f,G] + F G [f,g]                (*)
#
# which presumes the Lie bracket is a derivation in each argument with respect to pointwise
# multiplication. No Lie bracket is: for vector fields [f X, Y] = f [X,Y] - (Y f) X. This
# script shows (*) is false for the one-dimensional transport bracket, while the identity
# actually needed,
#
#     [h F, h G] = h^2 [F, G] ,                                                (**)
#
# holds there exactly. Since the transformed bracket is
#
#     {A,B} = int Phi(ubar) [ h Fbar, h Gbar ] ,    h = 1/Phi'(ubar) ,
#
# (**) together with Phi/(Phi')^2 = const is necessary and sufficient for the transformed
# bracket to be independent of ubar, and the second condition forces Phi(ubar) = (a ubar + b)^2
# -- the square root, uniquely up to an affine reparametrisation.
#
# Finally it checks that (**) fails for vector fields in d > 1 and for the canonical (Vlasov)
# bracket, so Section 1 is specific to one-dimensional transport-type brackets.
#
# Needs SymPy, provisioned by CondaPkg through SymPyPythonCall.

using SymPyPythonCall

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary

@syms x::positive y::positive ubar::positive
@syms xv::real vv::real

header("1. The four-term expansion (*) is false for [F,G] = F G' - G F'")

@syms f() g() F() G()
F_, G_, ff, gg = F(x), G(x), f(x), g(x)

br1(a, b) = a * diff(b, x) - b * diff(a, x)

lhs = br1(ff * F_, gg * G_)
rhs_paper = ff * gg * br1(F_, G_) + ff * G_ * br1(F_, gg) +
            gg * F_ * br1(ff, G_) + F_ * G_ * br1(ff, gg)
d = simplify(lhs - rhs_paper)
check("the expansion used in Appendix B does NOT hold", d != 0, "difference = $d")

check("the correct identity is [f F, g G] = f g [F,G] + F G [f,g]",
    simplify(lhs - (ff * gg * br1(F_, G_) + F_ * G_ * br1(ff, gg))) == 0)

header("2. The identity that is actually needed, [h F, h G] = h^2 [F,G]")

@syms h()
hx = h(x)
check("holds for the 1D transport bracket [F,G] = F G' - G F'",
    simplify(br1(hx * F_, hx * G_) - hx^2 * br1(F_, G_)) == 0)

# vector fields in d = 2: [F,G]^i = F^j d_j G^i - G^j d_j F^i
@syms F1() F2() G1() G2() hv()
f1, f2, g1, g2, hh = F1(xv, y), F2(xv, y), G1(xv, y), G2(xv, y), hv(xv, y)

function br_vec(A, B)
    (
        A[1] * diff(B[1], xv) + A[2] * diff(B[1], y) -
        B[1] * diff(A[1], xv) - B[2] * diff(A[1], y),
        A[1] * diff(B[2], xv) + A[2] * diff(B[2], y) -
        B[1] * diff(A[2], xv) - B[2] * diff(A[2], y))
end

lhs_v = br_vec((hh * f1, hh * f2), (hh * g1, hh * g2))
rhs_v = br_vec((f1, f2), (g1, g2))
diff_v = [simplify(lhs_v[i] - hh^2 * rhs_v[i]) for i in 1:2]
check("FAILS for vector fields in d = 2", any(!=(0), diff_v))
# the discrepancy is exactly h (G^i F^j - F^i G^j) d_j h
disc = simplify(diff_v[1] -
                hh * ((g1 * f1 - f1 * g1) * diff(hh, xv) +
                 (g1 * f2 - f1 * g2) * diff(hh, y)))
check("   and the discrepancy is h (G^i F^j - F^i G^j) d_j h", simplify(disc) == 0)

# canonical / Vlasov bracket on (x, v)
@syms Fc() Gc() hcf()
fc, gc, hc = Fc(xv, vv), Gc(xv, vv), hcf(xv, vv)

br_can(a, b) = diff(a, xv) * diff(b, vv) - diff(a, vv) * diff(b, xv)

d_can = simplify(br_can(hc * fc, hc * gc) - hc^2 * br_can(fc, gc))
check("FAILS for the canonical (Vlasov) bracket", d_can != 0)
check("   and the discrepancy is h (F [h,G] - G [h,F])",
    simplify(d_can - hc * (fc * br_can(hc, gc) - gc * br_can(hc, fc))) == 0)

header("3. Phi / (Phi')^2 = const forces Phi(ubar) = (a ubar + b)^2")

@syms Phi() c::positive
sol = dsolve(Eq(Phi(ubar) / diff(Phi(ubar), ubar)^2, c), Phi(ubar))
sols = sol isa AbstractVector ? sol : [sol]
println("   general solution: [",
    join([string(simplify(s.rhs())) for s in sols], ", "), "]")
# `==` on a Sym returns a Julia Bool; `Int` of one does not convert, so compare rather
# than cast. Degree two, in ubar, is the whole claim.
ok = any(sols) do s
    expr = expand(simplify(s.rhs()))
    # fatou-ignore redundant-boolean
    expr.is_polynomial(ubar) == true && sympy.degree(expr, ubar) == 2
end
check("the solution is a quadratic in ubar, i.e. ubar = sqrt(u) up to an affine map", ok)

header("4. so(3) is not of the form sqrt(u_a) K_ab sqrt(u_b) with constant K")

@syms u1::positive u2::positive u3::positive
u = [u1, u2, u3]
# J_ab = eps_abc u_c; if J_ab = sqrt(u_a) K_ab sqrt(u_b) then K_ab = J_ab / sqrt(u_a u_b)
nonconst = Bool[]
for ((a, b), cc) in ((1, 2) => 3, (2, 3) => 1, (3, 1) => 2)
    K = simplify(u[cc] / sqrt(u[a] * u[b]))
    push!(nonconst, any(diff(K, s) != 0 for s in u))
end
check("every off-diagonal K_ab depends on u, so no constant K exists", all(nonconst))

summary("verify_burgers_leibniz.jl")
