#!/usr/bin/env julia
#
# Can regrouping the integrand restore the Jacobi identity of the second KdV bracket?
#
#     julia --project=scripts scripts/verify_kdv_jacobi_family.jl
#
# The second structure is 4u d_x + 2u_x - d_x^3 (in the textbook sign convention,
# -4u d_x - 2u_x - d_x^3). Its u-dependent part can be assembled in several ways that agree
# after integration by parts in the continuum but not discretely, giving a one-parameter
# family alpha T1 + beta T3. This asks whether any member of that family closes into a Lie
# algebra.
#
#   0. validation of the structure-constant routine on known controls;
#   1. the family has exactly one relation, T1 + T2 + T3 = 0;
#   2. antisymmetry forces alpha = 2 beta, and consistency then pins (4,2);
#   3. the Jacobiator splits by degree in u into a degree-2 Lie-algebra condition and a
#      degree-1 2-cocycle condition, both of order one and FLAT under refinement;
#   4. scanning beta/alpha over the whole family never brings either near zero.
#
# The index convention matches the package throughout: `C[m,i,j]` is c_ij^m, which is what
# `structure_constant_residual` and `jacobi_residual` already expect.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary

const TOL = 1e-9

sci(v) = "[" * join(["'" * (@sprintf("%.4f", x)) * "'" for x in v], ", ") * "]"

"Mass matrix, its inverse, the three integration-by-parts tensors and K0."
function build(n; p = 3)
    s = SplineSpace(n, p)
    M = Matrix(mass_matrix(s))
    Minv = inv(M)
    Φ0, Φ1, Φ2 = basis_values(s, 0), basis_values(s, 1), basis_values(s, 2)
    W = quadrature_weights(s)
    nb = nbasis(s)
    T1 = [sum(Φ0[m, q] * Φ0[k, q] * Φ1[l, q] * W[q] for q in eachindex(W))
          for m in 1:nb, k in 1:nb, l in 1:nb]
    T2 = [sum(Φ0[m, q] * Φ1[k, q] * Φ0[l, q] * W[q] for q in eachindex(W))
          for m in 1:nb, k in 1:nb, l in 1:nb]
    T3 = [sum(Φ1[m, q] * Φ0[k, q] * Φ0[l, q] * W[q] for q in eachindex(W))
          for m in 1:nb, k in 1:nb, l in 1:nb]
    # weights scale the quadrature index, i.e. the SECOND axis of Φ
    K0 = Φ1 * transpose(Φ2 .* transpose(W))
    return s, M, Minv, T1, T2, T3, K0
end

"C^m = Minv ( alpha T1 + beta T3 )[m] Minv."
function C_of(Minv, T1, T3, α, β)
    nb = size(Minv, 1)
    Tm = α .* T1 .+ β .* T3
    C = similar(Tm)
    for m in 1:nb
        C[m, :, :] = Minv * Tm[m, :, :] * Minv
    end
    C
end

rel_antisym3(C) = maximum(abs, C .+ permutedims(C, (1, 3, 2))) / maximum(abs, C)
nrm(f) = first(f) / last(f)

header("0. validation of the structure-constant routine")
# so(3) is a DEGENERATE positive control: it lies in the six-parameter Bianchi class A family
# c_ij^k = eps_ijl n^lk with n SYMMETRIC, every member of which is a Lie algebra, and
# rescaling one C^m keeps n symmetric and diagonal. It is NOT the case that antisymmetry
# alone forces Jacobi in three dimensions -- a general antisymmetric c has nine parameters
# against this family's six, and random ones fail. The Python prototype stated the broader
# claim; it is an erratum, and the narrower reason is the true one. Use se(3).
let Cso3 = Float64.(so3())
    Cbad = copy(Cso3)
    Cbad[1, :, :] *= 1.5     # rescale one generator, keeping C^m antisym
    r = nrm(structure_constant_residual(Cbad; normalised = false))
    check(
        "so(3) is degenerate: it still passes after a generator is rescaled off " *
        "the algebra, so it is NOT a usable positive control",
        r < TOL,
        @sprintf("residual %.2e", r))
end
let r = nrm(structure_constant_residual(Float64.(se3()); normalised = false))
    check("se(3), 6-dimensional, is a Lie algebra: residual 0", r < TOL,
        @sprintf("residual %.2e", r))
end
let r = nrm(structure_constant_residual(
        Float64.(random_antisymmetric_c(MersenneTwister(5), 5)); normalised = false))
    check("random antisymmetric structure constants in dim 5 are NOT: residual != 0",
        r > 0.1, @sprintf("residual %.2f", r))
end

const N = 24
s, M, Minv, T1, T2, T3, K0 = build(N)

header("1. the integration-by-parts family")
let sc = max(maximum(abs, T1), maximum(abs, T3))
    e = maximum(abs, T1 .+ T2 .+ T3) / sc
    check("there is exactly one relation, T1 + T2 + T3 = 0", e < TOL,
        @sprintf("rel. error %.2e", e))
    check("the transposition k <-> l exchanges T1 and T2",
        maximum(abs, T1 .- permutedims(T2, (1, 3, 2))) / sc < TOL)
    check("...and fixes T3", maximum(abs, T3 .- permutedims(T3, (1, 3, 2))) / sc < TOL)
end

header("2. antisymmetry and consistency leave a single member")
for (al, be, want) in ((4, 2, true), (2, 1, true), (6, 3, true),
    (4, 1, false), (4, 3, false), (1, 2, false))
    e = rel_antisym3(C_of(Minv, T1, T3, al, be))
    ok = e < TOL
    check("alpha=$al, beta=$be: antisymmetric = $ok (expected $want, i.e. alpha = 2 beta)",
        ok == want, @sprintf("rel. error %.2e", e))
end

# consistency: applying the assembled operator to the coefficients of a smooth w must
# reproduce int phi_k ( 4 u w' + 2 u' w ).
let X = quadrature_nodes(s), W = quadrature_weights(s), Φ0 = basis_values(s, 0)
    uf(x) = sin(x)
    duf(x) = cos(x)
    wf(x) = cos(2x) + 0.3
    dwf(x) = -2 * sin(2x)
    ud = project(s, uf)
    wd = project(s, wf)
    exact = Φ0 * (W .* (4 .* uf.(X) .* dwf.(X) .+ 2 .* duf.(X) .* wf.(X)))
    println("\n  consistency of alpha*T1 + beta*T3 with 4u d_x + 2u_x:")
    best = nothing
    for (al, be) in ((4, 2), (2, 1), (4, 1), (3, 2), (5, 2), (4, 3))
        Tm = al .* T1 .+ be .* T3
        got = [sum(ud[m] * Tm[m, k, l] * wd[l] for m in axes(Tm, 1), l in axes(Tm, 3))
               for k in axes(Tm, 2)]
        err = maximum(abs, got - exact) / maximum(abs, exact)
        @printf("      (alpha,beta) = (%d,%d):  rel. error %.3e\n", al, be, err)
        (best === nothing || err < best[2]) && (best = ((al, be), err))
    end
    check("only (alpha,beta) = (4,2) is consistent", best[1] == (4, 2),
        @sprintf("best was (%d, %d) at %.2e", best[1][1], best[1][2], best[2]))
end

header("3. the two obstructions, and their behaviour under refinement")
let b = kdv_bracket_2(s)
    P20, C = poisson_tensor(b)
    dofs = project(s, sin)
    Pm = P20 + sum(dofs[m] .* C[m, :, :] for m in axes(C, 1))
    # the split by degree in u must reproduce the full Jacobiator
    J1 = jacobiator(P20, C)
    A2 = zeros(size(C))
    for i in axes(C, 2), j in axes(C, 2), k in axes(C, 2)
        A2[i, j, k] = sum(dofs[n] * C[n, i, m] * C[m, j, k]
        for n in axes(C, 1), m in axes(C, 1))
    end
    J2 = A2 + permutedims(A2, (2, 3, 1)) + permutedims(A2, (3, 1, 2))
    Jd = jacobiator(Pm, C)
    e = maximum(abs, Jd - (J1 + J2)) / maximum(abs, Jd)
    check("the degree-1 + degree-2 split reproduces the full Jacobiator", e < 1e-10,
        @sprintf("rel. error %.2e", e))
end

println("\n  normalised residuals (they do not decrease):")
let d2s = Float64[], d1s = Float64[]
    for n in (12, 16, 24, 32)
        _, _, Mi, t1, t2, t3, k0 = build(n)
        Cn = C_of(Mi, t1, t3, 4, 2)
        r2 = nrm(structure_constant_residual(Cn; normalised = false))
        r1 = nrm(jacobi_residual(Mi * k0 * Mi, Cn; normalised = false))
        push!(d2s, r2)
        push!(d1s, r1)
        @printf("      N = %3d   Lie algebra %.4f   cocycle %.4f\n", n, r2, r1)
    end
    check("the Lie-algebra residual is of order one and flat under refinement",
        minimum(d2s) > 0.4 && (maximum(d2s) - minimum(d2s)) < 0.01, sci(d2s))
    check("the cocycle residual is of order one and flat under refinement",
        minimum(d1s) > 0.4 && (maximum(d1s) - minimum(d1s)) < 0.06, sci(d1s))
end

header("4. scanning the whole family never restores the Lie-algebra condition")
# A note on the numbers off the axis alpha = 2 beta. `common.py` carries TWO
# structure-constant residuals -- the exact one and `structure_constant_residual_float` --
# and they are NOT the same function. They agree exactly when C is antisymmetric in (i,j);
# off that locus one is the other evaluated at C transposed. Since C(1,t)^T = -C(1,1-t) in
# this family, the Python's scan is this one mirrored about t = 1/2. The package keeps the
# exact convention, which is what every Lie-Poisson script uses and what is verified against
# se(3) and so(N).
#
# Nothing here turns on the choice. Only alpha = 2 beta gives an antisymmetric C, so only
# there is the tensor a candidate set of structure constants at all; at that point both
# conventions give 0.4212, and the minimum over the family is 0.4184 either way.
let ratios = range(-1.0, 1.5; length = 51)
    vals = [nrm(structure_constant_residual(C_of(Minv, T1, T3, 1.0, t); normalised = false))
            for t in ratios]
    for t in (-0.5, 0.0, 0.5, 1.0, 1.5)
        i = argmin(abs.(ratios .- t))
        @printf("      beta/alpha = % .3f   residual %.4f%s\n", ratios[i], vals[i],
            abs(ratios[i] - 0.5) < 1e-9 ? "   <- the notes' choice" : "")
    end
    check("the residual never approaches zero anywhere in the family", minimum(vals) > 0.3,
        @sprintf("minimum %.4f at beta/alpha = %.3f",
            minimum(vals), ratios[argmin(vals)]))
end

summary("verify_kdv_jacobi_family.jl")
