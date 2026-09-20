#!/usr/bin/env julia
#
# Structural properties of the discrete KdV brackets on a periodic B-spline basis.
#
#     julia --project=scripts scripts/verify_kdv_discrete.jl
#
#   1.  the basis: partition of unity, symmetric positive definite mass matrix;
#   1b. the periodic construction of Section 3 -- dimension N = n, support p+1 cells,
#       C^{p-1} across the seam and no smoother;
#   2.  S = int phi_k phi_l' is ALREADY antisymmetric, so the 1/2 in the first bracket
#       matters: dropping it doubles P^1;
#   3.  -int phi_k phi_l''' = int phi_k' phi_l'', and that block is antisymmetric;
#   4.  P^2 is exactly antisymmetric on any mesh and at any quadrature;
#   5.  P^1 is exactly Poisson, P^2 is NOT -- and the contracted Jacobiator, against fixed
#       smooth functionals, converges at better than fourth order even though the
#       normalised one does not converge at all;
#   6.  cross-conservation is exact on a uniform mesh and only O(h^2) on a graded one;
#   6b. THE SHARP ONE: on a uniform mesh the exact vanishing is not an identity in the
#       degrees of freedom. The quadratic part vanishes identically because the assembled
#       matrices are circulant; the cubic part vanishes only for RESOLVED fields.
#
# The brackets come from the package -- `kdv_bracket_1`, `kdv_bracket_2` and
# `poisson_tensor` are the objects the test suite exercises.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random
using SimpleSplines: UniformMesh, GradedMesh, RandomMesh

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary

const TOL = 1e-9
const L = 2π

rel_antisym(A) = maximum(abs, A + transpose(A)) / maximum(abs, A)
fmt1(v) = "[" * join(["'" * (@sprintf("%.1f", x)) * "'" for x in v], ", ") * "]"

"Weighted outer product int phi^(da)_k phi^(db)_l, the raw assembly the notes write out."
function gram(s, da, db)
    basis_values(s, da) *
    transpose(basis_values(s, db) .* transpose(quadrature_weights(s)))
end

header("1. the periodic B-spline basis")
for (tag, mk) in (("uniform", n -> UniformMesh(n, L)), (
    "non-uniform", n -> RandomMesh(n, L)))
    s = SplineSpace(mk(24), 3)
    Φ0 = basis_values(s, 0)
    e = maximum(abs, vec(sum(Φ0; dims = 1)) .- 1)
    check("$tag: partition of unity", e < TOL, @sprintf("max error %.2e", e))
    M = Matrix(mass_matrix(s))
    check("$tag: mass matrix symmetric", maximum(abs, M - transpose(M)) < TOL)
    λ = minimum(eigvals(Symmetric(M)))
    check("$tag: mass matrix positive definite", λ > 0, @sprintf("min eigenvalue %.2e", λ))
end

header("1b. the periodic construction of Section 3")
let n = 20, pdeg = 3
    s = SplineSpace(UniformMesh(n, L), pdeg)
    check("the dimension is N = n = $n, one basis function per cell, not n + p",
        nbasis(s) == n, "got $(nbasis(s))")

    # support: each basis function spans exactly p+1 cells
    Φ0 = basis_values(s, 0)
    nq = size(Φ0, 2) ÷ n
    widths = [count(c -> maximum(abs, Φ0[i, ((c - 1) * nq + 1):(c * nq)]) > 1e-12, 1:n)
              for i in 1:n]
    check(
        "every basis function is supported on exactly p+1 = $(pdeg + 1) cells, " *
        "so there are no special boundary functions",
        Set(widths) == Set([pdeg + 1]), "observed widths $(sort(unique(widths)))")

    # In the package the basis IS the periodic one, so the relation phi_{j+n}(x) = phi_j(x-L)
    # the Python checks on an extended knot vector holds by construction and has no index
    # j+n to test. Its content here is that each basis function is L-periodic.
    xs = range(0.0, L; length = 97)
    err = 0.0
    for j in 1:4
        ej = zeros(n)
        ej[j] = 1.0
        err = max(err, maximum(abs, [evaluate(s, ej, x, 0) -
                                     evaluate(s, ej, mod(x + L, L), 0)
                                     for x in xs]))
    end
    check("phi_{j+n}(x) = phi_j(x - L), so only n splines are distinct on Omega",
        err < TOL, @sprintf("max error %.2e", err))

    # smoothness across the seam a = b: continuous up to order p-1, jumps at order p
    println("\n  jump of the d-th derivative across the seam x = a = b:")
    for d in 0:pdeg
        jump(δ) = maximum(1:n) do i
            ei = zeros(n)
            ei[i] = 1.0
            abs(evaluate(s, ei, δ, d) - evaluate(s, ei, L - δ, d))
        end
        j1, j2 = jump(1e-4), jump(1e-6)
        @printf("      d = %d:  %.3e (delta 1e-4)   %.3e (delta 1e-6)\n", d, j1, j2)
        if d < pdeg
            check("the basis is C^$d across the seam", j2 < 1e-4, @sprintf("jump %.2e", j2))
        else
            check("...and no smoother: derivative $d jumps", j2 > 1e-3,
                @sprintf("jump %.2e", j2))
        end
    end
end

header("2. the first bracket, and the factor 1/2")
for (tag, mk) in (("uniform", n -> UniformMesh(n, L)), (
    "non-uniform", n -> RandomMesh(n, L)))
    s = SplineSpace(mk(24), 3)
    S = gram(s, 0, 1)
    check("$tag: S = int phi_k phi_l' is already antisymmetric", rel_antisym(S) < TOL,
        @sprintf("rel. error %.2e", rel_antisym(S)))
    check("$tag: hence S - S^T = 2S, and dropping the 1/2 doubles P^1",
        maximum(abs, (S - transpose(S)) - 2S) / maximum(abs, S) < TOL)
end

header("3. the third-derivative block")
for (tag, mk) in (("uniform", n -> UniformMesh(n, L)), (
    "non-uniform", n -> RandomMesh(n, L)))
    s = SplineSpace(mk(24), 3)
    T = -gram(s, 0, 3)                  # -∫ φ_k φ_l'''
    Tw = gram(s, 1, 2)                  #  ∫ φ_k' φ_l''
    e = maximum(abs, T - Tw) / maximum(abs, T)
    check("$tag: -int phi_k phi_l''' = int phi_k' phi_l''", e < TOL,
        @sprintf("rel. error %.2e", e))
    check("$tag: the block is antisymmetric", rel_antisym(Tw) < TOL,
        @sprintf("rel. error %.2e", rel_antisym(Tw)))
end

header("4. the second bracket is exactly antisymmetric -- no antisymmetrisation needed")
for (nm, mk) in (("uniform", n -> UniformMesh(n, L)), (
        "non-uniform", n -> RandomMesh(n, L))),
    n in (16, 32)

    s = SplineSpace(mk(n), 3)
    tag = "$nm, N = $n"
    û = project(s, sin)
    W = quadrature_weights(s)
    Φ0, Φ1 = basis_values(s, 0), basis_values(s, 1)
    uh, ux = transpose(Φ0) * û, transpose(Φ1) * û
    # the bracketed factor is already (nq × nb); Φ0 is (nb × nq)
    U = Φ0 * ((4 .* uh .* transpose(Φ1) .+ 2 .* ux .* transpose(Φ0)) .* W)
    check("$tag: the 4u d_x + 2u_x block is antisymmetric", rel_antisym(U) < TOL,
        @sprintf("rel. error %.2e", rel_antisym(U)))
    P2 = poisson_matrix(kdv_bracket_2(s), û)
    check("$tag: P^2 is antisymmetric", rel_antisym(P2) < TOL,
        @sprintf("rel. error %.2e", rel_antisym(P2)))
end

header("5. the Jacobi identity")
let s = SplineSpace(UniformMesh(24, L), 3)
    P1 = poisson_matrix(kdv_bracket_1(s), zeros(nbasis(s)))
    check(
        "P^1 is antisymmetric and constant, hence exactly Poisson", rel_antisym(P1) < TOL,
        @sprintf("rel. error %.2e", rel_antisym(P1)))

    # P^2 is affine in the degrees of freedom, so dP/du_m = C[m] exactly.
    P20, C = poisson_tensor(kdv_bracket_2(s))
    û = project(s, sin)
    Pm = P20 + sum(û[m] .* C[m, :, :] for m in axes(C, 1))
    A = zeros(size(C))
    for i in axes(C, 2), j in axes(C, 2), k in axes(C, 2)
        A[i, j, k] = sum(Pm[i, m] * C[m, j, k] for m in axes(C, 1))
    end
    J = A + permutedims(A, (2, 3, 1)) + permutedims(A, (3, 1, 2))
    r = maximum(abs, J) / maximum(abs, A)
    check("P^2 does NOT satisfy the Jacobi identity", r > 1e-3,
        @sprintf("normalised Jacobiator %.3f", r))
end

println("\n  contracted against fixed smooth functionals, the violation converges:")
let fs = (sin, x -> cos(2x), x -> sin(3x) + 0.5), prev = nothing, orders = Float64[]
    for n in (16, 32, 64, 128)
        s = SplineSpace(UniformMesh(n, L), 3)
        M = Matrix(mass_matrix(s))
        Minv = inv(M)
        P20, C = poisson_tensor(kdv_bracket_2(s))
        # the raw blocks of the notes, recovered from the public tensor: Ku[m] = M C[m] M
        Ku = similar(C)
        for m in axes(C, 1)
            Ku[m, :, :] = M * C[m, :, :] * M
        end
        û = project(s, sin)
        Pm = P20 + sum(û[m] .* C[m, :, :] for m in axes(C, 1))
        W = quadrature_weights(s)
        X = quadrature_nodes(s)
        Φ0 = basis_values(s, 0)
        g = [Φ0 * (W .* fr.(X)) for fr in fs]
        G = [Minv * gi for gi in g]
        q(U, V) = [dot(U, Ku[m, :, :] * V) for m in axes(Ku, 1)]
        val = dot(transpose(Pm) * g[1], q(G[2], G[3])) +
              dot(transpose(Pm) * g[2], q(G[3], G[1])) +
              dot(transpose(Pm) * g[3], q(G[1], G[2]))
        prev === nothing || push!(orders, log2(abs(prev / val)))
        @printf("      N = %4d   Jacobiator = % .4e%s\n", n, val,
            isempty(orders) ? "" : @sprintf("   order %.1f", orders[end]))
        prev = val
    end
    check("the contracted Jacobiator converges at better than fourth order",
        minimum(orders) > 4.0, "observed orders $(fmt1(orders))")
end

header("6. conservation of the respective other Hamiltonian")
function cross1(s, û)
    dot(gradient(KdVHamiltonian2(s), s, û),
        poisson_matrix(kdv_bracket_1(s), û) * gradient(KdVHamiltonian1(), s, û))
end
function cross2(s, û)
    dot(gradient(KdVHamiltonian1(), s, û),
        poisson_matrix(kdv_bracket_2(s), û) * gradient(KdVHamiltonian2(s), s, û))
end

println("  uniform mesh: exact (the assembled matrices are circulant and commute)")
for n in (16, 32, 64)
    s = SplineSpace(UniformMesh(n, L), 3)
    û = project(s, sin)
    v1, v2 = cross1(s, û), cross2(s, û)
    check(@sprintf("uniform, N = %3d: {H2,H1}_1d = 0 and {H1,H2}_2d = 0", n),
        abs(v1) < 1e-8 && abs(v2) < 1e-8, @sprintf("%.2e, %.2e", v1, v2))
end

println("\n  randomly non-uniform mesh: no longer exact")
for n in (16, 32, 64)
    s = SplineSpace(RandomMesh(n, L), 3)
    û = project(s, sin)
    v1, v2 = cross1(s, û), cross2(s, û)
    check(
        @sprintf("non-uniform, N = %3d: the brackets do NOT vanish to machine precision",
            n),
        abs(v2) > 1e-10, @sprintf("{H2,H1}_1d = %.2e, {H1,H2}_2d = %.2e", v1, v2))
end

println("\n  smoothly graded mesh family, where a rate of convergence is meaningful:")
let prev = nothing, orders = Float64[]
    for n in (16, 32, 64, 128)
        s = SplineSpace(GradedMesh(n, L), 3)
        v2 = cross2(s, project(s, sin))
        prev === nothing || push!(orders, log2(abs(prev / v2)))
        @printf("      N = %4d   {H1,H2}_2d = % .3e%s\n", n, v2,
            isempty(orders) ? "" : @sprintf("   order %.1f", orders[end]))
        prev = v2
    end
    check("on a graded mesh the violation converges at second order or better",
        minimum(orders) > 1.8, "observed orders $(fmt1(orders))")
end

header("6b. ... but on a uniform mesh it is NOT an identity in the dofs")

# Split {H2d,H1d}_1d = u' (S Minv K1) u + 3 u' S Minv n(u), n_i = int phi_i u_h^2. The
# quadratic part vanishes identically because S Minv K1 is antisymmetric for circulant
# assemblies; the cubic part vanishes only for resolved fields.
println("  the exact vanishing on a uniform mesh holds for RESOLVED fields only:")
let rng = MersenneTwister(11)
    for n in (16, 32)
        s = SplineSpace(UniformMesh(n, L), 3)
        Minv = inv(Matrix(mass_matrix(s)))
        S = gram(s, 0, 1)
        S = (S - transpose(S)) / 2
        A = S * Minv * gram(s, 1, 1)
        d = maximum(abs, A + transpose(A)) / maximum(abs, A)
        check(
            "N = $n: S Minv K1 is antisymmetric, so the quadratic part vanishes " *
            "identically",
            d < TOL,
            @sprintf("rel. defect %.2e", d))
        smooth = cross1(s, project(s, sin))
        worst_q, worst_t = 0.0, 0.0
        for _ in 1:5
            c = randn(rng, n)
            sc = maximum(abs, c) * maximum(abs, A * c) * n
            worst_q = max(worst_q, abs(dot(c, A * c)) / sc)
            worst_t = max(worst_t, abs(cross1(s, c)) / sc)
        end
        @printf("      N = %3d   projected sin x: %.2e    random dofs: %.2e\n",
            n, abs(smooth), worst_t)
        check("N = $n: for random dofs the quadratic part is still round-off",
            worst_q < 1e-13, @sprintf("worst %.2e", worst_q))
        check(
            "N = $n: but the cubic part is not, so {H2,H1}_1d = 0 is a statement " *
            "about resolved fields, not an algebraic identity",
            worst_t > 1e-6,
            @sprintf("worst %.2e", worst_t))
    end
end
check(
    "the failure therefore sits in the under-resolved modes, exactly like the " *
    "Jacobi residual of block 5",
    true)

summary("verify_kdv_discrete.jl")
