#!/usr/bin/env julia
#
# Satisfying the structure-constant condition: what fails and what works.
#
#     julia --project=scripts scripts/verify_liepoisson_structure_constants.jl
#
# A bracket J_ij(u) = sum_m c_ij^m u_m is Poisson iff
#
#     sum_m ( c_ij^m c_mk^n + c_jk^m c_mi^n + c_ki^m c_mj^n ) = 0        (*)
#
# for all i,j,k,n, i.e. iff the c_ij^m are structure constants of a Lie algebra. Two
# requirements have to be kept apart:
#
#   (H) closure -- the c must be structure constants of SOME Lie algebra. This is what makes
#       the Jacobi identity hold, exactly, at every N.
#   (C) asymptotic homomorphism -- the discretisation must converge to the continuum
#       commutator. This is what makes the scheme consistent.
#
# Conventional finite elements deliver (C) and not (H), and (H) cannot be approximated: (*)
# is a CLOSED condition, so there is no "second-order accurate Jacobi identity". This script
# measures both sides of that statement.
#
#   1. The nodal finite element coefficients of Section 4 violate (*), and the violation does
#      not decrease under mesh refinement.
#   2. The sine bracket closes into su(N) exactly, and matches the closed form
#      c_{mn}^{m+n} = (N/2pi) sin( (2pi/N) m x n ), which converges to m x n at order N^-2.
#   3. A plain Fourier truncation, structure constants m x n with indices wrapped mod N,
#      does NOT close.
#   4. Feeding the sine structure constants into the four-bracket of Theorem 5.7 reproduces
#      the Zeitlin bracket and satisfies the Jacobi identity.

using GeometricBrackets
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "check.jl"));
using .Checks: header, check, summary

# `structure_constant_residual(C; normalised = false)` returns exactly the pair the Python
# helper did -- the same cyclic sum and the same scale max|c·c| -- once the tensor is stored
# as C[m,i,j] rather than c[i,j,m].
resid(C) = structure_constant_residual(C; normalised = false)

antisym_defect(C) = maximum(abs, C .+ permutedims(C, (1, 3, 2)))

header("1. Nodal finite elements: the violation does not decrease with h")

println("   c_ij^m = b_m sum_pq M^-1_ip [phi_p, phi_q](x_m) M^-1_qj ,")
println("   [phi_p, phi_q] = phi_p phi_q' - phi_q phi_p' ,  b_m = int phi_m .")
println()
for p in (1, 2)
    rows = NTuple{4, Float64}[]
    for ne in (8, 16, 32, 64)
        s = LagrangeSpace(p, ne)
        N = nbasis(s)
        M = Matrix(mass_matrix(s))
        Minv = inv(M)
        I₀ = basis_integrals(s)
        E = nodal_derivative_matrix(s)                    # E[q, m] = φ_q'(x_m)
        # [φ_p, φ_q](x_m) = δ_pm φ_q'(x_m) - δ_qm φ_p'(x_m)
        #   ⟹  c^m_ij = b_m [ Minv_im (Eᵀ Minv)_mj - (Minv E)_im Minv_mj ]
        A = Minv * E                                      # A[i, m]
        B = transpose(E) * Minv                           # B[m, j]
        C = [I₀[m] * (Minv[i, m] * B[m, j] - A[i, m] * Minv[m, j])
             for m in 1:N, i in 1:N, j in 1:N]
        res, scale = resid(C)
        push!(rows, (N, 2π / ne, antisym_defect(C) / maximum(abs, C), res / scale))
    end
    println("   P$p:")
    println("     N     h        rel. antisym. err   normalised residual of (*)")
    for (N, h, a, r) in rows
        @printf("     %4d  %.4f   %.2e            %.4f\n", N, h, a, r)
    end
    check("P$p: c is anti-symmetric in i,j", maximum(r[3] for r in rows) < 1e-10)
    check("P$p: c violates the structure-constant condition", rows[end][4] > 1e-3)
    check("P$p: the violation does not decrease under refinement",
        rows[end][4] > 0.5 * rows[1][4],
        @sprintf("residual %.4f -> %.4f over a 8x refinement", rows[1][4], rows[end][4]))
end

header("2. The sine bracket closes into su(N), from clock and shift matrices")

for N in (3, 5)
    modes, C = sine_algebra(N)
    d = length(modes)
    pos = Dict(m => i for (i, m) in enumerate(modes))
    res, scale = resid(C)
    check("N = $N: dim = N^2 - 1 = $(N*N-1) = dim su($N)", d == N * N - 1)
    check("N = $N: c is anti-symmetric in i,j", antisym_defect(C) < 1e-10)
    check("N = $N: the sine bracket satisfies the structure-constant condition",
        res / scale < 1e-10, @sprintf("normalised residual = %.2e", res / scale))
    err = 0.0
    for m in modes, n in modes

        s, closed = sine_coefficient(N, m, n)
        s == (0, 0) && continue
        err = max(err, abs(C[pos[s], pos[m], pos[n]] - closed))
    end
    check("N = $N: c_mn^(m+n) = (N/2pi) sin( (2pi/N) m x n )", err < 1e-9,
        @sprintf("max deviation = %.2e", err))
end

# convergence of the sine structure constants to the continuum ones:
#   (N/2pi) sin( (2pi/N) z ) = z ( 1 - (2 pi^2 / 3) z^2 / N^2 + ... )
println()
println("   (N/2pi) sin( (2pi/N) z ) -> z  as N -> infinity, for z = m x n:")
const Ns = [41, 81, 161, 321]
for z in (1, 3)
    vals = [(N, (N / 2π) * sin(2π * z / N)) for N in Ns]
    println("     z = $z: " * join([@sprintf("N=%d: %.6f", N, v) for (N, v) in vals], ", "))
    errs = [abs(v - z) for (_, v) in vals]
    rates = [log(errs[i] / errs[i + 1]) / log(Ns[i + 1] / Ns[i])
             for i in 1:(length(Ns) - 1)]
    println("              rates: " * join([@sprintf("%.3f", r) for r in rates], ", "))
    check("z = $z: convergence to z at rate N^-2", abs(rates[end] - 2.0) < 0.05,
        @sprintf("finest rate = %.3f", rates[end]))
end

header("3. A plain Fourier truncation does not close")

for N in (5, 7)
    modes = [(m1, m2) for m1 in 0:(N - 1) for m2 in 0:(N - 1) if (m1, m2) != (0, 0)]
    pos = Dict(m => i for (i, m) in enumerate(modes))
    d = length(modes)
    rep(a) = a > N ÷ 2 ? a - N : a                        # symmetric representatives
    C = zeros(d, d, d)
    for m in modes, n in modes

        s = (mod(m[1] + n[1], N), mod(m[2] + n[2], N))
        s == (0, 0) && continue
        C[pos[s], pos[m], pos[n]] = rep(m[1]) * rep(n[2]) - rep(m[2]) * rep(n[1])
    end
    res, scale = resid(C)
    check("N = $N: truncated Fourier structure constants do NOT close",
        res / scale > 1e-3, @sprintf("normalised residual = %.3f", res / scale))
end

header("4. Zeitlin's bracket as an instance of Theorem 5.7")

# K^{i α j β} = δ^{αβ} c_ij^α with s = 2/3 Σ z^{3/2} gives J_ij = Σ_m c_ij^m z_m, which must
# satisfy the Jacobi identity.
let N = 3
    modes, C = sine_algebra(N)
    d = length(modes)
    rng = MersenneTwister(0)
    w = 1.0 .+ 3.0 .* rand(rng, d)                        # w_a = √z_a
    z = w .^ 2
    J = lie_poisson_matrix(C, z)
    # contraction through the four-bracket: J_ij = Σ_ab w_a K^{iajb} w_b
    J4 = zeros(ComplexF64, d, d)
    for a in 1:d
        J4 .+= w[a] .* C[a, :, :] .* w[a]
    end
    check("the four-bracket contraction reproduces sum_m c_ij^m z_m",
        maximum(abs, J - J4) < 1e-10)
    worst = first(jacobi_residual(J, lie_poisson_derivative(C); normalised = false))
    nrm = worst / maximum(abs, J)^2
    check("the induced two-bracket satisfies the Jacobi identity", nrm < 1e-10,
        @sprintf("normalised residual = %.2e", nrm))
end

summary("verify_liepoisson_structure_constants.jl")
