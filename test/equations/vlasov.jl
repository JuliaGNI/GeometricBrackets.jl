using GeometricBrackets
using LinearAlgebra
using Random
using SimpleSplines: Periodic, Free, Dirichlet
using Test

@testset "$(rpad("Vlasov Tests",80))" begin
    @testset "$(rpad("direct Galerkin Vlasov tensor is antisymmetric and fails Jacobi",76))" begin
        s = TensorSplineSpace((3, 3), 2, (Periodic(), Free()))
        C = vlasov_galerkin_tensor(s)
        N = nbasis(s)
        @test size(C) == (N, N, N)
        @test is_antisymmetric_c(C)

        rng = MersenneTwister(42)
        c = randn(rng, N)
        J_ten = lie_poisson_matrix(C, c)
        J_int = vlasov_weighted_poisson_matrix(s, field(s, c, (0, 0)))
        @test J_ten ≈ J_int
        @test maximum(abs, J_ten + J_ten') < 1e-13

        # Polynomial splines do not close under the canonical bracket: O(1) Jacobi failure
        @test structure_constant_residual(C) > 0.1
        @test jacobi_residual(J_ten, C) > 0.05
    end

    @testset "$(rpad("time-reversal parity split: isotropic diagonal blocks and exact Jacobi",76))" begin
        for bc_v in (Free(), Dirichlet()), orth in (false, true)

            s = TensorSplineSpace((4, 5), 2, (Periodic(), bc_v))
            U_plus, U_minus = velocity_parity_projectors(s; orthonormal = orth)
            N = nbasis(s)
            @test size(U_plus, 1) == N
            @test size(U_minus, 1) == N
            @test size(U_plus, 2) + size(U_minus, 2) == N

            if orth
                M = Matrix(mass_matrix(s))
                @test U_plus' * M * U_plus ≈ I
                @test U_minus' * M * U_minus ≈ I
                @test maximum(abs, U_plus' * M * U_minus) < 1e-13
            else
                @test U_plus' * U_plus ≈ I
                @test U_minus' * U_minus ≈ I
                @test maximum(abs, U_plus' * U_minus) < 1e-14
            end

            # Velocity-even background f0(x, 2π - v) == f0(x, v): diagonal blocks vanish
            f0_even = xv -> (1.0 + 0.25 * cos(xv[1])) * exp(-0.5 * (xv[2] - π)^2)
            J_pp, J_pm, J_mp, J_mm = parity_vlasov_blocks(
                s, f0_even; orthonormal = orth)
            @test maximum(abs, J_pp) < 1e-13
            @test maximum(abs, J_mm) < 1e-13
            @test J_mp ≈ -J_pm'
            @test norm(J_pm) > 0.1

            # Negative control: drifted (velocity-skew) background breaks isotropy
            f0_skew = xv -> (1.0 + 0.25 * cos(xv[1])) * exp(-0.5 * (xv[2] - π - 0.7)^2)
            J_pp_skew, _, _, J_mm_skew = parity_vlasov_blocks(
                s, f0_skew; orthonormal = orth)
            @test maximum(abs, J_pp_skew) > 1e-2
            @test maximum(abs, J_mm_skew) > 1e-2

            # Parity Vlasov bracket and DLRA parity Vlasov bracket are ConstantBrackets
            b_par = parity_vlasov_bracket(s, f0_even; orthonormal = orth)
            z = ones(N)
            @test isantisymmetric(b_par, z)
            @test iszero(jacobi_residual(b_par, z))

            nx, nv = size(s)
            m_minus = div(nv, 2)
            m_plus = nv - m_minus
            r = 2
            Ux = Matrix(qr(randn(MersenneTwister(1), nx, r)).Q)
            Uv_p = Matrix(qr(randn(MersenneTwister(2), m_plus, r)).Q)
            Uv_m = Matrix(qr(randn(MersenneTwister(3), m_minus, r)).Q)
            b_dlra = dlra_parity_vlasov_bracket(J_pm, Ux, Uv_p, Uv_m)
            z_dlra = ones(2 * r^2)
            @test isantisymmetric(b_dlra, z_dlra)
            @test iszero(jacobi_residual(b_dlra, z_dlra))
        end
    end

    @testset "$(rpad("canonical Clebsch Vlasov map and non-submersion theorem",76))" begin
        s = TensorSplineSpace((3, 3), 2, (Periodic(), Periodic()))
        N = nbasis(s)
        B = clebsch_vlasov_tensor(s)
        @test size(B) == (N, N, N)
        @test is_antisymmetric_c(B)

        b_can = canonical_vlasov_bracket(N)
        rng = MersenneTwister(7)
        Q = randn(rng, N)
        P = randn(rng, N)
        z = vcat(Q, P)
        @test isantisymmetric(b_can, z)
        @test iszero(jacobi_residual(b_can, z))

        c0 = ones(N)
        c = clebsch_vlasov_map(B, Q, P, c0)
        b_int = basis_integrals(s)
        # Partition of unity on T^2: total mass b^T c(Q, P) == b^T c0
        @test dot(b_int, c) ≈ dot(b_int, c0) atol = 1e-13

        # Pushforward bivector Π_{mn}(Q, P) == -Q^T [B_m, B_n] P
        Π = clebsch_pushforward_matrix(B, Q, P)
        @test maximum(abs, Π + Π') < 1e-13
        Π_comm = [-dot(Q, (B[m, :, :] * B[n, :, :] - B[n, :, :] * B[m, :, :]) * P)
                  for m in 1:N, n in 1:N]
        @test Π ≈ Π_comm

        # Non-submersion on non-closed spline space: Π(Q, P) != J_LP(c(Q, P) - c0)
        C_gal = vlasov_galerkin_tensor(s)
        J_lp = lie_poisson_matrix(C_gal, c - c0)
        @test norm(Π - J_lp) / norm(Π) > 0.1
        @test norm(Π + J_lp) / norm(Π) > 0.1

        # Positive control: when B is the adjoint representation of a closed Lie algebra (so(3)),
        # the pushforward coincides identically with the Lie-Poisson bracket on c - c0
        C_so3 = Float64.(so3())
        q3 = [0.4, -0.7, 1.1]
        p3 = [-0.5, 0.3, 0.8]
        c3 = clebsch_vlasov_map(C_so3, q3, p3)
        Π_so3 = clebsch_pushforward_matrix(C_so3, q3, p3)
        @test Π_so3 ≈ lie_poisson_matrix(C_so3, c3)
    end

    @testset "$(rpad("u(N)* Weyl-Vlasov quantisation and log-mean spectral pushforward",76))" begin
        for N in (2, 3, 4)
            E = un_hermitian_basis(N)
            M = N^2
            @test length(E) == M
            gram = [real(tr(E[a] * E[b])) for a in 1:M, b in 1:M]
            @test gram ≈ I

            C = un_structure_constants(N; ħ = 0.5)
            @test is_antisymmetric_c(C)
            @test structure_constant_residual(C) < 1e-14
            rng = MersenneTwister(10 + N)
            w = randn(rng, M)
            J_un = lie_poisson_matrix(C, w)
            @test maximum(abs, J_un + J_un') < 1e-14
            @test jacobi_residual(J_un, C) < 1e-14
        end

        # Weyl quantisation: exact on the inner block for degree <= 1, Wigner O(ħ^2) at degree 2
        K = 8
        ħ = 0.25
        a_deg1 = Dict((1, 1) => 0.5 + 0.0im, (-1, 1) => 0.5 + 0.0im, (2, 0) => 0.3 + 0.0im)
        b_deg1 = Dict((1, 1) => -0.25im, (-1, 1) => 0.25im, (-1, 0) => 0.4 + 0.0im)
        # Poisson bracket {a, b}_{(x, v)} = ∂_x a ∂_v b - ∂_v a ∂_x b for degree <= 1 has
        # zero third-order Moyal correction, so [T_N(a), T_N(b)]/(iħ) == T_N({a, b}) on |k|,|l| <= K - 3
        pb_deg1 = Dict{Tuple{Int, Int}, ComplexF64}()
        for ((n1, r1), c1) in a_deg1, ((n2, r2), c2) in b_deg1

            if r2 >= 1
                key = (n1 + n2, r1 + r2 - 1)
                pb_deg1[key] = get(pb_deg1, key, 0.0 + 0.0im) + (im * n1 * r2) * c1 * c2
            end
            if r1 >= 1
                key = (n1 + n2, r1 + r2 - 1)
                pb_deg1[key] = get(pb_deg1, key, 0.0 + 0.0im) - (im * n2 * r1) * c1 * c2
            end
        end
        Ta = weyl_vlasov_matrix(a_deg1, K, ħ)
        Tb = weyl_vlasov_matrix(b_deg1, K, ħ)
        Tpb = weyl_vlasov_matrix(pb_deg1, K, ħ)
        comm_ab = (Ta * Tb - Tb * Ta) / (im * ħ)
        inner = (K + 1 - 4):(K + 1 + 4)
        @test maximum(abs, comm_ab[inner, inner] - Tpb[inner, inner]) < 1e-13

        # Logarithmic-mean spectral pushforward (φ(λ) = exp(2λ)/2) passes Jacobi at N = 3
        rng = MersenneTwister(99)
        w3 = 0.3 .* randn(rng, 9)
        J_bkm, dJ_bkm = spectral_pushforward_bracket(
            λ -> 0.5 * exp(2λ), λ -> exp(2λ), w3)
        @test maximum(abs, J_bkm + J_bkm') < 1e-13
        @test jacobi_residual(J_bkm, dJ_bkm) < 1e-8
    end

    @testset "$(rpad("semidirect-product Vlasov moment bracket: pointwise-flat vs direct",76))" begin
        # Exact Rational{BigInt} check on both periodic (antisymmetric Ka) and wall-bounded (general Ka)
        Ka_per = Rational{BigInt}[0 1 0 -1; -1 0 1 0; 0 -1 0 1; 1 0 -1 0] .// 2
        Ka_wall = Rational{BigInt}[-3 4 -1 0; -2 0 2 0; 0 -2 0 2; 0 1 -4 3] .// 5
        Kb_mag = Rational{BigInt}[0 2 -1 0; -2 0 0 1; 1 0 0 -3; 0 -1 3 0] .// 7
        ρ_q = Rational{BigInt}[2, 3, 5, 4] .// 2
        m_q = Rational{BigInt}[1, -3, 2, 5] .// 3

        for Ka in (Ka_per, Ka_wall),
            Kb in (zeros(Rational{BigInt}, 4, 4), Kb_mag)

            J_flat, dJ_flat = vlasov_moment_flat_bracket(Ka, ρ_q, m_q; Kb = Kb)
            @test J_flat == -transpose(J_flat)
            @test iszero(jacobi_residual(J_flat, dJ_flat))
        end

        # Direct collocation fails Jacobi at O(1)
        J_dir, dJ_dir = vlasov_moment_direct_bracket(Ka_per, ρ_q, m_q)
        @test J_dir == -transpose(J_dir)
        @test !iszero(jacobi_residual(J_dir, dJ_dir))
        @test Float64(jacobi_residual(J_dir, dJ_dir)) > 0.05
    end

    @testset "$(rpad("argument validation",76))" begin
        s = TensorSplineSpace((3, 3), 2)
        @test_throws DimensionMismatch vlasov_weighted_poisson_matrix(s, ones(5))
        @test_throws DimensionMismatch parity_vlasov_blocks(s, ones(5))
        @test_throws ArgumentError canonical_vlasov_bracket(0)
        @test_throws ArgumentError un_hermitian_basis(0)
        @test_throws ArgumentError weyl_vlasov_matrix(Dict((0, 0) => 1.0), -1, 1.0)
        @test_throws DimensionMismatch spectral_pushforward_bracket(identity, one, ones(5))
        @test_throws DimensionMismatch clebsch_vlasov_map(zeros(3, 3, 3), ones(2), ones(3))
        @test_throws DimensionMismatch clebsch_pushforward_matrix(
            zeros(3, 3, 3), ones(3), ones(2))
        @test_throws DimensionMismatch vlasov_moment_flat_bracket(
            zeros(3, 3), ones(2), ones(3))
        @test_throws DimensionMismatch vlasov_moment_direct_bracket(
            zeros(3, 3), ones(3), ones(2))
    end
end
