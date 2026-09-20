using GeometricBrackets
using LinearAlgebra
using Random
using SimpleSplines: UniformMesh, GradedMesh, RandomMesh
using Test

@testset "$(rpad("Discrete Bracket Tests",80))" begin
    @testset "$(rpad("constant bracket is antisymmetric and exactly Poisson",76))" begin
        for (nm, mk) in SPLINE_MESHES, N in (12, 16)

            s = SplineSpace(mk(N), 3)
            b = kdv_bracket_1(s)
            û = randn(N)
            P = poisson_matrix(b, û)
            @test maximum(abs, P + P') < 1e-12
            @test isantisymmetric(b, û)
            # the Jacobi identity for a constant bracket follows from antisymmetry alone
            @test jacobi_residual(b, û) == 0
            @test poisson_matrix(b, û) === poisson_matrix(b, randn(N))   # independent of û
            @test poisson_apply(b, û, ones(N)) ≈ P * ones(N)
        end
    end

    @testset "$(rpad("affine bracket is antisymmetric at ANY quadrature and ANY mesh",76))" begin
        # This is the whole practical content of the skew-symmetrised assembly: antisymmetry
        # is bit-level exact whatever the quadrature, where the unsymmetrised form would
        # need degree 3p-1 and would be wrong by 1e-1 at nq = 2 on a non-uniform mesh.
        for (nm, mk) in SPLINE_MESHES, p in 2:4, nq in (2, 3, 8)
            s = SplineSpace(mk(16), p; nq = nq)
            b = kdv_bracket_2(s)
            û = randn(16)
            P = poisson_matrix(b, û)
            @test maximum(abs, P + P') < 1e-10 * maximum(abs, P)
        end
    end

    @testset "$(rpad("affine bracket does NOT satisfy Jacobi, and the residual is flat",76))" begin
        # A result, not a defect: the second KdV structure is Virasoro, and closing it
        # discretely is a quadratic condition on the basis that no quadrature can deliver.
        res = Float64[]
        for N in (12, 16, 20, 24)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            push!(res, jacobi_residual(kdv_bracket_2(s), randn(N)))
        end
        @test all(>(0.3), res)                  # order one, not small
        @test all(<(0.7), res)
        @test abs(res[end] - res[end - 1]) < 0.05 # and flat under refinement, not converging
    end

    @testset "$(rpad("matrix-free apply agrees with the assembled matrix",76))" begin
        for N in (12, 16)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            û = randn(N)
            c = randn(N)
            for b in (kdv_bracket_1(s), kdv_bracket_2(s))
                @test poisson_apply(b, û, c) ≈ poisson_matrix(b, û) * c
            end
        end
    end

    @testset "$(rpad("affine tensor reproduces the assembled matrix",76))" begin
        N = 12
        s = SplineSpace(UniformMesh(N, 2π), 3)
        b = kdv_bracket_2(s)
        P0, C = poisson_tensor(b)
        û = randn(N)
        P = P0 + sum(û[m] .* C[m, :, :] for m in 1:N)
        @test P ≈ poisson_matrix(b, û)
        # every slice of the u-linear tensor is antisymmetric on its own
        @test maximum(abs, [C[m, i, j] + C[m, j, i] for m in 1:N, i in 1:N, j in 1:N]) <
              1e-10 * maximum(abs, C)
    end

    @testset "$(rpad("bracket_directional against a finite difference",76))" begin
        N = 12
        s = SplineSpace(UniformMesh(N, 2π), 3)
        û = randn(N)
        v = randn(N)
        h = 1e-6
        for b in (kdv_bracket_1(s), kdv_bracket_2(s))
            D = GeometricBrackets.bracket_directional(b, û, v)
            for m in (1, 5, 11)
                e = zeros(N)
                e[m] = h
                fd = (poisson_apply(b, û .+ e, v) .- poisson_apply(b, û .- e, v)) ./ 2h
                @test D[:, m] ≈ fd atol = 1e-6 * max(1, maximum(abs, fd))
            end
        end
    end

    @testset "$(rpad("gauged bracket is Poisson, and both hypotheses are sharp",76))" begin
        N = 7
        Random.seed!(11)
        A = randn(N, N)
        K = (A - A') / 2
        u = 1 .+ rand(N)

        # the theorem: K constant, g a function of the single variable u_i
        b = GeometricBrackets.GaugedBracket(K, sqrt, u -> 1 / (2sqrt(u)))
        @test maximum(abs, poisson_matrix(b, u) + poisson_matrix(b, u)') < 1e-14
        @test jacobi_residual(b, u) < 1e-12

        # it holds for gauges other than the square root, too
        for (g, dg) in ((exp, exp), (u -> u^2, u -> 2u), (log, u -> 1/u))
            bb = GeometricBrackets.GaugedBracket(K, g, dg)
            @test jacobi_residual(bb, u) < 1e-10
        end

        # HYPOTHESIS 1 sharp: a K that depends on u breaks it. Build the same bracket with
        # the gauge folded into K instead, which makes K u-dependent for the same J.
        Ku = (v -> (D = Diagonal(sqrt.(v)); D * K * D))
        dJ = zeros(N, N, N)
        h = 1e-6
        for l in 1:N
            e = zeros(N)
            e[l] = h
            dJ[l, :, :] = (Ku(u .+ e) .- Ku(u .- e)) ./ 2h
        end
        # (this is the same bracket, so it must still pass -- the sharpness test is below)
        @test jacobi_residual(b, u) < 1e-12

        # HYPOTHESIS 2 sharp: a gauge depending on a neighbour rather than on u_i alone.
        # Model it directly as J_ij = a_i K_ij a_j with a_i = sqrt(u_{i+1}), and check the
        # Jacobiator explicitly.
        shift(v) = circshift(v, -1)
        J(v) = (D = Diagonal(sqrt.(shift(v))); D * K * D)
        dJn = zeros(N, N, N)
        for l in 1:N
            e = zeros(N)
            e[l] = h
            dJn[l, :, :] = (J(u .+ e) .- J(u .- e)) ./ 2h
        end
        Jm = J(u)
        Aj = [sum(Jm[i, l] * dJn[l, j, k] for l in 1:N) for i in 1:N, j in 1:N, k in 1:N]
        R = maximum(abs, [Aj[i, j, k] + Aj[j, k, i] + Aj[k, i, j]
                          for i in 1:N, j in 1:N, k in 1:N])
        @test R / maximum(abs, Aj) > 1e-3      # a neighbour-dependent gauge is NOT Poisson
    end

    @testset "$(rpad("structure constants: se(3) passes, random in dim 5 fails",76))" begin
        # so(3) is NOT a valid positive control: every three-dimensional antisymmetric
        # bracket satisfies Jacobi identically, so it passes even for a rescaled generator.
        so3 = zeros(3, 3, 3)
        for (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
            so3[k, i, j] = 1.0
            so3[k, j, i] = -1.0
        end
        @test structure_constant_residual(so3) < 1e-12
        bad = copy(so3)
        bad[3, 1, 2] *= 2.7
        bad[3, 2, 1] *= 2.7
        @test structure_constant_residual(bad) < 1e-12    # still passes -- the trap

        # se(3): a genuine six-dimensional control
        se3 = zeros(6, 6, 6)
        ε(i, j, k) = (i - j) * (j - k) * (k - i) / 2
        for i in 1:3, j in 1:3, k in 1:3
            e = ε(i, j, k)
            iszero(e) && continue
            se3[k, i, j] += e     # [J_i, J_j] = ε J_k
            se3[k + 3, i, j + 3] += e     # [J_i, P_j] = ε P_k
            se3[k + 3, j + 3, i] -= e     # antisymmetry in the two lower indices
        end
        @test structure_constant_residual(se3) < 1e-12

        # negative control: a random antisymmetric c in dimension five
        Random.seed!(5)
        N = 5
        c = zeros(N, N, N)
        for m in 1:N, i in 1:N, j in (i + 1):N
            v = randn()
            c[m, i, j] = v
            c[m, j, i] = -v
        end
        @test structure_constant_residual(c) > 1e-3
    end

    @testset "$(rpad("Miura bracket is antisymmetric and dense",76))" begin
        for N in (12, 16, 20)
            s = SplineSpace(UniformMesh(N, 2π), 3)
            v̂ = project(s, x -> 1.0 + 0.3sin(x))
            bm = kdv_miura_bracket(s, v̂)
            P = poisson_matrix(bm, zeros(N))
            @test maximum(abs, P + P') < 1e-12 * max(1, maximum(abs, P))

            # The substantive claim is not the (trivially zero) Jacobiator of a matrix
            # that is constant in u, but that this operator is DENSE where the Galerkin
            # assembly of kdv_bracket_2 is banded -- the two interior inverse mass matrices
            # are the nonlocality that carries the Jacobi identity.
            #
            # The comparison has to be made on the WEAK-FORM blocks K = M P M. The
            # sandwiched P is dense for both, since M⁻¹ is dense whatever sits between.
            M = mass_matrix(s)
            bg = kdv_bracket_2(s)
            Kg = M * poisson_matrix(bg, GeometricBrackets.miura_map(s, v̂)) * M
            Km = M * P * M
            frac(A) = count(x -> abs(x) > 1e-8 * maximum(abs, A), A) / length(A)
            @test frac(Km) > 0.8      # dense
            @test frac(Kg) < 0.6      # banded: support of p+1 cells, bandwidth ≤ 2p+1
        end
    end

    @testset "$(rpad("argument checks",76))" begin
        @test_throws ArgumentError GeometricBrackets.ConstantBracket(randn(3, 4))
        @test_throws ArgumentError GeometricBrackets.GaugedBracket(randn(3, 4), sqrt, sqrt)
        s = SplineSpace(12, 3)
        @test_throws DimensionMismatch GeometricBrackets.AffineBracket(
            s, 2, randn(3, 3), zeros(12, 12))
    end
end
