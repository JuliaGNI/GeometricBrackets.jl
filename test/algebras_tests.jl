using GeometricBrackets
using LinearAlgebra
using Random
using Test

@testset "$(rpad("Lie Algebra Tests",80))" begin
    @testset "$(rpad("se(3) is the positive control and passes EXACTLY",76))" begin
        C = se3()
        @test is_antisymmetric_c(C)
        @test structure_constant_residual(C) == 0        # over ℚ, not below a tolerance
        @test eltype(C) == Rational{BigInt}
    end

    @testset "$(rpad("a random antisymmetric c in dimension five FAILS: the control bites",76))" begin
        rng = MersenneTwister(0xb1a5)
        for _ in 1:20
            C = random_antisymmetric_c(rng, 5)
            @test is_antisymmetric_c(C)
            @test structure_constant_residual(C) != 0
        end
    end

    @testset "$(rpad("so(3) is DEGENERATE: it passes after being broken, so prove nothing with it",76))" begin
        C = so3()
        @test structure_constant_residual(C) == 0

        # Rescaling one m-slice, the perturbation verify_kdv_jacobi_family.py reaches for,
        # leaves it a Lie algebra. So does moving a single structure constant.
        Cs = copy(C)
        Cs[1, :, :] *= 3 // 2
        @test structure_constant_residual(Cs) == 0
        Cb = copy(C)
        Cb[3, 1, 2] = 2
        Cb[3, 2, 1] = -2
        @test structure_constant_residual(Cb) == 0

        # The reason is that all three sit in the six-parameter family c_ij^k = ε_ijl n^lk
        # with n SYMMETRIC, every member of which is a Lie algebra. It is NOT true that
        # antisymmetry alone forces Jacobi in three dimensions -- drop the symmetry of n, or
        # take a general antisymmetric c, and the residual is nonzero.
        ε = Dict((1, 2, 3) => 1, (2, 3, 1) => 1, (3, 1, 2) => 1,
            (3, 2, 1) => -1, (2, 1, 3) => -1, (1, 3, 2) => -1)
        function bianchiA(n)
            D = zeros(Rational{BigInt}, 3, 3, 3)
            for i in 1:3, j in 1:3, k in 1:3, l in 1:3
                e = get(ε, (i, j, l), 0)
                e == 0 || (D[k, i, j] += e * n[l, k])
            end
            D
        end
        rng = MersenneTwister(0x5417)
        for _ in 1:20
            M = Rational{BigInt}[rand(rng, -6:6) // rand(rng, 1:4) for _ in 1:3, _ in 1:3]
            @test structure_constant_residual(bianchiA((M + transpose(M)) // 2)) == 0
        end
        @test any(structure_constant_residual(random_antisymmetric_c(rng, 3)) != 0
        for _ in 1:20)
    end

    @testset "$(rpad("so(N) from the trace form is a Lie algebra for N = 3, 4, 5",76))" begin
        for N in (3, 4, 5)
            basis, C = so_n(N)
            @test length(basis) == N * (N - 1) ÷ 2
            @test size(C, 1) == length(basis)
            @test structure_constant_residual(C) == 0
            # the basis is orthonormal for ⟨X,Y⟩ = -tr(XY)/2, which is how C was read off
            @test all(-tr(basis[i] * basis[j]) // 2 == (i == j)
            for i in eachindex(basis),
            j in eachindex(basis))
        end
    end

    @testset "$(rpad("the Lie-Poisson tensor is antisymmetric and Poisson iff c is",76))" begin
        z = Rational{BigInt}[1 // 2, -3 // 4, 5 // 6, 7 // 8, -1 // 3, 2 // 5]
        J = lie_poisson_matrix(se3(), z)
        dJ = lie_poisson_derivative(se3())
        @test J == -transpose(J)
        @test jacobi_residual(J, dJ) == 0                # exact, for the linear bracket

        rng = MersenneTwister(0x9e11)
        Cbad = random_antisymmetric_c(rng, 6)
        @test jacobi_residual(lie_poisson_matrix(Cbad, z), Cbad) != 0
    end

    @testset "$(rpad("the sine algebra closes into su(N) exactly at every odd N",76))" begin
        for N in (5, 7, 9)
            modes, C = sine_algebra(N)
            @test length(modes) == N^2 - 1               # = dim su(N)
            res, scale = structure_constant_residual(C; normalised = false)
            @test res / scale < 1e-13
        end
        @test_throws ArgumentError sine_algebra(6)
    end

    @testset "$(rpad("the closed-form sine coefficient agrees with the assembled tensor",76))" begin
        for N in (5, 7, 9)
            modes, C = sine_algebra(N)
            pos = Dict(m => i for (i, m) in enumerate(modes))
            worst = 0.0
            for m in modes, n in modes

                s, c = sine_coefficient(N, m, n)
                s == (0, 0) && continue
                worst = max(worst, abs(C[pos[s], pos[m], pos[n]] - c))
            end
            @test worst < 1e-13
        end
    end

    @testset "$(rpad("the sine coefficients approach m×n at SECOND order in 1/N",76))" begin
        # The comparison must be made on a low-mode window held fixed as N grows, and the
        # N^-2 law is asymptotic in N ≫ 2π·(m×n): expanding the sine gives a leading error
        # -(2π)²(m×n)³/(6N²). At N = 11 with a cross product of 8 the angle is 4.6 rad and
        # the observed order is 0.8, which says nothing about the claim. These N are where
        # the asymptotic regime actually starts.
        window = [(m1, m2) for m1 in -2:2 for m2 in -2:2 if (m1, m2) != (0, 0)]
        deviation(N) = maximum(
            let (s, c) = sine_coefficient(N, (mod(m[1], N), mod(m[2], N)),
                    (mod(n[1], N), mod(n[2], N)))
                s == (0, 0) ? 0.0 : abs(c - (m[1] * n[2] - m[2] * n[1]))
            end
        for m in window, n in window)

        Ns = (41, 81, 161)
        errs = deviation.(Ns)
        @test all(errs[i + 1] < errs[i] for i in 1:(length(errs) - 1))
        orders = [log(errs[i] / errs[i + 1]) / log(Ns[i + 1] / Ns[i])
                  for i in 1:(length(Ns) - 1)]
        @test all(1.6 .< orders .< 2.2)
        @test 1.9 < last(orders) < 2.1      # and it is converging ON two, not past it
    end

    @testset "$(rpad("the mode truncations close on V1 but are NOT Lie algebras",76))" begin
        for tr in (witt_truncation(1), witt_truncation(2), poly_truncation(2),
            torus_truncation(1))
            C, keep, con = tr.C, tr.keep, tr.con
            @test is_antisymmetric_c(C)
            @test length(keep) + length(con) == size(C, 1) == length(tr.labels)
            # [V1,V1] ⊆ V1 ⊕ V2 exactly: this is the premise the hierarchy needs
            @test closes_on(C, keep, con)
            @test leak(C, keep, con) == 0
            # but the truncation itself drops terms and so is not a Lie algebra
            @test structure_constant_residual(C) != 0
            # and V2 is not an ideal, which is why naive truncation and Dirac differ
            @test !is_ideal(C, con, keep)
        end
    end

    @testset "$(rpad("witt_truncation reproduces [L_k,L_l] = (l-k) L_{k+l}",76))" begin
        tr = witt_truncation(1)                       # modes -2:2, so index i ↔ k = i-3
        C = tr.C
        for k in -2:2, l in -2:2

            abs(k + l) ≤ 2 || continue
            @test C[k + l + 3, k + 3, l + 3] == l - k
        end
        @test tr.labels == ["L(-2)", "L(-1)", "L(0)", "L(1)", "L(2)"]
        @test tr.keep == [2, 3, 4]                    # |k| ≤ 1
        @test tr.con == [1, 5]
        @test size(witt_truncation(1; drop_zero = true).C, 1) == 4
    end

    @testset "$(rpad("galerkin_c reduces to c itself when the mass matrix is the identity",76))" begin
        C = se3()
        T3 = similar(C)
        for m in 1:6, i in 1:6, j in 1:6
            T3[m, i, j] = C[m, i, j]                  # T_mpq = Σ_r g_pq^r M_mr with M = I
        end
        @test galerkin_c(Matrix{Rational{BigInt}}(I, 6, 6), T3) == C
    end
end
