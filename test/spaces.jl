using GeometricBrackets
using LinearAlgebra
using SimpleSplines: UniformMesh, GradedMesh, RandomMesh, MassOperator
using SparseArrays
using Random
using Test

include("helpers/meshes.jl")

Random.seed!(0x5c1e9a3b)

@testset "$(rpad("Discrete Space Tests",80))" begin
    @testset "$(rpad("spline space accessors",76))" begin
        s = SplineSpace(16, 3)
        @test nbasis(s) == 16
        @test degree(s) == 3
        @test ncells(s) == 16
        @test domainlength(s) ≈ 2π
        @test length(s) == 16
        @test eltype(s) == Float64
        @test length(nodes(s)) == 16
        @test length(quadrature_weights(s)) == length(quadrature_nodes(s))
        @test sum(quadrature_weights(s)) ≈ 2π
        @test size(basis_values(s, 0)) == (16, length(quadrature_nodes(s)))
    end

    @testset "$(rpad("Lagrange space accessors",76))" begin
        s = LagrangeSpace(2, 13)
        @test nbasis(s) == 26          # N = p * ne
        @test degree(s) == 2
        @test ncells(s) == 13
        @test domainlength(s) ≈ 2π
        @test length(nodes(s)) == 26
        @test issorted(nodes(s))
        @test sum(quadrature_weights(s)) ≈ 2π
        @test_throws ArgumentError LagrangeSpace(0, 13)
        @test_throws ArgumentError LagrangeSpace(2, 0)
        # only C⁰, so the second derivative does not exist across element boundaries
        @test_throws ArgumentError basis_values(s, 2)
    end

    @testset "$(rpad("the Lagrange tabulation is sparse, with the support-sized nnz",76))" begin
        # A basis function is supported on at most two elements, so only p+1 of the N rows
        # are nonzero in any quadrature column. Storing that densely is what this asserts
        # against: at p = 2, ne = 192 a dense table is 0.8 % full.
        for (p, ne) in ((1, 25), (2, 13), (3, 16))
            s = LagrangeSpace(p, ne)
            nq = 2p + 4
            for d in 0:1
                Φ = basis_values(s, d)
                @test Φ isa SparseMatrixCSC
                @test size(Φ) == (p * ne, ne * nq)
                # exactly one structural entry per (local dof, quadrature point) per element
                @test nnz(Φ) == ne * (p + 1) * nq
            end
        end

        # the mass matrix is sparse too, and its operator answers `mass_solve!`
        s = LagrangeSpace(2, 13)
        @test mass_matrix(s) isa SparseMatrixCSC
        @test mass_factorization(s) isa MassOperator
        # `inverse_mass_matrix` stays dense, as it does for a spline space
        @test inverse_mass_matrix(s) isa Matrix
        @test inverse_mass_matrix(s) * mass_matrix(s) ≈ I

        # `mixed_matrix` is memoised, so the second call returns the SAME object
        A = mixed_matrix(s, 1, 1)
        @test A === mixed_matrix(s, 1, 1)
        @test A ≈ stiffness_matrix(s)
    end

    @testset "$(rpad("project! works on a Lagrange space and matches project",76))" begin
        # It did not: `mass_factorization` returned a bare Cholesky, for which `mass_solve!`
        # has no method, so every in-place projection onto a Lagrange space was a MethodError.
        # Nothing exercised it until the tabulation was made sparse.
        for (p, ne) in ((1, 25), (2, 13))
            s = LagrangeSpace(p, ne)
            f = x -> 2.0 + sin(x) + 0.3cos(2x)
            u = project(s, f)
            fv = f.(quadrature_nodes(s))
            û = similar(u)
            project!(û, s, fv)
            @test û ≈ u
        end
    end

    @testset "$(rpad("mass matrix and partition of unity",76))" begin
        for s in (SplineSpace(16, 3), SplineSpace(GradedMesh(16, 2π), 2),
            LagrangeSpace(1, 25), LagrangeSpace(2, 13))
            M = mass_matrix(s)
            N = nbasis(s)
            @test M ≈ M'
            @test isposdef(M)
            @test sum(M) ≈ domainlength(s)
            # M 1 = ∫φ_i, and the integrals sum to |Ω|
            @test M * ones(N) ≈ basis_integrals(s)
            @test sum(basis_integrals(s)) ≈ domainlength(s)
            @test mass_factorization(s) \ (M * ones(N)) ≈ ones(N)
        end
    end

    @testset "$(rpad("S is antisymmetric on every space",76))" begin
        for s in (SplineSpace(16, 3), SplineSpace(RandomMesh(16, 2π), 3),
            LagrangeSpace(1, 25), LagrangeSpace(2, 13))
            S = derivative_matrix(s)
            @test maximum(abs, S + S') < 1e-10
        end
    end

    @testset "$(rpad("stiffness matrix",76))" begin
        for s in (SplineSpace(16, 3), LagrangeSpace(2, 13))
            K = stiffness_matrix(s)
            @test K ≈ K'
            @test maximum(abs, K * ones(nbasis(s))) < 1e-10   # constants in the kernel
        end
    end

    @testset "$(rpad("weighted and mixed assembly",76))" begin
        s = SplineSpace(16, 3)
        @test weighted_matrix(s, one, 0, 0) ≈ mass_matrix(s)
        @test mixed_matrix(s, 0, 0) ≈ mass_matrix(s)
        @test mixed_matrix(s, 0, 1) ≈ derivative_matrix(s)
        @test weighted_matrix(s, sin, 0, 1) ≈
              weighted_matrix(s, sin.(quadrature_nodes(s)), 0, 1)
        @test_throws DimensionMismatch weighted_matrix(s, [1.0], 0, 1)
    end

    @testset "$(rpad("projection and field evaluation",76))" begin
        for s in (SplineSpace(32, 3), LagrangeSpace(2, 16))
            û = project(s, sin)
            xs = collect(range(0, 2π, length = 61))
            @test maximum(abs, evaluate(s, û, xs) .- sin.(xs)) < 1e-3
            # projecting a field already in the space returns its own coefficients
            v̂ = randn(nbasis(s))
            @test project(s, field(s, v̂)) ≈ v̂
        end
    end

    @testset "$(rpad("Lagrange space is nodal",76))" begin
        # the defining property, and the one the sqrt transformation of the Burgers bracket
        # rests on: the coefficients ARE the values at the nodes
        s = LagrangeSpace(2, 13)
        f(x) = 2 + sin(x) + 0.3cos(2x)
        û = f.(nodes(s))
        @test maximum(abs, [evaluate(s, û, x) - f(x) for x in nodes(s)]) < 1e-12
    end

    @testset "$(rpad("nodal derivatives are averaged over the one-sided limits",76))" begin
        # a C⁰ basis has a jump in φ' exactly at the nodes; averaging is the only choice
        # that leaves the resulting coefficients antisymmetric
        s = LagrangeSpace(2, 13)
        E = GeometricBrackets.nodal_derivative_matrix(s)
        @test size(E) == (nbasis(s), nbasis(s))
        # E differentiates a periodic field: f'(x_m) = Σ_q f_q φ_q'(x_m) = Σ_q f[q] E[q,m].
        # A linear test function would be the wrong probe here -- it is not periodic, and
        # what it would measure is the jump at the seam rather than the derivative.
        x = nodes(s)
        @test maximum(abs, [sum(E[q, m] * sin(x[q]) for q in eachindex(x)) - cos(x[m])
                            for m in eachindex(x)]) < 0.05
        # the averaging is what keeps the coefficients antisymmetric; a one-sided rule
        # would not
        A = [dot(basis_integrals(s) .* E[:, m], ones(nbasis(s))) for m in eachindex(x)]
        @test length(A) == nbasis(s)
    end
end
