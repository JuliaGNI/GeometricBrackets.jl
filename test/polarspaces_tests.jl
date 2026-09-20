using PoissonBrackets
using LinearAlgebra
using Random
using SparseArrays
# By name, never a bare `using SimpleSplines`: the test files share one `Main`, and
# `stiffness_matrix`, `weighted_matrix` and `derivative_matrix` are each a different generic
# in the two packages.
using SimpleSplines: UniformMesh, BSplineBasis, PeriodicBSplineBasis, PolarSplineBasis,
                     bases, (..)
using Test

@testset "$(rpad("Polar Spline Space Tests",80))" begin
    s = PolarSplineSpace((6, 12), 3)
    N = nbasis(s)
    Q = length(quadrature_weights(s))

    @testset "$(rpad("the interface, and the shapes the generic assemblies need",76))" begin
        @test s isa DiscreteSpace
        @test ndims(s) == 2
        @test eltype(s) == Float64
        @test N == length(s)
        @test degree(s) == (3, 3)
        @test ncells(s) == (6, 12)
        @test domainlength(s) == (1.0, 2π)
        @test domainvolume(s) ≈ 2π
        @test pole(s) == 0.0
        @test size(pole_triangle(s)) == (3, 2)
        @test pseudo_cartesian(s, (0.0, 1.1)) == (0.0, 0.0)

        # The three things every generic assembly of `spaces.jl` is written against.
        @test quadrature_nodes(s) isa Vector{NTuple{2, Float64}}
        @test length(quadrature_nodes(s)) == Q
        @test size(basis_values(s, (0, 0))) == (N, Q)
        @test basis_values(s) === basis_values(s, (0, 0))
        @test_throws ArgumentError basis_values(s, 1)

        # Built the same from a basis, a pair of bases, or a cell count.
        B = PolarSplineBasis(BSplineBasis(UniformMesh(6, 0 .. 1), 3),
            PeriodicBSplineBasis(UniformMesh(12, 0 .. 2π), 3))
        @test nbasis(PolarSplineSpace(B)) == N
        @test nbasis(PolarSplineSpace(bases(basis(s))...)) == N
    end

    @testset "$(rpad("the mass matrix, the constants, and the parameter measure",76))" begin
        M = mass_matrix(s)
        𝟙 = ones(N)
        @test issymmetric(M)
        @test isposdef(Symmetric(Matrix(M)))
        @test dot(𝟙, M * 𝟙) ≈ 2π                 # the area of the parameter square
        @test M * 𝟙 ≈ basis_integrals(s)          # the basis is a partition of unity
        @test mass_factorization(s) === mass_operator(s)

        # `inverse_mass_matrix` is dense and has no Kronecker shortcut, but it is still the
        # inverse.
        @test inverse_mass_matrix(s) * Matrix(M) ≈ I atol=1e-9

        # The constants project exactly, at the pole and away from it.
        û = project(s, x -> 1.0)
        @test all(abs(evaluate(s, û, x) - 1) < 1e-11
        for x in [(0.0, 0.0), (0.0, 2.9), (0.5, 1.0), (1.0, 4.0)])

        # `field` is the tabulation, and `project` inverts it on a representable function.
        v̂ = project(s, x -> 1 - x[1]^2)
        @test maximum(abs, field(s, v̂, (0, 0)) .-
                           [1 - x[1]^2 for x in quadrature_nodes(s)]) <
              1e-11
    end

    @testset "$(rpad("the assembly entry points agree with their definitions",76))" begin
        w = quadrature_weights(s)
        Φ = basis_values(s, (0, 0))
        @test mixed_matrix(s, (0, 0), (0, 0)) ≈ mass_matrix(s)
        @test weighted_matrix(s, x -> 1.0, (0, 0), (0, 0)) ≈ mass_matrix(s)
        @test weighted_matrix(s, ones(Q), (0, 0), (0, 0)) ≈ mass_matrix(s)
        @test_throws DimensionMismatch weighted_matrix(s, ones(Q - 1), (0, 0), (0, 0))

        K = stiffness_matrix(s)
        @test norm(K * ones(N), Inf) < 1e-10      # the constants are its kernel
        @test K ≈ mixed_matrix(s, (1, 0), (1, 0)) + mixed_matrix(s, (0, 1), (0, 1))
        @test derivative_matrix(s, 1) ≈ mixed_matrix(s, (0, 0), (1, 0))

        # A constant tensor coefficient is the sum of the constant blocks.
        @test tensor_weighted_matrix(s, [1.0 0.0; 0.0 1.0]) ≈ K
        𝔻 = [k == l ? ones(Q) : zeros(Q) for k in 1:2, l in 1:2]
        @test tensor_weighted_matrix(s, 𝔻) ≈ K
        @test tensor_weighted_matrix(s, _ -> [1.0 0.0; 0.0 1.0]) ≈ K
        @test_throws DimensionMismatch tensor_weighted_matrix(s, ones(2, 3))
    end

    @testset "$(rpad("the pulled-back Laplacian on the whole disk, pole included",76))" begin
        # `u = 1 − s²` is radial, so it is in the polar space exactly; it vanishes on the rim,
        # and ∫|∇u|² over the unit disk is 2π in closed form. No tensor-product space can be
        # asked this at all, because it is not even C⁰ at the pole.
        F(x) = (x[1] * cos(x[2]), x[1] * sin(x[2]))
        DF(x) = [cos(x[2]) -x[1]*sin(x[2]); sin(x[2]) x[1]*cos(x[2])]
        sd = PolarSplineSpace((16, 32), 3)
        pb = PulledBack(sd, F, DF)

        û = project(sd, x -> 1 - x[1]^2)
        @test all(abs(evaluate(sd, û, x) - (1 - x[1]^2)) < 1e-11
        for x in [(0.0, 0.4), (0.3, 2.0), (1.0, 5.0)])

        K = tensor_weighted_matrix(sd, metric(pb))
        @test dot(û, K * û) ≈ 2π atol=1e-9
        @test dot(quadrature_weights(sd), measure(pb)) ≈ π      # the area of the disk

        # CONTROL: the plain stiffness is a different operator.
        @test abs(dot(û, stiffness_matrix(sd) * û) - 2π) / 2π > 1e-2

        # CONTROL: the metric dropped to |det J| times the identity. Invisible on a radial
        # field — the θθ component never appears — so it is checked on one with angular
        # structure, which is the reason a radial test alone would prove nothing here.
        𝔹 = [k == l ? copy(measure(pb)) : zeros(length(measure(pb))) for k in 1:2, l in 1:2]
        Kbad = tensor_weighted_matrix(sd, 𝔹)
        @test dot(û, Kbad * û) ≈ 2π atol=1e-9                   # invisible, as it must be
        ŵ = project(sd, x -> (1 - x[1]^2) * x[1]^2 * cos(2 * x[2]))
        @test abs(dot(ŵ, Kbad * ŵ) - dot(ŵ, K * ŵ)) / abs(dot(ŵ, K * ŵ)) > 1e-2
    end

    @testset "$(rpad("a CollisionBracket carries the flow on the polar space",76))" begin
        rng = MersenneTwister(0x70147)
        sb = PolarSplineSpace((6, 12), 2)
        M = Matrix(mass_matrix(sb))
        Λ = (Matrix(stiffness_matrix(sb)) + 0.5M) \ M
        b = CollisionBracket(sb, Λ)
        @test space(b) === sb

        û = randn(rng, nbasis(sb))
        G = metric_matrix(b, û)
        λ = eigvals(Symmetric(G))
        @test issymmetric(b, û)
        @test ispositive_semidefinite(b, û)
        @test minimum(λ) < 1e-11 * maximum(λ)            # singular, not definite
        @test degeneracy_residual(b, û) < 1e-12
        @test degeneracy_residual(b, û, randn(rng, nbasis(sb))) > 0.1

        # The mapped measure of `eq:mapping` is admissible — it does not depend on the outer
        # point — so every property survives it.
        #
        # The Jacobian here is differenced, which `PulledBack` exists to discourage. It is
        # acceptable *for this test and nowhere else*: what is under test is whether the
        # bracket stays symmetric, semi-definite and degenerate under a positive admissible
        # measure, and any such measure would do. An assembly whose numbers are the answer
        # needs the analytic Jacobian and `jacobian_residual` to confirm it, as
        # `scripts/verify_polar_bracket.jl` does.
        gs(x) = (q = sqrt(1 + 0.3 * (0.3 + 2x[1] * cos(x[2])));
            (4 * (3 + (1 - q) / 0.3),
                6.3 * 1.4 * x[1] * sin(x[2]) /
                (sqrt(1 - 0.3^2 / 4) * (2 - q))))
        dgs(x) = (h = 1e-6;
            hcat((collect(gs((x[1] + h, x[2]))) .- collect(gs((x[1] - h, x[2])))) ./ 2h,
                (collect(gs((x[1], x[2] + h))) .- collect(gs((x[1], x[2] - h)))) ./ 2h))
        pbgs = PulledBack(sb, gs, dgs; density = x -> 1 / x[1])
        bm = CollisionBracket(sb, Λ; density = measure(pbgs))
        @test issymmetric(bm, û)
        @test ispositive_semidefinite(bm, û)
        @test degeneracy_residual(bm, û) < 1e-12

        # CONTROL: `eq:M-condition` requires M > 0, and nothing enforces it. A sign-changing
        # mobility makes κ indefinite and the bracket is no longer a Gram matrix.
        bs = CollisionBracket(sb, Λ; mobility = (x, u) -> cos(6 * x[2]),
            mobility_derivative = 0)
        λs = eigvals(Symmetric(metric_matrix(bs, û)))
        @test minimum(λs) < -1e-6 * maximum(λs)
        @test !ispositive_semidefinite(bs, û)
    end
end
