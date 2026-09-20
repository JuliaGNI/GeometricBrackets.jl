using GeometricBrackets
using LinearAlgebra
using Random
using SimpleSplines: UniformMesh, GradedMesh, Periodic, Dirichlet, Free, KroneckerMass,
                     quadratures
using SparseArrays
using Test

# The component form of a tensor coefficient. Written out rather than as `[a b; c d]`,
# which concatenates vector entries into one long matrix of numbers instead of building the
# 2x2 matrix OF vectors the assembly wants.
function tensor_coefficient(a, b, c, d)
    𝔻 = Matrix{Vector{Float64}}(undef, 2, 2)
    𝔻[1, 1], 𝔻[1, 2], 𝔻[2, 1], 𝔻[2, 2] = a, b, c, d
    return 𝔻
end

@testset "$(rpad("Tensor Spline Space Tests",80))" begin
    rng = MersenneTwister(0x2c7f1a95)

    @testset "$(rpad("the accessors are PER-AXIS tuples, never a scalar",76))" begin
        s = TensorSplineSpace((8, 12), (3, 2))
        @test s isa DiscreteSpace{Float64}
        @test ndims(s) == 2
        @test size(s) == (8, 12)
        @test nbasis(s) == 96
        @test length(s) == 96
        @test eltype(s) == Float64
        @test degree(s) == (3, 2)
        @test order(s) == (4, 3)
        @test ncells(s) == (8, 12)
        # not a scalar, and not the volume masquerading as a length
        @test domainlength(s) == (2π, 2π)
        @test domainvolume(s) ≈ (2π)^2
        @test length(nodes(s)) == 2
        @test length(quadrature_nodes(s)) == length(quadrature_weights(s))
        @test eltype(quadrature_nodes(s)) == NTuple{2, Float64}
        @test sum(quadrature_weights(s)) ≈ (2π)^2
        @test size(basis_values(s, (0, 0))) == (96, length(quadrature_nodes(s)))
        @test basis_values(s, (0, 0)) isa SparseMatrixCSC
        # a per-axis argument of the wrong length is a mistake, not something to recycle
        @test_throws DimensionMismatch TensorSplineSpace((8, 8), (3, 3, 3))
        @test_throws DimensionMismatch TensorSplineSpace((8, 8), 3, (Periodic(),))
    end

    @testset "$(rpad("a scalar derivative order is REJECTED beyond zero",76))" begin
        s = TensorSplineSpace((8, 8), 3)
        # d = 0 stays meaningful, which is what keeps `project`, `basis_integrals` and
        # `field` the generic implementations of spaces.jl
        @test basis_values(s, 0) === basis_values(s, (0, 0))
        @test basis_values(s) === basis_values(s, (0, 0))
        # anything else would have to guess an axis
        @test_throws ArgumentError basis_values(s, 1)
        @test_throws ArgumentError basis_values(s, 2)
        # and so do the two matrices of `spaces.jl` that are built from a scalar order
        @test_throws ArgumentError derivative_matrix(s)
        @test_throws ArgumentError mixed_matrix(s, 0, 1)
        # the multi-index form is memoised, so the second call returns the SAME object
        Φ = basis_values(s, (1, 0))
        @test Φ === basis_values(s, (1, 0))
        @test Φ !== basis_values(s, (0, 1))
        # only what the one-dimensional quadratures tabulated exists
        @test_throws ArgumentError basis_values(s, (4, 0))
    end

    @testset "$(rpad("the mass matrix reproduces ∫1 = |Ω| on the PERIODIC torus only",76))" begin
        s = TensorSplineSpace((8, 12), (3, 2))
        M = mass_matrix(s)
        N = nbasis(s)
        @test M isa SparseMatrixCSC
        @test M ≈ M'
        @test isposdef(Matrix(M))
        # the Kronecker mass and the flattened Φ diag(w) Φᵀ are the same matrix, which is
        # what the first-axis-fastest flattening convention buys
        @test maximum(abs, M - mixed_matrix(s, (0, 0), (0, 0))) < 1e-14
        # a periodic B-spline basis is a partition of unity, so the coefficient vector of
        # the constant one is all ones and 1ᵀM1 = ∫1 = |Ω|
        @test dot(ones(N), M * ones(N)) ≈ domainvolume(s)
        @test sum(M) ≈ domainvolume(s)
        @test M * ones(N) ≈ basis_integrals(s)
        @test sum(basis_integrals(s)) ≈ domainvolume(s)

        # the negative half: this is a property of the periodic basis, not of the assembly.
        # A homogeneous-Dirichlet basis cannot represent the constant at all, and its mass
        # matrix sums to strictly less.
        sd = TensorSplineSpace(ntuple(_ -> UniformMesh(8, 1.0), 2), 3, Dirichlet())
        @test sum(mass_matrix(sd)) < 0.9 * domainvolume(sd)
    end

    @testset "$(rpad("L² projection is EXACT for polynomials of degree ≤ p",76))" begin
        # on a clamped basis: a periodic one spans no polynomial but the constant, and a
        # Dirichlet one spans none at all
        s = TensorSplineSpace(ntuple(_ -> UniformMesh(6, 1.0), 2), 3, Free())
        pts = [(0.13, 0.77), (0.5, 0.5), (0.91, 0.24)]
        for g in (x -> 1.0, x -> x[1]^3, x -> x[1]^2 * x[2], x -> x[1]^3 * x[2]^3)
            û = project(s, g)
            @test maximum(abs, [evaluate(s, û, p) - g(p) for p in pts]) < 1e-13
        end
        # the negative half: degree p+1 on an axis is NOT reproduced, so the tolerance
        # above is measuring exactness rather than a fortunate mesh
        û = project(s, x -> x[1]^4)
        @test maximum(abs, [evaluate(s, û, p) - p[1]^4 for p in pts]) > 1e-6

        # a field already in the space projects onto its own coefficients
        v̂ = randn(rng, nbasis(s))
        @test project(s, field(s, v̂)) ≈ v̂
        # and the in-place form agrees with the out-of-place one
        w = similar(v̂)
        project!(w, s, field(s, v̂))
        @test w ≈ v̂
    end

    @testset "$(rpad("projection of a periodic field CONVERGES at order p+1",76))" begin
        # not spectrally: this is a spline space, and saying so is the point of measuring
        # the rate rather than asserting a tolerance
        f(x) = sin(x[1]) * cos(2x[2])
        pts = [(0.3, 1.1), (2.0, 4.0), (5.5, 0.2)]
        errs = map((16, 32)) do n
            s = TensorSplineSpace((n, n), 3)
            û = project(s, f)
            maximum(abs, [evaluate(s, û, p) - f(p) for p in pts])
        end
        @test errs[2] < errs[1]
        @test log2(errs[1] / errs[2]) > 3.5
        @test errs[2] < 1e-4
    end

    @testset "$(rpad("the stiffness spectrum APPROXIMATES k₁² + k₂² on the torus",76))" begin
        n = 12
        s = TensorSplineSpace((n, n), 3)
        K = Matrix(stiffness_matrix(s))
        M = Matrix(mass_matrix(s))
        @test maximum(abs, K - K') < 1e-14
        # the constants are the kernel of -Δ on the torus
        @test maximum(abs, K * ones(nbasis(s))) < 1e-12

        λ = sort(real.(eigvals(Symmetric(K), Symmetric(M))))
        # the N smallest values of |k|² on the lattice, in the same order as λ
        exact = sort(vec([k1^2 + k2^2 for k1 in (-n):n, k2 in (-n):n]))[1:nbasis(s)]
        @test abs(λ[1]) < 1e-10
        @test maximum(abs(λ[i] - exact[i]) / exact[i] for i in 2:25) < 1e-3
        # the negative half: the unresolved end of the spectrum does NOT track |k|², so
        # the bound above is a statement about the resolved modes and not about eigvals
        @test abs(λ[end] - exact[end]) / exact[end] > 0.5
    end

    @testset "$(rpad("a Dirichlet space VANISHES on the boundary, with λ₁ = 2π²",76))" begin
        s = TensorSplineSpace(ntuple(_ -> UniformMesh(8, 1.0), 2), 3, Dirichlet())
        @test nbasis(s) == 81
        @test domainlength(s) == (1.0, 1.0)
        # every basis function vanishes on ∂Ω, so every field in the space does
        û = randn(rng, nbasis(s))
        edge = [(0.0, t) for t in 0:0.25:1]
        edge = vcat(edge, [(1.0, t) for t in 0:0.25:1], [(t, 0.0) for t in 0:0.25:1])
        @test maximum(abs, [evaluate(s, û, p) for p in edge]) < 1e-14

        # the first Dirichlet eigenvalue of -Δ on the unit square, λ₁₁ = 2π² — the
        # reference the reduced-Euler relaxation of §5.4 is checked against
        λ = eigvals(Symmetric(Matrix(stiffness_matrix(s))),
            Symmetric(Matrix(mass_matrix(s))))
        @test abs(minimum(real.(λ)) - 2π^2) / (2π^2) < 1e-5
    end

    @testset "$(rpad("the tensor-coefficient assembly with 𝔻 = I IS the stiffness matrix",76))" begin
        s = TensorSplineSpace((8, 8), 3)
        K = stiffness_matrix(s)
        Q = length(quadrature_weights(s))
        one_ = ones(Q)
        zero_ = zeros(Q)

        # all four ways of handing the same coefficient in agree, to round-off
        @test maximum(abs, tensor_weighted_matrix(s, Matrix(1.0I, 2, 2)) - K) < 1e-14
        @test maximum(abs, tensor_weighted_matrix(s, x -> Matrix(1.0I, 2, 2)) - K) < 1e-14
        @test maximum(abs, tensor_weighted_matrix(s, [Matrix(1.0I, 2, 2) for _ in 1:Q]) -
                           K) <
              1e-14
        @test maximum(
            abs, tensor_weighted_matrix(s, tensor_coefficient(one_, zero_, zero_, one_)) -
                 K) <
              1e-14

        # 𝔻 = diag(1,0) is the ∂₁-only block and nothing else
        @test maximum(abs, tensor_weighted_matrix(s, [1.0 0.0; 0.0 0.0]) -
                           mixed_matrix(s, (1, 0), (1, 0))) < 1e-14

        @test_throws DimensionMismatch tensor_weighted_matrix(s, Matrix(1.0I, 3, 3))
        @test_throws DimensionMismatch tensor_weighted_matrix(s, [Matrix(1.0I, 2, 2)])
        @test_throws DimensionMismatch weighted_matrix(s, [1.0], (1, 0), (0, 1))
    end

    @testset "$(rpad("an anisotropic 𝔻 gives a DIFFERENT matrix from the isotropic one",76))" begin
        # the control that can fail: if the k,l sum were dropped or the components were
        # summed with equal weights, every one of the assertions above would still pass
        s = TensorSplineSpace((8, 8), 3)
        A_iso = tensor_weighted_matrix(s, Matrix(1.0I, 2, 2))
        A_ani = tensor_weighted_matrix(s, [2.0 0.0; 0.0 1.0])
        @test maximum(abs, A_ani - A_iso) > 0.1 * maximum(abs, A_iso)
        # and it is the ∂₁ block that moved, by exactly the extra unit of weight
        @test maximum(abs, A_ani - A_iso - mixed_matrix(s, (1, 0), (1, 0))) < 1e-14

        # off-diagonal components are not the diagonal ones either
        A_off = tensor_weighted_matrix(s, [0.0 1.0; 1.0 0.0])
        @test maximum(abs, A_off) > 1e-3
        @test maximum(abs, A_off - A_iso) > 0.1 * maximum(abs, A_iso)
    end

    @testset "$(rpad("symmetry of the assembly follows 𝔻, and is NOT automatic",76))" begin
        s = TensorSplineSpace((10, 10), 3)
        x = quadrature_nodes(s)
        f = [1.0 + 0.5sin(p[1]) * cos(p[2]) for p in x]
        z = zeros(length(f))

        A_sym = tensor_weighted_matrix(s, tensor_coefficient(f, z, z, f))
        @test maximum(abs, A_sym - A_sym') < 1e-14

        # a coefficient that is not symmetric must produce a matrix that is not either;
        # a CONSTANT one would not settle this, because the two antisymmetries of
        # ∫φ' φ on a periodic axis cancel and ∫∂₁φ_K ∂₂φ_L is symmetric by accident
        A_ns = tensor_weighted_matrix(s, tensor_coefficient(z, f, z, z))
        @test maximum(abs, A_ns - A_ns') > 0.1 * maximum(abs, A_ns)
        @test maximum(abs,
            tensor_weighted_matrix(s, [0.0 1.0; 0.0 0.0]) -
            tensor_weighted_matrix(s, [0.0 1.0; 0.0 0.0])') < 1e-14

        # positive semi-definiteness likewise: it is a property of 𝔻, checked here so that
        # the metric brackets can rest on it
        g1 = [1.0 + 0.5sin(p[1]) for p in x]
        g2 = [0.3cos(p[2]) for p in x]
        g3 = [2.0 + 0.5cos(p[1]) for p in x]
        A_psd = tensor_weighted_matrix(s, tensor_coefficient(g1, g2, g2, g3))
        @test minimum(eigvals(Symmetric(Matrix(A_psd)))) > -1e-12
        A_ind = tensor_weighted_matrix(s, tensor_coefficient(-g1, g2, g2, g3))
        @test minimum(eigvals(Symmetric(Matrix(A_ind)))) < -1e-2
    end

    @testset "$(rpad("the mass solve is KRONECKER-factored and M⁻¹ is not stored",76))" begin
        s = TensorSplineSpace((12, 12), 3)
        N = nbasis(s)
        @test mass_factorization(s) isa KroneckerMass
        û = randn(rng, N)
        @test mass_factorization(s) \ (mass_matrix(s) * û) ≈ û
        # M⁻¹ is the exact Kronecker product of the per-axis inverses, and it is rebuilt on
        # every call rather than cached — a new array each time is the observable half of
        # "not stored"
        Minv = inverse_mass_matrix(s)
        @test Minv * Matrix(mass_matrix(s)) ≈ I
        @test Minv !== inverse_mass_matrix(s)
        @test size(Minv) == (N, N)
    end

    @testset "$(rpad("per-axis meshes and boundary conditions MIX on one space",76))" begin
        # the §5 case in one call site: periodic on one axis, Dirichlet on the other
        s = TensorSplineSpace((UniformMesh(8, 2π), UniformMesh(6, 1.0)), (3, 2),
            (Periodic(), Dirichlet()))
        @test degree(s) == (3, 2)
        @test domainlength(s) == (2π, 1.0)
        @test size(s) == (8, 6)      # 8 periodic; 6 + 2 clamped, less the two ends
        û = randn(rng, nbasis(s))
        # Dirichlet on the second axis only
        @test maximum(abs, [evaluate(s, û, (t, 0.0)) for t in 0:1.0:6]) < 1e-14
        @test maximum(abs, [evaluate(s, û, (0.0, t)) for t in 0.2:0.2:0.8]) > 1e-8

        # a non-uniform mesh on one axis changes nothing structurally
        sg = TensorSplineSpace((GradedMesh(8, 2π), UniformMesh(8, 2π)), 3)
        @test sum(mass_matrix(sg)) ≈ domainvolume(sg)
    end

    @testset "$(rpad("at D = 1 the space IS the SplineSpace, to the last bit",76))" begin
        # The reference is the one-dimensional space itself rather than a recomputation of
        # the tensor assembly, so the two code paths are compared and not one of them with
        # itself. They agree exactly, not approximately: on one axis the Kronecker product
        # has a single factor and no arithmetic happens.
        s = TensorSplineSpace((10,), 3)
        r = SplineSpace(10, 3)
        @test ndims(s) == 1
        @test nbasis(s) == nbasis(r) == 10
        @test degree(s) == (3,)
        @test domainvolume(s) ≈ domainlength(s)[1]
        @test Matrix(mass_matrix(s)) == Matrix(mass_matrix(r))
        @test Matrix(stiffness_matrix(s)) == Matrix(stiffness_matrix(r))
        @test Matrix(basis_values(s, (0,))) == Matrix(basis_values(r, 0))
        @test Matrix(basis_values(s, (1,))) == Matrix(basis_values(r, 1))
        @test Matrix(mixed_matrix(s, (1,), (1,))) == Matrix(stiffness_matrix(r))
        @test Matrix(tensor_weighted_matrix(s, Matrix(1.0I, 1, 1))) ==
              Matrix(stiffness_matrix(r))

        û = randn(rng, nbasis(s))
        @test field(s, û, (0,)) == field(r, û, 0)
        @test inverse_mass_matrix(s) * Matrix(mass_matrix(s)) ≈ I

        # the memo is one object per multi-index, and on one axis it is a copy rather than
        # the quadrature's own table — see the warning on `basis_values`
        @test basis_values(s, (0,)) === basis_values(s, (0,))
        @test mixed_matrix(s, (1,), (1,)) === mixed_matrix(s, (1,), (1,))
        q = quadratures(s.quadrature)[1]
        @test basis_values(s, (1,)) == basis_values(q, 1)
        @test basis_values(s, (1,)) !== basis_values(q, 1)

        # a metric bracket built on it degenerates exactly as it does on two axes
        b = ProjectorBracket(s, project(s, x -> sin(x[1])))
        @test degeneracy_residual(b, û) < 1e-14
    end

    @testset "$(rpad("everything above works UNCHANGED in three dimensions",76))" begin
        s = TensorSplineSpace((4, 4, 4), 2)
        @test ndims(s) == 3
        @test nbasis(s) == 64
        @test domainlength(s) == (2π, 2π, 2π)
        @test domainvolume(s) ≈ (2π)^3
        @test sum(mass_matrix(s)) ≈ domainvolume(s)
        K = stiffness_matrix(s)
        @test maximum(abs, K - K') < 1e-14
        @test maximum(abs, K * ones(nbasis(s))) < 1e-12
        @test maximum(abs, tensor_weighted_matrix(s, Matrix(1.0I, 3, 3)) - K) < 1e-14
        @test_throws ArgumentError basis_values(s, 1)
        û = project(s, x -> sin(x[1]) * cos(x[2]) * sin(x[3]))
        @test length(û) == 64
        @test_throws DimensionMismatch evaluate(s, û[1:10], (0.1, 0.2, 0.3))
    end
end
