using PoissonBrackets
using LinearAlgebra
using Random
using SimpleSplines: UniformMesh, Dirichlet
using Test

# A bracket that answers only `metric_matrix`. The claim of `MetricBracket` is that the
# interface is three methods and everything else is generic, and the only way to test that
# claim is with a type that implements less than the two concrete brackets do.
struct MinimalMetricBracket{T} <: MetricBracket{T}
    G::Matrix{T}
end

PoissonBrackets.metric_matrix(b::MinimalMetricBracket, û::AbstractVector) = b.G

# The central-difference Jacobian of `metric_matrix`, as the tensor `dG[m,i,j]`.
function finite_difference_derivative(b::MetricBracket, û::AbstractVector, ε::Real)
    N = length(û)
    dG = zeros(eltype(û), N, N, N)
    for m in 1:N
        e = zeros(N)
        e[m] = ε
        dG[m, :, :] = (metric_matrix(b, û + e) - metric_matrix(b, û - e)) / (2ε)
    end
    return dG
end

@testset "$(rpad("Metric Bracket Tests",80))" begin
    rng = MersenneTwister(0x4d3b7c11)

    @testset "$(rpad("the interface is THREE methods and the rest is generic",76))" begin
        N = 8
        A = randn(rng, N, N)
        G = A' * A                      # symmetric positive definite by construction
        b = MinimalMetricBracket(G)
        û = randn(rng, N)
        c = randn(rng, N)

        @test b isa MetricBracket{Float64}
        @test eltype(b) == Float64
        # `metric_apply` falls back to the matrix product with nothing else defined
        @test metric_apply(b, û, c) == G * c
        @test issymmetric(b, û)
        @test ispositive_semidefinite(b, û)
        # and it is not degenerate on anything, which is what makes the residual meaningful
        @test degeneracy_residual(b, û, c) > 1e-3

        # the TWO-argument form is the documented exception to "three methods and the rest is
        # generic": it needs `space` and the generator hook as well. A bracket supplying
        # neither is refused by name, not with a FieldError about a field the interface never
        # asked it to have
        @test_throws MethodError degeneracy_residual(b, û)

        # the negative half: an indefinite matrix is rejected, a non-symmetric one too
        shifted = G - 2 * maximum(eigvals(G)) * Matrix(I, N, N)
        @test !ispositive_semidefinite(MinimalMetricBracket(shifted), û)
        @test !issymmetric(MinimalMetricBracket(A), û)
    end

    @testset "$(rpad("the double bracket is SYMMETRIC because X_h ⊗ X_h is",76))" begin
        s = TensorSplineSpace((6, 6), 3)
        N = nbasis(s)
        # a variable h, deliberately: the symmetry of ∫∂₁Φ_K ∂₂Φ_L is accidental on a
        # periodic axis with a CONSTANT coefficient, so a constant one would make this
        # testset vacuous
        ĥ = project(s, p -> cos(p[1])^2 * sin(p[2])^2 + 0.3sin(p[1]))
        b = DoubleBracket(s, ĥ)
        û = randn(rng, N)

        @test b isa MetricBracket{Float64}
        @test size(b) == (N, N)
        @test space(b) === s
        G = metric_matrix(b, û)
        @test maximum(abs, G - G') < 1e-14 * maximum(abs, G)
        @test issymmetric(b, û)
        # the sandwich does not depend on û at all here — h is prescribed
        @test metric_matrix(b, randn(rng, N)) ≈ G

        # the control: the same variable field in a coefficient that is NOT symmetric gives
        # a matrix that is not either, so the number above measures X_h ⊗ X_h and not the
        # assembly quietly symmetrising
        X = PoissonBrackets.hamiltonian_field(s, ĥ)
        𝔻 = PoissonBrackets._outer(X, X)
        𝔻[1, 2] = zeros(length(𝔻[1, 2]))
        A = tensor_weighted_matrix(s, 𝔻)
        @test maximum(abs, A - A') > 0.1 * maximum(abs, A)
    end

    @testset "$(rpad("the double bracket is positive SEMI-definite, never definite",76))" begin
        s = TensorSplineSpace((6, 6), 3)
        N = nbasis(s)
        ĥ = project(s, p -> cos(p[1])^2 * sin(p[2])^2 + 0.3sin(p[1]))
        b = DoubleBracket(s, ĥ)
        û = randn(rng, N)

        λ = eigvals(Symmetric(metric_matrix(b, û)))
        @test minimum(λ) > -1e-12 * maximum(λ)
        @test ispositive_semidefinite(b, û)
        # singular, not definite: the Hamiltonian is in the kernel by construction, so
        # `isposdef` is the wrong question and would fail
        @test minimum(λ) < 1e-12 * maximum(λ)

        # the control that must fail: X_h ⊗ X_h is a Gram matrix and cannot be indefinite,
        # so the coefficient has to be broken by hand — flipping one diagonal component
        X = PoissonBrackets.hamiltonian_field(s, ĥ)
        𝔻 = PoissonBrackets._outer(X, X)
        𝔻[1, 1] = -𝔻[1, 1]
        A = tensor_weighted_matrix(s, 𝔻)
        @test minimum(eigvals(Symmetric(Matrix(A)))) < -1e-3
    end

    @testset "$(rpad("the Hamiltonian is in the KERNEL, pointwise at the quadrature",76))" begin
        s = TensorSplineSpace((6, 6), 3)
        N = nbasis(s)
        ĥ = project(s, p -> cos(p[1])^2 * sin(p[2])^2 + 0.3sin(p[1]))
        b = DoubleBracket(s, ĥ)
        û = randn(rng, N)

        # X_h · ∇h vanishes at every quadrature point, so this is exact at the discrete
        # level rather than only in the continuum
        @test degeneracy_residual(b, û) < 1e-13
        @test maximum(abs, metric_apply(b, û, mass_matrix(s) * ĥ)) <
              1e-13 * maximum(abs, metric_matrix(b, û))
        # and it is degenerate on the Hamiltonian and on nothing else
        @test degeneracy_residual(b, û, randn(rng, N)) > 0.1

        # the control that must fail, and the reason `degeneracy_residual` exists: drop the
        # ⊥ and use ∇h ⊗ ∇h. It is still symmetric and still positive semi-definite —
        # a Gram matrix is a Gram matrix — and only the degeneracy notices.
        g = (field(s, ĥ, (1, 0)), field(s, ĥ, (0, 1)))
        A = tensor_weighted_matrix(s, PoissonBrackets._outer(g, g))
        @test maximum(abs, A - A') < 1e-14 * maximum(abs, A)
        @test minimum(eigvals(Symmetric(Matrix(A)))) > -1e-12 * maximum(abs, A)
        @test maximum(abs, A * ĥ) > 0.1 * maximum(abs, A) * maximum(abs, ĥ)
    end

    @testset "$(rpad("metric_apply agrees with the ASSEMBLED matrix, matrix-free",76))" begin
        s = TensorSplineSpace((6, 6), 3)
        N = nbasis(s)
        ĥ = project(s, p -> cos(p[1])^2 * sin(p[2])^2 + 0.3sin(p[1]))
        b = DoubleBracket(s, ĥ)
        û = randn(rng, N)
        c = randn(rng, N)

        G = metric_matrix(b, û)
        @test maximum(abs, metric_apply(b, û, c) - G * c) < 1e-12 * maximum(abs, G * c)
        # the sparse weak-form operator is the middle factor and nothing else
        @test metric_operator(b, û) ≈ mass_matrix(s) * G * mass_matrix(s)
        # a prescribed h makes the bracket constant, so its derivative vanishes identically
        @test metric_derivative(b, û) == zeros(N, N, N)
    end

    @testset "$(rpad("a LINEAR generating map keeps every property, and differentiates",76))" begin
        s = TensorSplineSpace((6, 6), 3)
        N = nbasis(s)
        M = Matrix(mass_matrix(s))
        # the screened Poisson solve, ψ̂ = (K + εM)⁻¹ M ω̂. The shift is what makes it
        # invertible on the periodic torus, where the constants are in the kernel of K;
        # MΛ stays symmetric, so Λ really is M⁻¹∇H of a Hamiltonian.
        Λ = (Matrix(stiffness_matrix(s)) + 0.5M) \ M
        b = DoubleBracket(s, Λ)
        û = randn(rng, N)
        c = randn(rng, N)

        G = metric_matrix(b, û)
        λ = eigvals(Symmetric(G))
        @test maximum(abs, G - G') < 1e-14 * maximum(abs, G)
        @test minimum(λ) > -1e-12 * maximum(λ)
        @test degeneracy_residual(b, û) < 1e-13
        @test maximum(abs, metric_apply(b, û, c) - G * c) < 1e-12 * maximum(abs, G * c)
        # unlike the prescribed-h case, this bracket really does depend on the state
        @test maximum(abs, metric_matrix(b, randn(rng, N)) - G) > 1e-3 * maximum(abs, G)

        # The analytic derivative against a central difference. The floor is set by the
        # difference scheme, not by the derivative: entries of order 10² differenced at
        # ε = 10⁻⁴ carry a relative error around 10⁻¹⁰, and any tighter tolerance would be
        # measuring the step size.
        dG = metric_derivative(b, û)
        fd = finite_difference_derivative(b, û, 1e-4)
        @test maximum(abs, dG - fd) < 1e-8 * maximum(abs, dG)
        # the control: the tolerance discriminates. A derivative wrong by one per cent —
        # far smaller than any plausible index or sign error — fails it.
        @test maximum(abs, 1.01 * dG - fd) > 1e-8 * maximum(abs, dG)
    end

    @testset "$(rpad("the projector bracket is a RANK-ONE correction to M⁻¹",76))" begin
        s = SplineSpace(16, 3)
        N = nbasis(s)
        φ̂ = project(s, sin)
        b = ProjectorBracket(s, φ̂)
        û = randn(rng, N)
        c = randn(rng, N)

        @test b isa MetricBracket{Float64}
        @test size(b) == (N, N)
        M = Matrix(mass_matrix(s))
        n = dot(φ̂, M * φ̂)
        G = metric_matrix(b, û)
        # no quadrature anywhere: this is the closed form, to the last bit of the inverse
        @test maximum(abs, G - (inv(M) - (φ̂ * φ̂') / n)) < 1e-12 * maximum(abs, G)
        @test rank(G) == N - 1

        λ = eigvals(Symmetric(G))
        @test maximum(abs, G - G') < 1e-14 * maximum(abs, G)
        @test minimum(λ) > -1e-12 * maximum(λ)
        @test ispositive_semidefinite(b, û)
        @test issymmetric(b, û)
        @test maximum(abs, metric_apply(b, û, c) - G * c) < 1e-12 * maximum(abs, G * c)
        @test metric_derivative(b, û) == zeros(N, N, N)
    end

    @testset "$(rpad("Π_H ANNIHILATES φ, and projecting onto φ does not",76))" begin
        s = SplineSpace(16, 3)
        N = nbasis(s)
        φ̂ = project(s, sin)
        b = ProjectorBracket(s, φ̂)
        û = randn(rng, N)
        c = randn(rng, N)
        M = Matrix(mass_matrix(s))

        @test maximum(abs, project_orthogonal(b, û, φ̂)) < 1e-14 * maximum(abs, φ̂)
        # idempotent, and L²-orthogonal to φ by construction
        Πc = project_orthogonal(b, û, c)
        @test maximum(abs, project_orthogonal(b, û, Πc) - Πc) < 1e-14 * maximum(abs, Πc)
        @test abs(dot(φ̂, M * Πc)) < 1e-13 * norm(φ̂) * norm(Πc)
        @test degeneracy_residual(b, û) < 1e-13
        @test degeneracy_residual(b, û, randn(rng, N)) > 1e-3

        # the control that must fail: the projector onto φ rather than onto its orthogonal
        # complement. It is symmetric and positive semi-definite just as the right one is,
        # and it is degenerate on everything EXCEPT the Hamiltonian.
        n = dot(φ̂, M * φ̂)
        G = (φ̂ * φ̂') / n
        @test maximum(abs, G - G') < 1e-14 * maximum(abs, G)
        @test minimum(eigvals(Symmetric(G))) > -1e-12 * maximum(abs, G)
        @test maximum(abs, G * (M * φ̂)) > maximum(abs, G) * maximum(abs, M * φ̂)

        # and the sign control: taking the correction twice over breaks positivity
        Gbad = inv(M) - 2 * (φ̂ * φ̂') / n
        @test minimum(eigvals(Symmetric(Gbad))) < -1e-3
    end

    @testset "$(rpad("the projector bracket differentiates ANALYTICALLY in 2D",76))" begin
        # a homogeneous-Dirichlet square, which is where §5 lives, and the Poisson solve
        # φ̂ = K⁻¹M û — invertible here, the constants no longer being in the kernel
        s = TensorSplineSpace(ntuple(_ -> UniformMesh(4, 1.0), 2), 2, Dirichlet())
        N = nbasis(s)
        Λ = Matrix(stiffness_matrix(s)) \ Matrix(mass_matrix(s))
        b = ProjectorBracket(s, Λ)
        û = randn(rng, N)
        c = randn(rng, N)

        G = metric_matrix(b, û)
        λ = eigvals(Symmetric(G))
        @test maximum(abs, G - G') < 1e-14 * maximum(abs, G)
        @test minimum(λ) > -1e-12 * maximum(λ)
        @test degeneracy_residual(b, û) < 1e-13
        @test maximum(abs, metric_apply(b, û, c) - G * c) < 1e-12 * maximum(abs, G * c)

        dG = metric_derivative(b, û)
        fd = finite_difference_derivative(b, û, 1e-5)
        @test maximum(abs, dG - fd) < 1e-8 * maximum(abs, dG)
        @test maximum(abs, 1.01 * dG - fd) > 1e-8 * maximum(abs, dG)
    end

    @testset "$(rpad("a generating field of the WRONG shape is rejected",76))" begin
        s = TensorSplineSpace((4, 4), 2)
        N = nbasis(s)
        @test_throws DimensionMismatch DoubleBracket(s, zeros(N - 1))
        @test_throws DimensionMismatch DoubleBracket(s, zeros(N, N - 1))
        @test_throws DimensionMismatch ProjectorBracket(s, zeros(N + 1))
        @test_throws DimensionMismatch ProjectorBracket(s, zeros(N, 2))
        # the double bracket is two-dimensional by definition — the ⊥ that carries its
        # degeneracy has no meaning on three axes, so there is no method rather than a
        # silently chosen plane
        @test_throws MethodError DoubleBracket(TensorSplineSpace((4, 4, 4), 2), zeros(64))
        # the projector bracket has no such restriction
        @test ProjectorBracket(TensorSplineSpace((4, 4, 4), 2), ones(64)) isa MetricBracket
    end

    @testset "$(rpad("ispositive_semidefinite is RELATIVE, and fails closed",76))" begin
        # The predicate answers a question about a SIGN, so its tolerance is relative to the
        # largest eigenvalue. A floor of one would make it absolute for any bracket scaled
        # below that -- and absolute in the direction that passes, which is the wrong way for
        # a check to be wrong.
        for scale in (1e2, 1.0, 1e-6, 1e-12)
            indefinite = MinimalMetricBracket(scale .* Matrix(Diagonal([1.0, -1.0])))
            @test !ispositive_semidefinite(indefinite, zeros(2))
            @test ispositive_semidefinite(
                MinimalMetricBracket(scale .* Matrix(Diagonal([1.0, 0.0]))), zeros(2))
        end
        # a genuinely singular semi-definite matrix with round-off in the kernel still passes
        @test ispositive_semidefinite(
            MinimalMetricBracket([1.0 0.0; 0.0 -1e-14]), zeros(2))
        # the zero bracket is semi-definite; one with no positive eigenvalue at all is not
        @test ispositive_semidefinite(MinimalMetricBracket(zeros(2, 2)), zeros(2))
        @test !ispositive_semidefinite(
            MinimalMetricBracket(Matrix(Diagonal([-1.0, -2.0]))), zeros(2))
    end
end
