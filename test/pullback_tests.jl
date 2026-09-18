using PoissonBrackets
using LinearAlgebra
# By name, never a bare `using SimpleSplines`: every test file is included into the same
# `Main`, and `stiffness_matrix`, `weighted_matrix` and `derivative_matrix` are each a
# *different* generic in the two packages. A bare `using` here makes those names ambiguous in
# every test file included after this one, which is a failure in somebody else's testset.
using SimpleSplines: UniformMesh, Periodic, Dirichlet, (..)
using Test

@testset "$(rpad("Pullback Tests",80))" begin

    # Polar coordinates on an annulus: the same coordinate map as the polar disk with the pole
    # cut out, so a tensor-product space carries it and every quantity has a closed form.
    R₀, R₁ = 0.4, 1.3
    F(x) = (x[1] * cos(x[2]), x[1] * sin(x[2]))
    DF(x) = [cos(x[2]) -x[1]*sin(x[2]); sin(x[2]) x[1]*cos(x[2])]

    s = TensorSplineSpace((UniformMesh(12, R₀ .. R₁), UniformMesh(24, 0 .. 2π)), 3,
        (Dirichlet(), Periodic()))
    x̂ = quadrature_nodes(s)
    w = quadrature_weights(s)
    pb = PulledBack(s, F, DF)

    @testset "$(rpad("the measure is |det J|, pointwise and in the integral",76))" begin
        @test length(pb) == length(w)
        @test ndims(pb) == 2
        @test eltype(pb) == Float64

        # |det J| = r for polar coordinates.
        @test all(abs(measure(pb)[q] - x̂[q][1]) < 1e-14 for q in eachindex(w))
        @test dot(w, measure(pb)) ≈ π * (R₁^2 - R₀^2)

        # The physical nodes are F of the parameter nodes, which is what a physical coefficient
        # is sampled at. Getting this wrong is silent: both are vectors of pairs.
        @test all(nodes(pb)[q] == F(x̂[q]) for q in eachindex(w))
        @test all(hypot(nodes(pb)[q]...) ≈ x̂[q][1] for q in eachindex(w))
    end

    @testset "$(rpad("the metric is J⁻¹ A J⁻ᵀ times the measure",76))" begin
        𝔻 = metric(pb)
        @test size(𝔻) == (2, 2)
        # m J⁻¹J⁻ᵀ = r diag(1, r⁻²) = diag(r, 1/r), the polar Laplacian's weak form.
        @test all(abs(𝔻[1, 1][q] - x̂[q][1]) < 1e-13 for q in eachindex(w))
        @test all(abs(𝔻[2, 2][q] - 1 / x̂[q][1]) < 1e-13 for q in eachindex(w))
        @test maximum(abs, 𝔻[1, 2]) < 1e-13
        @test maximum(abs, 𝔻[2, 1]) < 1e-13

        # The congruence preserves symmetry and definiteness, whatever the map does. A
        # non-diagonal A, so that a transposed J⁻¹ would show.
        A = [2.0 0.5; 0.5 1.0]
        pa = PulledBack(s, F, DF; tensor = _ -> A)
        𝔸 = metric(pa)
        @test all(abs(𝔸[1, 2][q] - 𝔸[2, 1][q]) < 1e-13 for q in eachindex(w))
        @test all(eigmin([𝔸[1, 1][q] 𝔸[1, 2][q]; 𝔸[2, 1][q] 𝔸[2, 2][q]]) > 0
        for q in eachindex(w))
    end

    @testset "$(rpad("the density is composed with the map, not with the identity",76))" begin
        # ρ = 1/|x| is the Grad-Shafranov weight in its own physical coordinates. Pulled back
        # onto the polar chart it cancels |det J| = r exactly, so the measure comes back as one.
        pρ = PulledBack(s, F, DF; density = x -> 1 / hypot(x[1], x[2]))
        @test maximum(abs, measure(pρ) .- 1) < 1e-13

        # A density given as samples must be taken as given.
        ρ = rand(length(w)) .+ 1
        pv = PulledBack(s, F, DF; density = ρ)
        @test all(abs(measure(pv)[q] - ρ[q] * x̂[q][1]) < 1e-13 for q in eachindex(w))
        @test_throws DimensionMismatch PulledBack(s, F, DF; density = ρ[1:(end - 1)])
    end

    @testset "$(rpad("the pulled-back stiffness is the Laplacian, and controls are not",76))" begin
        u(x) = (x[1] - R₀) * (R₁ - x[1]) * cos(2 * x[2])
        function Δu(x)
            r, θ = x[1], x[2]
            g = (r - R₀) * (R₁ - r)
            (-2.0 + (R₀ + R₁ - 2r) / r - 4g / r^2) * cos(2θ)
        end

        û = project(s, u.(x̂))
        v̂ = project(s, [u(pt) * (1 + 0.3 * sin(3 * pt[2])) for pt in x̂])

        exact = -dot(w .* measure(pb), (basis_values(s, (0, 0))' * v̂) .* Δu.(x̂))
        got = dot(v̂, tensor_weighted_matrix(s, metric(pb)) * û)
        @test abs(got - exact) / abs(exact) < 1e-5

        # CONTROL: the plain parameter-square stiffness is a different operator. Qualified
        # because `stiffness_matrix` is a separate generic in each of the two packages loaded.
        plain = dot(v̂, PoissonBrackets.stiffness_matrix(s) * û)
        @test abs(plain - exact) / abs(exact) > 1e-2

        # CONTROL: the measure kept and the metric dropped — the error an area check cannot
        # see, because the area never touches the metric.
        𝔹 = [k == l ? copy(measure(pb)) : zeros(length(w)) for k in 1:2, l in 1:2]
        dropped = dot(v̂, tensor_weighted_matrix(s, 𝔹) * û)
        @test abs(dropped - exact) / abs(exact) > 1e-2
    end

    @testset "$(rpad("the frame is J⁻ᵀ, and the volume element is |det J|",76))" begin
        𝔽 = frame(pb)

        # J⁻ᵀ for the polar chart is [cos θ  sin θ; −sin θ/r  cos θ/r], so ∇_x = J⁻ᵀ∇̂ is the
        # textbook ∂ₓ = cos θ ∂_r − (sin θ/r) ∂_θ.
        for q in [1, 37, 400, length(w)]
            r, θ = x̂[q]
            @test abs(𝔽[1, 1][q] - cos(θ)) < 1e-14
            @test abs(𝔽[1, 2][q] + sin(θ) / r) < 1e-14
            @test abs(𝔽[2, 1][q] - sin(θ)) < 1e-14
            @test abs(𝔽[2, 2][q] - cos(θ) / r) < 1e-14
        end

        # The frame is not symmetric here, which is what makes a transposed Jacobian a
        # different object rather than the same one.
        @test maximum(abs, 𝔽[1, 2] .- 𝔽[2, 1]) > 1

        # The volume element is the measure with the density left out, and the two are not
        # the same vector wherever the density is not one.
        @test all(abs(volume_element(pb)[q] - x̂[q][1]) < 1e-14 for q in eachindex(w))
        pbρ = PulledBack(s, F, DF; density = x -> 1 / hypot(x[1], x[2]))
        @test volume_element(pbρ) ≈ volume_element(pb)
        @test maximum(abs, measure(pbρ) .- volume_element(pbρ)) > 0.1
        @test all(abs(measure(pbρ)[q] - 1) < 1e-13 for q in eachindex(w))
    end

    @testset "$(rpad("jacobian_residual catches a wrong Jacobian",76))" begin
        @test jacobian_residual(F, DF, x̂[1:200]) < 1e-8

        # A transposed Jacobian — the slip the helper exists to catch, and one that leaves the
        # determinant, and therefore every area, untouched.
        DFᵀ(x) = permutedims(DF(x))
        @test abs(det(DFᵀ(x̂[1])) - det(DF(x̂[1]))) < 1e-14
        @test jacobian_residual(F, DFᵀ, x̂[1:200]) > 1e-2

        @test_throws DimensionMismatch PulledBack(s, F, x -> ones(3, 3))
    end
end
