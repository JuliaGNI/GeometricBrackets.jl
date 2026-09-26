using LinearAlgebra
using GeometricBrackets
using Random
using Test

@testset "$(rpad("Metriplectic Bracket Tests",80))" begin
    random_psd(rng, n, r) = (P = randn(rng, n, r); P * P')

    @testset "$(rpad("the Kulkarni-Nomizu tensor has the symmetries it is built for",76))" begin
        rng = MersenneTwister(20260820)
        n = 4
        σ, μ = random_psd(rng, n, n), random_psd(rng, n, n)
        R = kulkarni_nomizu(σ, μ)
        @test size(R) == (n, n, n, n)
        # compared as whole arrays against a tolerance, not entry by entry against `≈`: the
        # symmetries are exact in algebra but the two sides sum the same four products in a
        # different order, and entries that should vanish land at 1e-16 rather than at zero,
        # where `≈` demands equality.
        scale = maximum(abs, R)
        @test maximum(abs, R .+ permutedims(R, (2, 1, 3, 4))) < 1e-12scale   # first pair
        @test maximum(abs, R .+ permutedims(R, (1, 2, 4, 3))) < 1e-12scale   # second pair
        @test maximum(abs, R .- permutedims(R, (3, 4, 1, 2))) < 1e-12scale   # pair exchange
        # symmetric in its two generators, so the bracket does not depend on which is which
        @test kulkarni_nomizu(μ, σ) ≈ R
    end

    @testset "$(rpad("the closed form IS the contraction of the tensor",76))" begin
        # eq. (2.6) evaluated in O(n^2) against the O(n^4) definition it implements. Without
        # this the positivity sweep would be evidence about a rearrangement, not the bracket.
        rng = MersenneTwister(11)
        for n in (2, 3, 4, 5), _ in 1:5

            σ, μ = random_psd(rng, n, n), random_psd(rng, n, n)
            α, h = randn(rng, n), randn(rng, n)
            @test metriplectic_bracket(σ, μ, α, h) ≈
                  metriplectic_bracket(kulkarni_nomizu(σ, μ), α, h)
        end
    end

    @testset "$(rpad("Proposition 2.1 holds, including at deficient rank",76))" begin
        # The full sweep of 800000 draws is `scripts/verify_fourbracket_metriplectic.jl`; this
        # is the same test cut to a size a test suite can afford. The rank-deficient pairs are
        # kept, because they are where an inequality chain is likeliest to be lost.
        rng = MersenneTwister(4711)
        for n in (2, 3, 5), (rs, rm) in ((n, n), (1, n), (n, 1), (1, 1), (max(n - 1, 1), 1))

            worst = Inf
            for _ in 1:2000
                worst = min(worst,
                    metriplectic_bracket(random_psd(rng, n, rs),
                        random_psd(rng, n, rm),
                        randn(rng, n), randn(rng, n)))
            end
            @test worst > -1e-8
        end
    end

    @testset "$(rpad("semi-definiteness is NECESSARY, not decoration",76))" begin
        # an indefinite sigma breaks positivity immediately, so the proposition is not vacuous
        rng = MersenneTwister(1)
        worst = Inf
        for _ in 1:500
            A = randn(rng, 4, 4)
            worst = min(worst,
                metriplectic_bracket(A + A', random_psd(rng, 4, 4),
                    randn(rng, 4), randn(rng, 4)))
        end
        @test worst < -1e-6
    end

    @testset "$(rpad("a mismatched shape is refused rather than silently broadcast",76))" begin
        @test_throws DimensionMismatch kulkarni_nomizu(zeros(3, 3), zeros(4, 4))
        @test_throws DimensionMismatch metriplectic_bracket(zeros(2, 2, 2, 2), zeros(3), zeros(3))
    end
end
