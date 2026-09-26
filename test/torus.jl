using LinearAlgebra
using GeometricBrackets
using Test

@testset "$(rpad("Torus Grid Tests",80))" begin

    # A trigonometric polynomial of highest mode three, together with its exact derivatives,
    # so that the spectral grid at N = 16 has to reproduce them to roundoff and the
    # finite-difference grid has to converge to them at order eight.
    field(g) = sample(g, (x, y) -> sin(x) + 0.6cos(2y) + 0.4sin(x + y) + 0.25cos(2x - 3y))
    fieldx(g) = sample(g, (x, y) -> cos(x) + 0.4cos(x + y) - 0.5sin(2x - 3y))
    fieldy(g) = sample(g, (x, y) -> -1.2sin(2y) + 0.4cos(x + y) + 0.75sin(2x - 3y))

    @testset "$(rpad("a grid reports its own shape",76))" begin
        g = spectral_grid(16)
        @test eltype(g) == Float64
        @test size(g) == (16, 16)
        @test length(g) == 256
        @test length(nodes(g)) == 16
        @test nodes(g)[1] == 0.0
        @test nodes(g)[end] ≈ 2π - 2π / 16
        @test size(sample(g, (x, y) -> 1.0)) == (16, 16)
        @test occursin("spectral", string(g))
    end

    @testset "$(rpad("spectral differentiation is EXACT for a band-limited field",76))" begin
        g = spectral_grid(16)
        f = field(g)
        @test maximum(abs, ∂x(g, f) .- fieldx(g)) < 1e-13
        @test maximum(abs, ∂y(g, f) .- fieldy(g)) < 1e-13
        # a mode above the Nyquist frequency is aliased, so 16 points is not enough for
        # everything -- the exactness above is a statement about the field, not the grid
        @test maximum(abs, ∂x(g, sample(g, (x, y) -> cos(9x)))) > 1
    end

    @testset "$(rpad("the finite-difference grid converges at order eight",76))" begin
        # log u is not band-limited, so this is the regime the fd8 grid exists for
        err(N) =
            let g = finite_difference_grid(N)
                u = sample(g, (x, y) -> 3.0 + 0.4sin(x) + 0.3cos(y))
                exact = sample(g, (x, y) -> 0.4cos(x) / (3.0 + 0.4sin(x) + 0.3cos(y)))
                maximum(abs, ∂x(g, log.(u)) .- exact)
            end
        e64, e128 = err(64), err(128)
        @test e64 / e128 > 40           # 2^8 = 256 predicted; the constant is not one
        @test log2(e64 / e128) > 6
        @test e128 < 1e-11
    end

    @testset "$(rpad("both differentiation matrices are antisymmetric",76))" begin
        # exactly so for the centred stencil, and to roundoff for the spectral one, whose
        # matrix is assembled by complex arithmetic. This is what makes `canonical_bracket`
        # antisymmetric and the integral of a total derivative vanish.
        Dfd = finite_difference_grid(32).D
        @test Dfd == -transpose(Dfd)
        @test all(iszero, diag(Dfd))
        @test norm(spectral_grid(16).D + transpose(spectral_grid(16).D)) < 1e-13
    end

    @testset "$(rpad("a derivative integrates to zero, by periodicity",76))" begin
        for g in (spectral_grid(16), finite_difference_grid(32))
            f = field(g)
            @test abs(integrate(g, ∂x(g, f))) < 1e-12
            @test abs(integrate(g, ∂y(g, f))) < 1e-12
        end
    end

    @testset "$(rpad("quadrature is exact on the constant and on any single mode",76))" begin
        for g in (spectral_grid(16), finite_difference_grid(32))
            @test integrate(g, sample(g, (x, y) -> 1.0)) ≈ 4π^2
            @test abs(integrate(g, sample(g, (x, y) -> sin(2x) * cos(3y)))) < 1e-12
            @test integrate(g, sample(g, (x, y) -> 2.0 + sin(x))) ≈ 8π^2
        end
    end

    @testset "$(rpad("the two derivatives commute",76))" begin
        for g in (spectral_grid(16), finite_difference_grid(32))
            f = field(g)
            @test maximum(abs, ∂x(g, ∂y(g, f)) .- ∂y(g, ∂x(g, f))) < 1e-11
        end
    end

    @testset "$(rpad("the canonical bracket is antisymmetric and kills its own arguments",76))" begin
        for g in (spectral_grid(16), finite_difference_grid(32))
            a, b = field(g), sample(g, (x, y) -> cos(y) + 0.5sin(2x) + 0.3cos(x - y))
            @test maximum(abs, canonical_bracket(g, a, b) .+ canonical_bracket(g, b, a)) <
                  1e-12
            @test maximum(abs, canonical_bracket(g, a, a)) < 1e-12
            # int f[g,h] is fully cyclic, which is the identity every reduction rests on
            c = sample(g, (x, y) -> 2.0 + 0.5sin(x - y))
            @test integrate(g, a .* canonical_bracket(g, b, c)) ≈
                  integrate(g, c .* canonical_bracket(g, a, b)) atol = 1e-11
        end
    end

    @testset "$(rpad("the pseudoinverse undoes a derivative on the modes it can",76))" begin
        # every term carries a nonzero x-wavenumber, so none of it is in the null space of D
        # and the round trip is the identity rather than a projection
        xmodes(g) = sample(g, (x, y) -> sin(x) + 0.4sin(x + y) + 0.25cos(2x - 3y))
        for g in (spectral_grid(16), finite_difference_grid(32))
            D⁺ = pinv(g)
            u = xmodes(g)
            @test maximum(abs, D⁺ * ∂x(g, u) .- u) < 1e-10
            @test maximum(abs, ∂x(g, D⁺ * u) .- u) < 1e-10
            # the same operator serves y, transposed, exactly as ∂y mirrors ∂x
            v = permutedims(u)
            @test maximum(abs, ∂y(g, v) * transpose(D⁺) .- v) < 1e-10
        end
    end

    @testset "$(rpad("the pseudoinverse agrees with a dense SVD one",76))" begin
        for g in (spectral_grid(16), finite_difference_grid(16))
            D, D⁺ = Matrix(g.D), pinv(g)
            @test maximum(abs, D⁺ .- pinv(D)) < 1e-12
            # the four Moore-Penrose conditions, which is what "the" pseudoinverse means
            @test maximum(abs, D * D⁺ * D .- D) < 1e-11
            @test maximum(abs, D⁺ * D * D⁺ .- D⁺) < 1e-11
            @test maximum(abs, transpose(D * D⁺) .- D * D⁺) < 1e-12
            @test maximum(abs, transpose(D⁺ * D) .- D⁺ * D) < 1e-12
        end
    end

    @testset "$(rpad("what the pseudoinverse cannot undo is the null space of D",76))" begin
        # a constant has no derivative to invert, and at even N neither does the Nyquist
        # mode, so D is rank N - 2 and D⁺D is a projector rather than the identity
        for g in (spectral_grid(16), finite_difference_grid(16))
            D⁺ = pinv(g)
            @test rank(Matrix(g.D)) == g.N - 2
            @test maximum(abs, D⁺ * sample(g, (x, y) -> 1.0)) < 1e-12
            @test maximum(abs, D⁺ * sample(g, (x, y) -> cos(g.N * x / 2))) < 1e-12
            # a projector: idempotent, and it removes exactly those two modes
            P = D⁺ * Matrix(g.D)
            @test maximum(abs, P * P .- P) < 1e-11
            @test rank(P) == g.N - 2
        end
    end

    @testset "$(rpad("a grid too coarse for its own stencil is refused",76))" begin
        @test_throws ArgumentError finite_difference_grid(8)
        @test_throws ArgumentError spectral_grid(1)
        @test finite_difference_grid(9) isa TorusGrid
    end
end
