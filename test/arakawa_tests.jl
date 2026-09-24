using GeometricBrackets
using GeometricBrackets: poisson_derivative, _apply_P_h!, _apply_P_ϕ!
using LinearAlgebra
using Random
using Test

# The Arakawa Jacobian [c, h] on an nx × nv grid, from the matrix-free operator
# `_apply_P_h!`. It is written term by term from the three second-order Jacobians rather
# than from the sign tables, so it is an independent reference for them.
function arakawa_jacobian(c, h, nx, nv, hx, hv)
    ci = CartesianIndices((nx, nv))
    J = zeros(nx * nv)
    _apply_P_h!(J, c, h, ci, LinearIndices(ci), hx, hv)
    return J
end

const ARAKAWA_GRIDS = ((3, 3), (5, 4), (6, 7))

@testset "$(rpad("Arakawa Bracket Tests",80))" begin
    @testset "$(rpad("Arakawa is a DiscreteBracket, antisymmetric to the last bit",76))" begin
        for (nx, nv) in ARAKAWA_GRIDS
            b = Arakawa(nx, nv, 1 / nx, 2 / nv)
            N = nx * nv
            û = randn(N)
            @test b isa DiscreteBracket{Float64}
            P = poisson_matrix(b, û)
            @test size(P) == (N, N)
            @test P == -P'
            @test isantisymmetric(b, û)
            dP = poisson_derivative(b, û)
            @test all(dP[l, :, :] == -dP[l, :, :]' for l in 1:N)
        end
    end

    @testset "$(rpad("the matrix is linear in the state, the derivative constant",76))" begin
        for (nx, nv) in ARAKAWA_GRIDS
            b = Arakawa(nx, nv, 1 / nx, 2 / nv)
            N = nx * nv
            û, ŵ = randn(N), randn(N)
            P = poisson_matrix(b, û)
            dP = poisson_derivative(b, û)
            @test dP == poisson_derivative(b, ŵ)
            @test sum(û[l] * dP[l, :, :] for l in 1:N) ≈ P rtol=1e-14
            @test poisson_matrix(b, û + ŵ) ≈ P + poisson_matrix(b, ŵ) rtol=1e-14
        end
    end

    @testset "$(rpad("matrix-free apply agrees with the matrix and with _apply_P_h!",76))" begin
        for (nx, nv) in ARAKAWA_GRIDS
            hx, hv = 1 / nx, 2 / nv
            b = Arakawa(nx, nv, hx, hv)
            û, c = randn(nx * nv), randn(nx * nv)
            Pc = poisson_apply(b, û, c)
            @test Pc ≈ poisson_matrix(b, û) * c rtol=1e-14
            # (P(û) c)_J = hx hv Σ_{I,K} û_I A(I,J,K) c_K = hx hv [c, û]_J, by the cyclic
            # symmetry of Arakawa's coefficients
            @test Pc ≈ hx * hv * arakawa_jacobian(c, û, nx, nv, hx, hv) rtol=1e-14
        end
    end

    @testset "$(rpad("_apply_P_ϕ! is _apply_P_h! with h = ϕ + v²/2",76))" begin
        for (nx, nv) in ARAKAWA_GRIDS
            hx, hv = 1 / nx, 2 / nv
            ci = CartesianIndices((nx, nv))
            v = range(-1, 1 - hv, length = nv)
            ϕ, f = randn(nx), randn(nx * nv)
            h = vec([ϕ[i] + v[j]^2 / 2 for i in 1:nx, j in 1:nv])
            Pf = zeros(nx * nv)
            _apply_P_ϕ!(Pf, f, v, ϕ, ci, LinearIndices(ci), hx, hv)
            @test Pf ≈ arakawa_jacobian(f, h, nx, nv, hx, hv) rtol=1e-13
        end
    end

    @testset "$(rpad("mass and enstrophy are Casimirs",76))" begin
        for (nx, nv) in ARAKAWA_GRIDS
            b = Arakawa(nx, nv, 1 / nx, 2 / nv)
            N = nx * nv
            û = randn(N)
            P = poisson_matrix(b, û)
            # the gradients of Σ f and of Σ f²/2 are the constant and the state itself
            @test norm(P' * ones(N)) < 1e-13 * norm(P)
            @test norm(P' * û) < 1e-13 * norm(P) * norm(û)
        end
    end

    @testset "$(rpad("the Jacobi residual is of order one and flat under refinement",76))" begin
        # Reported, not asserted to be zero. The coefficients hx hv A are integers over 12
        # whatever the grid spacing, so refinement cannot close them into a Lie algebra.
        res = [jacobi_residual(Arakawa(n, n, 1 / n, 2 / n), randn(n^2)) for n in 5:8]
        @test all(>(0.3), res)
        @test all(<(0.9), res)
        @test res[end] > res[1] / 2
        # the residual of the structure constants hx hv A, which does not depend on the state
        @test all(5:8) do n
            b = Arakawa(n, n, 1 / n, 2 / n)
            structure_constant_residual(poisson_derivative(b, zeros(n^2))) ≈ 0.5
        end
    end

    @testset "$(rpad("the bracket converges to the analytic one at second order",76))" begin
        # The analytic solution of ReducedBasisMethods' bracket_operators_test.jl, for which
        # [f, h] = ∂ₓf ∂ᵥh - ∂ᵥf ∂ₓh is known in closed form.
        function analytic_error(n)
            h₁, h₂ = 1 / n, 2 / n
            x = range(0, 1 - h₁, length = n)
            v = range(-1, 1 - h₂, length = n)
            f = vec([1 / (2π) * cos(_x * 2π) * (cos(_v * 2π) - 1) for _x in x, _v in v])
            h = vec([sin(_x * 2π) * exp(-16 * _v^2) for _x in x, _v in v])
            fh = vec([32 * _v * sin(_x * 2π)^2 * (cos(_v * 2π) - 1) * exp(-16 * _v^2) +
                      2π * cos(_x * 2π)^2 * sin(_v * 2π) * exp(-16 * _v^2)
                      for _x in x, _v in v])
            # {f, h} with the state h: (P(h) f)_J = hx hv [f, h]_J
            b = Arakawa(n, n, h₁, h₂)
            norm(poisson_apply(b, h, f) / (h₁ * h₂) - fh) / norm(fh)
        end
        errs = [analytic_error(n) for n in (32, 64, 128)]
        @test all(r -> 3.8 < r < 4.2, errs[1:(end - 1)] ./ errs[2:end])
        # the tolerance of the original test, at its resolution
        @test analytic_error(512) < 5e-4
    end

    @testset "$(rpad("PoissonTensor and PoissonOperator index a non-square grid",76))" begin
        # nx > nv, so that an index bound checked against the wrong extent fails here
        nx, nv = 5, 3
        hx, hv = 1 / nx, 2 / nv
        a = Arakawa(nx, nv, hx, hv)
        pt = PoissonTensor(Float64, nx, nv, a)
        @test size(pt) == (nx * nv, nx * nv, nx * nv)
        I, J, K = CartesianIndex(nx, 1), CartesianIndex(1, 1), CartesianIndex(nx, 2)
        @test pt[I, J, K] == a(I, J, K)
        @test pt[nx, 1, 2nx] == a(I, J, K)
        # each with one component out of range, and every other component within 1:nv
        I₀, J₀, K₀ = CartesianIndex(1, 1), CartesianIndex(2, 1), CartesianIndex(1, 2)
        @test_throws BoundsError pt[I₀, CartesianIndex(1, nv + 1), K₀]
        @test_throws BoundsError pt[I₀, J₀, CartesianIndex(nx + 1, 1)]
        @test_throws BoundsError pt[CartesianIndex(0, 1), J₀, K₀]
        @test_throws BoundsError pt[0, 1, 1]
        @test_throws BoundsError pt[1, nx * nv + 1, 1]
        @test_throws BoundsError pt[1, 1, nx * nv + 1]

        # every row, including those whose first component exceeds nv
        h, f = randn(nx * nv), randn(nx * nv)
        T = Array(pt)
        @test size(T) == size(pt)
        @test [f' * T[i, :, :] * h for i in 1:(nx * nv)] ≈
              arakawa_jacobian(f, h, nx, nv, hx, hv) rtol=1e-14
        po = PoissonOperator(pt, h)
        @test Matrix(po) * f ≈ arakawa_jacobian(f, h, nx, nv, hx, hv) rtol=1e-14
        @test_throws BoundsError po[0, 1]
        @test_throws BoundsError po[1, nx * nv + 1]
        @test_throws DimensionMismatch PoissonOperator(pt, randn(nx * nv + 1))
    end

    @testset "$(rpad("spacings of any real types construct an Arakawa",76))" begin
        @test Arakawa(6, 7, 1, 2) isa Arakawa{Float64}
        @test Arakawa(6, 7, 1, 2.0) isa Arakawa{Float64}
        @test Arakawa(6, 7, 0.1f0, 0.2) isa Arakawa{Float64}
        @test Arakawa(6, 7, 0.1f0, 0.2f0) isa Arakawa{Float32}
        @test Arakawa(6, 7, 1 // 6, 2 // 7) isa Arakawa{Rational{Int}}
        b, a = Arakawa(6, 7, 1, 2), Arakawa(6, 7, 1.0, 2.0)
        û = randn(42)
        @test poisson_matrix(b, û) == poisson_matrix(a, û)
    end

    @testset "$(rpad("invalid grids and states are refused",76))" begin
        @test_throws ArgumentError Arakawa(2, 5, 0.5, 0.4)
        @test_throws ArgumentError Arakawa(5, 2, 0.2, 1.0)
        b = Arakawa(4, 3, 0.25, 2 / 3)
        @test_throws DimensionMismatch poisson_matrix(b, randn(11))
        @test_throws DimensionMismatch poisson_apply(b, randn(12), randn(13))
        @test_throws DimensionMismatch poisson_derivative(b, randn(13))
    end
end
