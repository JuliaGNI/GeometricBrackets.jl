using GeometricBrackets
using LinearAlgebra
using Random
using SimpleSplines: UniformMesh, RandomMesh
using Test

"""Central-difference gradient, for checking the analytic ones."""
function fd_gradient(H, s, û; h = 1e-6)
    [(hamiltonian(H, s, û .+ h .* e) - hamiltonian(H, s, û .- h .* e)) / 2h
     for e in eachcol(Matrix(I, length(û), length(û)))]
end

"""Central-difference Hessian, from the analytic gradient."""
function fd_hessian(H, s, û; h = 1e-6)
    N = length(û)
    J = zeros(N, N)
    for m in 1:N
        e = zeros(N)
        e[m] = h
        J[:, m] = (gradient(H, s, û .+ e) .- gradient(H, s, û .- e)) ./ 2h
    end
    J
end

@testset "$(rpad("Discrete Hamiltonian Tests",80))" begin
    @testset "$(rpad("quadratic Hamiltonian",76))" begin
        s = SplineSpace(12, 3)
        H = KdVHamiltonian2(s)
        û = randn(12)
        @test hamiltonian(H, s, û) ≈ dot(û, mass_matrix(s), û) / 2
        @test gradient(H, s, û) ≈ mass_matrix(s) * û
        @test hessian(H, s, û) ≈ mass_matrix(s)
        @test GeometricBrackets.value(H, s, û) == hamiltonian(H, s, û)
        @test_throws ArgumentError GeometricBrackets.QuadraticHamiltonian(randn(3, 3))
    end

    @testset "$(rpad("mass Casimir is linear",76))" begin
        for s in (SplineSpace(12, 3), LagrangeSpace(2, 7))
            C = GeometricBrackets.MassCasimir(s)
            N = nbasis(s)
            û = randn(N)
            @test hamiltonian(C, s, û) ≈ dot(basis_integrals(s), û)
            @test gradient(C, s, û) ≈ basis_integrals(s)
            @test gradient(C, s, û) == gradient(C, s, randn(N))     # constant
            @test maximum(abs, hessian(C, s, û)) == 0
            # the mass of the discrete field is the integral of the field
            @test hamiltonian(C, s, û) ≈
                  dot(quadrature_weights(s), field(s, û))
        end
    end

    @testset "$(rpad("KdV gradients against finite differences",76))" begin
        for (nm, mk) in SPLINE_MESHES, p in (2, 3)

            s = SplineSpace(mk(12), p)
            û = 0.3 .* randn(12)
            for H in (KdVHamiltonian1(), KdVHamiltonian2(s), GeometricBrackets.MassCasimir(s))
                g = gradient(H, s, û)
                @test g ≈ fd_gradient(H, s, û) atol = 1e-6 * max(1, maximum(abs, g))
            end
        end
    end

    @testset "$(rpad("KdV Hessians against finite differences, and symmetric",76))" begin
        s = SplineSpace(UniformMesh(12, 2π), 3)
        û = 0.3 .* randn(12)
        H = KdVHamiltonian1()
        A = hessian(H, s, û)
        @test A ≈ A'
        @test A ≈ fd_hessian(H, s, û) atol = 1e-6 * maximum(abs, A)
        # H₁ is CUBIC, so its Hessian genuinely depends on û -- which is why a symplectic
        # method does not conserve it
        @test !(hessian(H, s, û) ≈ hessian(H, s, 2 .* û))
    end

    @testset "$(rpad("H1 gradient is the discrete variational derivative",76))" begin
        # ∂H₁/∂û_i = ∫(φ_i' u_x - 3 φ_i u²), which after one integration by parts is the
        # weak form of -3u² - u_xx. No integration by parts is needed for the gradient
        # itself, so it is exact for p ≥ 1.
        s = SplineSpace(UniformMesh(64, 2π), 3)
        û = project(s, sin)
        g = gradient(KdVHamiltonian1(), s, û)
        w = quadrature_weights(s)
        Φ = basis_values(s, 0)
        direct = Φ * (w .* (-3 .* field(s, û) .^ 2
                            .-
                            field(s, û, 2)))
        @test maximum(abs, g .- direct) < 1e-8 * maximum(abs, g)
    end

    @testset "$(rpad("H3 is the fifth-order hierarchy member",76))" begin
        s = SplineSpace(UniformMesh(12, 2π), 4)
        û = 0.3 .* randn(12)
        H = KdVHamiltonian3()
        g = gradient(H, s, û)
        @test g ≈ fd_gradient(H, s, û) atol = 1e-5 * max(1, maximum(abs, g))
    end

    @testset "$(rpad("Burgers Casimir gradient",76))" begin
        s = LagrangeSpace(2, 7)
        C = burgers_casimir(s)
        u = 1 .+ rand(nbasis(s))
        @test hamiltonian(C, s, u) ≈ 2 * sum(basis_integrals(s) .* sqrt.(u))
        @test gradient(C, s, u) ≈ basis_integrals(s) ./ sqrt.(u)
        @test gradient(C, s, u) ≈ fd_gradient(C, s, u) atol = 1e-6
    end
end
