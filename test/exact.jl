using GeometricBrackets
using LinearAlgebra
using Test

@testset "$(rpad("Exact Linear Algebra Tests",80))" begin
    @testset "$(rpad("rref reduces to the identity on a nonsingular matrix",76))" begin
        A = Rational{BigInt}[2 1 0; 1 3 1; 0 1 2]
        R, piv = rref(A)
        @test R == I
        @test piv == [1, 2, 3]
        @test exact_rank(A) == 3
        @test size(kernel(A), 2) == 0
    end

    @testset "$(rpad("rank and kernel are EXACT, where the SVD-based ones do not run",76))" begin
        # LinearAlgebra.rank and nullspace go through the SVD and have no method for
        # Rational; that gap is the reason this file exists.
        A = Rational{BigInt}[1 2 3; 2 4 6; 1 0 1]
        @test exact_rank(A) == 2
        @test_throws MethodError rank(A)
        @test_throws MethodError nullspace(A)

        K = kernel(A)
        @test size(K) == (3, 1)
        @test A * K == zeros(Rational{BigInt}, 3, 1)   # exactly zero, not ‖·‖ < ε
    end

    @testset "$(rpad("kernel dimension obeys rank-nullity, on rectangular input too",76))" begin
        for A in (Rational{BigInt}[1 2 3; 2 4 6],
            Rational{BigInt}[1 2; 2 4; 3 6],
            Rational{BigInt}[0 0; 0 0],
            Rational{BigInt}[1 0 0 0; 0 1 0 0])
            K = kernel(A)
            @test size(K, 2) == size(A, 2) - exact_rank(A)
            @test A * K == zeros(Rational{BigInt}, size(A, 1), size(K, 2))
            # the returned vectors are independent, so they really are a basis
            @test exact_rank(K) == size(K, 2)
        end
    end

    @testset "$(rpad("a zero matrix has full kernel and a full one has rank one",76))" begin
        @test kernel(zeros(Rational{Int}, 3, 3)) == I
        @test exact_rank(ones(Rational{Int}, 4, 4)) == 1
    end

    @testset "$(rpad("the same code runs on floating point and agrees with LAPACK",76))" begin
        A = Float64[2 1 0; 1 3 1; 0 1 2]
        @test exact_rank(A) == rank(A) == 3
        R, _ = rref(A)
        @test R ≈ I
    end
end
