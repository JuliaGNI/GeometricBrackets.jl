using GeometricBrackets
using GeometricBrackets: Arakawa, DiscreteBracket, PoissonOperator, PoissonTensor,
                         isantisymmetric, jacobi_residual, lie_poisson_matrix,
                         poisson_apply,
                         poisson_derivative, poisson_matrix, random_antisymmetric_c, se3,
                         _nx, _nv
import MultiIndexArrays
using LinearAlgebra
using MultiIndexArrays: GridTensor, MultiIndexLazyArray, stencil_indices
using Random
using Test

# one seeded stream for the file, drawn from by every testset
const PIN_RNG = Xoshiro(0x6f2c81de)

# the grids and the spacings of `test/helpers/poisson_tensor_pins.jl`, which wrote the reference
const PIN_TYPES = (Float32, Float64)
const PIN_GRIDS = ((3, 4), (4, 5))
const PIN_SPACINGS = (0.7, 1.3)

"""
The entries of `Array(pt)` of GeometricBrackets 0.1.2, as `test/data/` holds them: one line
`i j k value` per nonzero entry. `split` drops the `\\r` of a CRLF checkout, as on windows-latest,
and `parse` of the shortest representation of a float is exact.
"""
function pin(DT, nx, nv)
    T = zeros(DT, nx * nv, nx * nv, nx * nv)
    for line in eachline(joinpath(@__DIR__, "data", "poisson_tensor_$(DT)_$(nx)x$(nv).txt"))
        i, j, k, value = split(line)
        T[parse(Int, i), parse(Int, j), parse(Int, k)] = parse(DT, value)
    end
    return T
end

# `f(x)` with the bytes it allocates after one warm call, through a fixed-arity barrier whose
# arguments have concrete types: an `@allocated` at testset scope, or through a closure over a loop
# variable, boxes on Julia 1.11 and would fail the `min` job on its own.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
allocations(f::F, a::A, b::B, c::C) where {F, A, B, C} = (f(a, b, c); @allocated f(a, b, c))
function allocations(f::F, a::A, b::B, c::C, d::D) where {F, A, B, C, D}
    (f(a, b, c, d); @allocated f(a, b, c, d))
end

"the Arakawa of a test grid, at the spacings the pin files were generated with"
arakawa(nx, nv, DT) = Arakawa(nx, nv, DT(PIN_SPACINGS[1]), DT(PIN_SPACINGS[2]))

"the tensor of a coefficient `f` over a grid of `extents`, with the stencil width `w`"
function wrapped(f, extents, DT, w)
    gt = GridTensor(
        MultiIndexLazyArray(
            DT, (I, J, K) -> f(I, J, K), extents, extents, extents), w)
    return PoissonTensor(gt)
end

"the entries of a coefficient table `T`, linear-indexed, as a function over the grid"
gridded(T, nx, nv) =
    let li = LinearIndices((nx, nv))
        (I, J, K) -> T[li[I], li[J], li[K]]
    end

@testset "$(rpad("the wrapper reproduces the entries of GeometricBrackets 0.1.2",76))" begin
    for DT in PIN_TYPES, (nx, nv) in PIN_GRIDS

        N = nx * nv
        w = PoissonTensor(DT, nx, nv, arakawa(nx, nv, DT))
        ref = pin(DT, nx, nv)

        @test w isa PoissonTensor{DT}
        @test w isa DiscreteBracket{DT}
        @test eltype(w) == DT
        @test size(w) == (N, N, N)
        # the entries of the tensor 0.1.2 built, to the last bit, and its lazy tensor of width 1
        @test Array(w) == ref
        @test w.stencil

        # the two index forms name the same entry
        grid = CartesianIndices((nx, nv))
        for (i, j, k) in ((1, 1, 1), (N, N, N), (2, 3, 5), (nx, 1, nv + 1))
            I, J, K = grid[i], grid[j], grid[k]
            @test w[i, j, k] == w[I, J, K] == ref[i, j, k]
        end
        # and an index off the grid throws in either form, in every position
        @test_throws BoundsError w[0, 1, 1]
        @test_throws BoundsError w[1, N + 1, 1]
        @test_throws BoundsError w[1, 1, N + 1]
        @test_throws BoundsError w[CartesianIndex(0, 1), grid[1], grid[1]]
        @test_throws BoundsError w[grid[1], grid[2], CartesianIndex(nx + 1, 1)]

        # the extents of the grid, which an `nx`/`nv` swap exchanges on these two grids
        @test _nx(w) == nx
        @test _nv(w) == nv
        po = PoissonOperator(w, zeros(DT, N))
        @test _nx(po) == nx
        @test _nv(po) == nv

        # the grid arguments of any integer width, as a caller passes an `Int32`
        @test Array(PoissonTensor(DT, Int32(nx), Int32(nv), arakawa(nx, nv, DT))) == ref
        @test w[Int32(2), Int32(3), Int32(5)] == w[2, 3, 5]
    end
end

@testset "$(rpad("the wrapper is antisymmetric, and the check is not vacuous",76))" begin
    for DT in PIN_TYPES, (nx, nv) in PIN_GRIDS

        N = nx * nv
        w = PoissonTensor(DT, nx, nv, arakawa(nx, nv, DT))
        û = randn(PIN_RNG, DT, N)
        P = poisson_matrix(w, û)
        # `isantisymmetric` scales by `max(1, maximum(abs, P))`, so a zero matrix passes it
        @test maximum(abs, P) > 0
        # the entries (J, K) and (K, J) take the same terms negated, in the same order of I
        @test P == -P'
        @test isantisymmetric(w, û)
    end
end

@testset "$(rpad("a tensor that is not a GridTensor wraps too",76))" begin
    N = 12
    # a dense tensor, antisymmetric in (J, K), with entries outside every 3 × 3 stencil
    T = randn(PIN_RNG, N, N, N)
    T = (T .- permutedims(T, (1, 3, 2))) ./ 2
    w = PoissonTensor(T)
    û, c, h = randn(PIN_RNG, N), randn(PIN_RNG, N), randn(PIN_RNG, N)

    @test w isa DiscreteBracket{Float64}
    @test !w.stencil

    # the full sum, over the whole axis: the fixed -1:1 stencil of the pre-split `poisson_apply`
    # sees only a ninth of these terms
    full = [sum(û[I] * T[I, J, K] * c[K] for I in 1:N, K in 1:N) for J in 1:N]
    @test maximum(abs, full) > 0
    @test poisson_apply(w, û, c) ≈ full rtol = N^2 * eps(Float64)
    @test poisson_matrix(w, û) * c ≈ poisson_apply(w, û, c) rtol = N^2 * eps(Float64)
    @test Matrix(PoissonOperator(w, h)) ≈
          [sum(T[I, J, K] * h[K] for K in 1:N) for I in 1:N, J in 1:N] rtol = N^2 *
                                                                              eps(Float64)

    # a tensor that is not antisymmetric in (J, K) is accepted, and reported as it is: the wrapper
    # contracts the tensor it is given, and does not skew-symmetrise it on the way through
    Ta = randn(PIN_RNG, N, N, N)
    wa = PoissonTensor(Ta)
    @test !isantisymmetric(wa, û)
    @test poisson_matrix(wa, û) ==
          [sum(û[I] * Ta[I, J, K] for I in 1:N) for J in 1:N, K in 1:N]
end

@testset "$(rpad("the type boundary holds, and nothing of MultiIndexArrays leaks",76))" begin
    w = PoissonTensor(Float64, 3, 4, arakawa(3, 4, Float64))

    # the parameters are the element type and the wrapped tensor, with no `nx`, `nv` or `f` field
    @test fieldnames(typeof(w)) == (:tensor, :stencil)
    @test !(:nx in fieldnames(typeof(w))) && !(:nv in fieldnames(typeof(w)))
    @test PoissonTensor(se3()) isa PoissonTensor{Rational{BigInt}}

    # `src/` loads MultiIndexArrays by name only: the module itself is not bound, and no name of
    # it that is not exported is bound either. `names(…; imported = true)` cannot see this,
    # because it omits `using M: x` bindings.
    @test !isdefined(GeometricBrackets, :MultiIndexArrays)
    mibound = filter(setdiff(names(MultiIndexArrays; all = true), names(MultiIndexArrays))) do n
        isdefined(GeometricBrackets, n) &&
            Base.binding_module(GeometricBrackets, n) === MultiIndexArrays
    end
    @test isempty(mibound)
end

@testset "$(rpad("the contraction is the Lie-Poisson one, and the Jacobiator runs",76))" begin
    # the structure constants of se3, where the sums are exact and a transposed contraction
    # shows: it is antisymmetric and nonzero too, so `isantisymmetric` alone cannot catch it
    z = Rational{BigInt}[1 // 2, -3 // 4, 5 // 6, 7 // 8, -1 // 3, 2 // 5]
    C = se3()
    @test poisson_matrix(PoissonTensor(C), z) == lie_poisson_matrix(C, z)
    @test poisson_derivative(PoissonTensor(C), z) == C
    @test jacobi_residual(PoissonTensor(C), z) == 0
    Cbad = random_antisymmetric_c(Xoshiro(0x2c7b41), 6)
    @test jacobi_residual(PoissonTensor(Cbad), z) != 0

    for DT in PIN_TYPES, (nx, nv) in PIN_GRIDS

        N = nx * nv
        hx, hv = DT(PIN_SPACINGS[1]), DT(PIN_SPACINGS[2])
        a = Arakawa(nx, nv, hx, hv)
        w = PoissonTensor(DT, nx, nv, a)
        û, c = randn(PIN_RNG, DT, N), randn(PIN_RNG, DT, N)
        @test maximum(abs, poisson_apply(w, û, c)) > 0
        @test poisson_matrix(w, û) * c ≈ poisson_apply(w, û, c) rtol = N^2 * eps(DT)
        # Arakawa's own product carries the hx hv of its structure matrix; the wrapper's does not,
        # and the right-hand side here is the independent `_apply_P_h!` path
        @test (hx * hv) .* poisson_apply(w, û, c) ≈ poisson_apply(a, û, c) rtol = N^2 *
                                                                                  eps(DT)
    end
end

@testset "$(rpad("the stencil rule, on a grid whose stencil repeats a node",76))" begin
    # 2 × 5 at width 1: the two x-neighbours of a node are the same node, so the stencil names it
    # twice and the sums may not be restricted to it
    nx, nv = 2, 5
    N = nx * nv
    T = zeros(Float64, N, N, N)
    for I in CartesianIndices((nx, nv))
        s = collect(stencil_indices(LinearIndices((nx, nv))[I], 1, (nx, nv)))
        for J in s, K in s

            T[LinearIndices((nx, nv))[I], J, K] = 1.0
        end
    end
    w = wrapped(gridded(T, nx, nv), (nx, nv), Float64, 1)
    @test !w.stencil

    û, c, h = randn(PIN_RNG, N), randn(PIN_RNG, N), randn(PIN_RNG, N)
    full = [sum(û[I] * T[I, J, K] * c[K] for I in 1:N, K in 1:N) for J in 1:N]
    @test maximum(abs, full) > 0
    @test poisson_apply(w, û, c) ≈ full rtol = N^2 * eps(Float64)
    full_h = [sum(T[I, J, K] * h[K] for K in 1:N) for I in 1:N, J in 1:N]
    @test maximum(abs, full_h) > 0
    @test Matrix(PoissonOperator(w, h)) ≈ full_h rtol = N^2 * eps(Float64)
end

@testset "$(rpad("the sums are restricted to the stencil where it does not repeat",76))" begin
    nx, nv = 4, 5
    N = nx * nv
    calls = Ref(0)
    a = arakawa(nx, nv, Float64)
    w = wrapped((nx, nv), Float64, 1) do I, J, K
        calls[] += 1
        a(I, J, K)
    end
    @test w.stencil

    û, c = randn(PIN_RNG, N), randn(PIN_RNG, N)
    calls[] = 0
    Pc = poisson_apply(w, û, c)
    @test maximum(abs, Pc) > 0
    # one read per node of the stencil, not one per node of the grid: N (2w + 1)^4 = 1620
    @test calls[] <= N * 9^2
end

@testset "$(rpad("the width comes from the tensor, not from a fixed -1:1",76))" begin
    # a 1-D grid of 7 nodes at width 2, whose coefficients are nonzero at offset 2
    ng = 7
    T = zeros(Float64, ng, ng, ng)
    for i in 1:ng, j in collect(stencil_indices(i, 2, (ng,))),
        k in collect(stencil_indices(i, 2, (ng,)))
        T[i, j, k] = 1.0
    end
    @test T[1, 3, 1] == 1.0             # nonzero at offset 2, out of reach of a width of 1
    w = wrapped((I, J, K) -> T[I[1], J[1], K[1]], (ng,), Float64, 2)
    @test w.stencil
    @test length(stencil_indices(
        GridTensor(
            MultiIndexLazyArray(Float64, (I, J, K) -> 0.0, (ng,), (ng,), (ng,)), 2),
        1)) == 5

    û, c, h = randn(PIN_RNG, ng), randn(PIN_RNG, ng), randn(PIN_RNG, ng)
    full = [sum(û[I] * T[I, J, K] * c[K] for I in 1:ng, K in 1:ng) for J in 1:ng]
    @test maximum(abs, full) > 0
    @test poisson_apply(w, û, c) ≈ full rtol = ng^2 * eps(Float64)
    full_h = [sum(T[I, J, K] * h[K] for K in 1:ng) for I in 1:ng, J in 1:ng]
    @test Matrix(PoissonOperator(w, h)) ≈ full_h rtol = ng^2 * eps(Float64)
end

@testset "$(rpad("the extents of the tensor and its rank are checked",76))" begin
    for DT in PIN_TYPES
        @test_throws DimensionMismatch PoissonTensor(zeros(DT, 3, 3, 4))
        @test_throws DimensionMismatch PoissonTensor(zeros(DT, 3, 4, 3))
        @test_throws DimensionMismatch PoissonTensor(zeros(DT, 4, 3, 3))
        # an array of another rank is not a tensor of this kind at all
        @test_throws MethodError PoissonTensor(zeros(DT, 3, 3))
        @test_throws MethodError PoissonTensor(zeros(DT, 3, 3, 3, 3))
    end
end

@testset "$(rpad("only a two-dimensional grid has extents",76))" begin
    @test_throws MethodError _nx(PoissonTensor(randn(PIN_RNG, 4, 4, 4)))
    @test_throws MethodError _nv(PoissonTensor(randn(PIN_RNG, 4, 4, 4)))
    gt1 = GridTensor(MultiIndexLazyArray(Float64, (I, J, K) -> 0.0, (7,), (7,), (7,)), 2)
    @test_throws MethodError _nx(PoissonTensor(gt1))
    @test_throws MethodError _nv(PoissonTensor(gt1))
end

@testset "$(rpad("the states of a bracket are checked",76))" begin
    for DT in PIN_TYPES, (nx, nv) in PIN_GRIDS

        N = nx * nv
        w = PoissonTensor(DT, nx, nv, arakawa(nx, nv, DT))
        û, c = randn(PIN_RNG, DT, N), randn(PIN_RNG, DT, N)
        # N + 1 is longer than the grid, so an unchecked sum reads in bounds and returns a value
        @test_throws DimensionMismatch poisson_matrix(w, randn(PIN_RNG, DT, N + 1))
        @test_throws DimensionMismatch poisson_apply(w, randn(PIN_RNG, DT, N + 1), c)
        @test_throws DimensionMismatch poisson_apply(w, û, randn(PIN_RNG, DT, N + 1))
        @test_throws DimensionMismatch poisson_derivative(w, randn(PIN_RNG, DT, N + 1))
        # a state of the right length whose axis is 0:N-1 rather than 1:N
        @test_throws ArgumentError poisson_matrix(w, Base.IdentityUnitRange(0:(N - 1)))
        @test_throws ArgumentError poisson_apply(w, û, Base.IdentityUnitRange(0:(N - 1)))
        @test_throws ArgumentError poisson_derivative(w, Base.IdentityUnitRange(0:(N - 1)))
    end
end

@testset "$(rpad("the result's element type is the promoted one",76))" begin
    for (nx, nv) in PIN_GRIDS
        N = nx * nv
        w32 = PoissonTensor(Float32, nx, nv, arakawa(nx, nv, Float32))
        w64 = PoissonTensor(Float64, nx, nv, arakawa(nx, nv, Float64))
        û64, c64 = randn(PIN_RNG, N), randn(PIN_RNG, N)
        û32, c32 = randn(PIN_RNG, Float32, N), randn(PIN_RNG, Float32, N)
        @test poisson_apply(w32, û64, c64) isa Vector{Float64}
        @test eltype(poisson_matrix(w32, û64)) == Float64
        @test poisson_apply(w64, û32, c32) isa Vector{Float64}
        @test poisson_apply(w32, û32, c32) isa Vector{Float32}
        @test eltype(poisson_matrix(w32, û32)) == Float32
    end
end

@testset "$(rpad("the derivative tensor and the operator of a wrapped tensor",76))" begin
    for DT in PIN_TYPES, (nx, nv) in PIN_GRIDS

        N = nx * nv
        a = arakawa(nx, nv, DT)
        w = PoissonTensor(DT, nx, nv, a)
        û, h = randn(PIN_RNG, DT, N), randn(PIN_RNG, DT, N)

        # the entries of the tensor itself, in the layout dP[l, i, j] of the Jacobiator
        @test poisson_derivative(w, û) == Array(w.tensor)
        @test eltype(poisson_derivative(w, û)) == DT

        po = PoissonOperator(w, h)
        @test size(po) == (N, N)
        @test eltype(po) == DT
        full_h = [sum(w[i, j, k] * h[k] for k in 1:N) for i in 1:N, j in 1:N]
        @test maximum(abs, full_h) > 0
        @test Matrix(po) ≈ full_h rtol = N^2 * eps(DT)
        @test_throws DimensionMismatch PoissonOperator(w, randn(PIN_RNG, DT, N + 1))
        @test_throws BoundsError po[0, 1]
        @test_throws BoundsError po[1, N + 1]
    end
end

@testset "$(rpad("indexing and an operator entry are inferred and allocate nothing",76))" begin
    for DT in PIN_TYPES, (nx, nv) in PIN_GRIDS

        N = nx * nv
        w = PoissonTensor(DT, nx, nv, arakawa(nx, nv, DT))
        po = PoissonOperator(w, randn(PIN_RNG, DT, N))
        û, c = randn(PIN_RNG, DT, N), randn(PIN_RNG, DT, N)

        @test @inferred(w[2, 3, 5]) == w[2, 3, 5]
        @test @inferred(po[2, 3]) == po[2, 3]

        # Coverage instrumentation allocates on every counted line, so under it the assertion does
        # not run rather than being weakened.
        if Base.JLOptions().code_coverage == 0
            @test allocations(getindex, w, 2, 3, 5) == 0
            @test allocations(getindex, po, 2, 3) == 0
            # `poisson_apply` builds no matrix: its result, and not an N × N one
            @test allocations(poisson_apply, w, û, c) < @allocated(zeros(DT, N, N))
        end
        # a control: the barrier sees an allocation, so the zeros above are not vacuous
        @test allocations(collect, û) > 0
    end
end
