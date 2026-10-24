# `f(x)` with the bytes it allocates after one warm call, through a fixed-arity barrier whose
# arguments have concrete types. A bare `@allocated w[2, 3, 5]` or `@allocated po[2, 3]` in a
# `@testset` body reads 16 bytes: that body is top-level code, so the call is dynamically
# dispatched and its isbits result is boxed. The same call through this barrier is compiled
# against concrete parameter types and reads 0 (measured on Julia 1.12.7, `--check-bounds=yes`).
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
allocations(f::F, a::A, b::B, c::C) where {F, A, B, C} = (f(a, b, c); @allocated f(a, b, c))
function allocations(f::F, a::A, b::B, c::C, d::D) where {F, A, B, C, D}
    (f(a, b, c, d); @allocated f(a, b, c, d))
end
