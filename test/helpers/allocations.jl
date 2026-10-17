# `f(x)` with the bytes it allocates after one warm call, through a fixed-arity barrier whose
# arguments have concrete types: an `@allocated` at testset scope, or through a closure over a loop
# variable, boxes on Julia 1.11 and would fail the `min` job on its own.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
allocations(f::F, a::A, b::B, c::C) where {F, A, B, C} = (f(a, b, c); @allocated f(a, b, c))
function allocations(f::F, a::A, b::B, c::C, d::D) where {F, A, B, C, D}
    (f(a, b, c, d); @allocated f(a, b, c, d))
end
