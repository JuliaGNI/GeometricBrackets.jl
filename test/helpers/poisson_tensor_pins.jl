# The reference entries of `PoissonTensor`, as the released GeometricBrackets 0.1.2 gives them.
#
#     julia --startup-file=no test/helpers/poisson_tensor_pins.jl
#
# 0.2.0's `PoissonTensor` and 0.1.2's are the same name in the same package, so the two cannot be
# loaded in one session: the reference is written out here, once, and `test/poisson_tensors.jl`
# reads it back from `test/data/`. One file per element type and grid, one line `i j k value` per
# nonzero entry of `Array(pt)`, in the order `Array` has it.
#
# The grids and the spacings are those of §5V.P8's checks: `3 × 4` and `4 × 5` are non-square, so
# an `nx`/`nv` swap shows on them, and neither of `0.7` and `1.3` is a power of two, so a rounded
# coefficient is in the reference.
import Pkg

const GRIDS = ((3, 4), (4, 5))
const TYPES = (Float32, Float64)
const SPACINGS = (0.7, 1.3)

# a temporary environment, so that the `using` below takes 0.1.2 and not the tree of this checkout
Pkg.activate(mktempdir(); io = devnull)
Pkg.add(Pkg.PackageSpec(; name = "GeometricBrackets", version = "0.1.2"); io = devnull)

using GeometricBrackets: Arakawa, PoissonTensor

function write_pins()
    data = normpath(joinpath(@__DIR__, "..", "data"))
    mkpath(data)
    for DT in TYPES, (nx, nv) in GRIDS

        pt = PoissonTensor(DT, nx, nv, Arakawa(nx, nv, DT(SPACINGS[1]), DT(SPACINGS[2])))
        T = Array(pt)
        open(joinpath(data, "poisson_tensor_$(DT)_$(nx)x$(nv).txt"), "w") do io
            for i in axes(T, 1), j in axes(T, 2), k in axes(T, 3)
                T[i, j, k] == 0 && continue
                println(io, i, " ", j, " ", k, " ", string(T[i, j, k]))
            end
        end
    end
    return data
end

write_pins()
