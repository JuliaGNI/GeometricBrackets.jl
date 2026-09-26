using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Exact integration" include("exact.jl")
    @safetestset "Discrete spaces" include("spaces.jl")
    @safetestset "Tensor-product spaces" include("tensorspaces.jl")
    @safetestset "Polar spaces" include("polarspaces.jl")
    @safetestset "Pullback" include("pullback.jl")
    @safetestset "Brackets" include("brackets.jl")
    @safetestset "Arakawa" include("arakawa.jl")
    @safetestset "Metric brackets" include("metricbrackets.jl")
    @safetestset "Collision brackets" include("collisionbrackets.jl")
    @safetestset "Lie algebras" include("algebras.jl")
    @safetestset "Torus" include("torus.jl")
    @safetestset "Four-brackets" include("fourbrackets.jl")
    @safetestset "Metriplectic" include("metriplectic.jl")
    @safetestset "Dirac" include("dirac.jl")
    @safetestset "Hamiltonians" include("hamiltonians.jl")
    @safetestset "Flows" include("flows.jl")
    @safetestset "Metriplectic flows" include("metriplecticflows.jl")
    @safetestset "KdV" include("equations/kdv.jl")
    @safetestset "Miura" include("miura.jl")
    @safetestset "Burgers" include("equations/burgers.jl")
    @safetestset "Camassa-Holm" include("equations/camassaholm.jl")
    @safetestset "Diagnostics" include("diagnostics.jl")
end
if "slow" in GROUPS
    @safetestset "Integrators" include("integrators.jl")
    @safetestset "Doctests" include("quality/doctests.jl")
end
