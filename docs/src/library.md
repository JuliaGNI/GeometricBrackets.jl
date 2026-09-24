```@meta
CurrentModule = GeometricBrackets
```

# Library

The docstrings that belong to a topic live with it and are not repeated here: the three
equation modules on their own pages — [Korteweg-de Vries](@ref), [Camassa-Holm](@ref) and
[Burgers](@ref) — the Lie algebras and the exact linear algebra under
[Discrete Lie-Poisson brackets](@ref), the reduction and the broken hierarchical space under
[Dirac reduction](@ref), and the measurement layer under [Diagnostics](@ref).

The index below is complete regardless.

```@index
```

## Spaces — `spaces.jl`

`Pages` is matched against the *end* of each source path, so this filter has to be qualified
too: a bare `"spaces.jl"` also matches `tensorspaces.jl` and `polarspaces.jl`, whose docstrings
belong to the two sections below and would otherwise be emitted twice.

```@autodocs
Modules = [GeometricBrackets]
Pages = ["src/spaces.jl"]
```

## Tensor-product spaces — `tensorspaces.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["tensorspaces.jl"]
```

## Polar spaces — `polarspaces.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["polarspaces.jl"]
```

## Mapped domains — `pullback.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["pullback.jl"]
```

## Brackets — `brackets.jl`

`Pages` is matched against the *end* of each source path, so this filter has to be qualified:
a bare `"brackets.jl"` also matches `fourbrackets.jl`, whose docstrings belong to
[Poisson brackets from four-brackets](@ref), and `metricbrackets.jl`, whose belong to the
section below; both would otherwise be emitted twice.

```@autodocs
Modules = [GeometricBrackets]
Pages = ["src/brackets.jl"]
```

## The Arakawa bracket — `arakawa.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["arakawa.jl"]
```

## Grid tensors — `poisson_tensors.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["poisson_tensors.jl"]
```

## Metric brackets — `metricbrackets.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["metricbrackets.jl"]
```

## The collision-like bracket — `collisionbrackets.jl`

No qualification is needed here: no other source file's path ends in `collisionbrackets.jl`,
and the filter of the section above does not match it either.

```@autodocs
Modules = [GeometricBrackets]
Pages = ["collisionbrackets.jl"]
```

## Hamiltonians — `hamiltonians.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["hamiltonians.jl"]
```

## Flows — `flows.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["src/flows.jl"]
```

## Metriplectic flows — `metriplecticflows.jl`

`Pages` is matched against the *end* of each source path, so the section above has to be
qualified against this one: a bare `"flows.jl"` matches `metriplecticflows.jl` as well, and
every docstring here would be emitted twice.

```@autodocs
Modules = [GeometricBrackets]
Pages = ["metriplecticflows.jl"]
```

## Integrators — `integrators.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["integrators.jl"]
```

## The mixed formulation — `mixed.jl`

```@autodocs
Modules = [GeometricBrackets]
Pages = ["mixed.jl"]
```
