# Known issues

### KI-1 · No test catches a wrong RK4 stage.

- **location:** `src/integrators.jl`
- **evidence:** `mutate.jl <pkg> src/integrators.jl 'k4 = vectorfield(f, û .+ Δt .* k3)'
  'k4 = vectorfield(f, û .+ Δt .* k2)'` SURVIVED against `integrators.jl`, and against
  `equations/kdv.jl`, `metriplecticflows.jl` and `integration/miura.jl` together. The fix is an RK4
  order-of-convergence check.
- **kind:** missing test
- **found:** #22

### KI-3 · A redundant `import SparseArrays` in `test/integrators.jl`.

- **location:** `test/integrators.jl`
- **evidence:** The top-level `import SparseArrays` makes the `import SparseArrays` inside the
  "mixed formulation" testset redundant.
- **kind:** dead code
- **found:** #22

### KI-4 · Weak mutant coverage in `test/algebras.jl`.

- **location:** `src/algebras.jl`
- **evidence:** A `-` → `+` change in the `so_n` commutator of `src/algebras.jl` survives
  `test/algebras.jl`.
- **kind:** missing test
- **found:** #22

### KI-5 · A stale test file name in `src/metriplectic.jl`.

- **location:** `src/metriplectic.jl:27`
- **evidence:** `src/metriplectic.jl:27` names `test/metriplectic_tests.jl`, which is now
  `test/metriplectic.jl`. A test migration does not change `src/`.
- **kind:** docs
- **found:** #22

### K6 · `PoissonOperator` takes its element type from the tensor alone.

- **location:** `src/poisson_tensors.jl:149`
- **problem:** `PoissonOperator{DT, PT, HT} <: AbstractMatrix{DT}` fixes `DT` at the element type of
  the tensor, so a Hamiltonian of another element type either rounds in `Matrix(po)` — a `Float32`
  tensor with a `Float64` Hamiltonian — or raises `InexactError` — a `Float64` tensor with a
  `ComplexF64` Hamiltonian.
- **evidence:** `PoissonOperator(PoissonTensor(ones(Float64, 1, 1, 1)), ComplexF64[1 + 2im])`, then
  `Matrix(po)`, raises `InexactError: Float64(1 + 2im)`. Measured 2026-10-12 on 1.13.1. The class
  is on `main`: `origin/main`'s `src/poisson_tensors.jl:67` declares the same `AbstractMatrix{DT}`
  and its `:95` the same `x = zero(DT)`, and a tree of `origin/main` under the same script raised
  `InexactError: Float64(1.098901098901099 + 2.197802197802198im)`. Mixed Hamiltonians are outside
  the part's tested domains: §5V.P8 decides the operator's signature and says `h` holds one value
  per node, and the made edges claim the promoted element type of `poisson_matrix` and
  `poisson_apply` only.
- **kind:** found late
- **found:** 2026-10-12, critic round 1c of part P8
