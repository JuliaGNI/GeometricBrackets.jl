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

### K8 · `poisson_matrix` and `poisson_apply` drop the coefficients of a non-local `f` that `Array`, `getindex` and `poisson_derivative` keep.

- **location:** `src/poisson_tensors.jl:59`
- **evidence:** `PoissonTensor(DT, nx, nv, f)` wraps `f` in a `GridTensor` of width 1, so the sums
  run over the nearest-neighbour stencil, while `Array(pt)`, `pt[i, j, k]` and
  `poisson_derivative` read every entry. The docstring states that `f` must vanish outside the
  stencil, and nothing checks it. On a 5 × 5 grid,
  `f(I, J, K) = Float64((I[1] + 2J[1] + 3K[1] + I[2] * J[2] - K[2]) % 7 - 3) * (J != K)` gives
  `maximum(abs, poisson_matrix(w, u) - lie_poisson_matrix(Array(w), u))` of 25.8 and 21.2
  for `w = PoissonTensor(Float64, 5, 5, f)` and two draws of `u = randn(25)`. For the `Arakawa`
  of the same grid, which vanishes outside the stencil, the difference is 0.0. The options are a stencil-width
  argument for the constructor or a check of `f`; neither is decided.
- **kind:** defect
- **found:** #37

### K9 · `velocity_parity_projectors` reflects the velocity index `j → nv + 1 - j`, which is not the velocity reflection on a `Periodic()` axis, and nothing rejects the axis.

- **location:** `src/equations/vlasov.jl`
- **evidence:** With a v-even `f0` on `TensorSplineSpace((4, 5), 2, (Periodic(), Periodic()))`,
  `parity_vlasov_blocks(s, f0)` gives `maximum(abs, J_pp)` of 0.744 and `maximum(abs, J_mm)` of
  0.656 for `orthonormal = false`, and 0.241 and 0.180 for `orthonormal = true`. The blocks
  vanish to machine precision on `Free()` and `Dirichlet()`. Whether the index reversal is the
  intended parity on a periodic velocity axis, or whether the function needs the periodic
  reflection or an `ArgumentError`, is not decided.
- **kind:** defect
- **found:** #43

### K10 · `_hermitian_eigen` is a local Jacobi eigensolver for the non-BLAS element types, and `GeometricSolvers` is to provide that function.

- **location:** `src/equations/vlasov.jl`
- **evidence:** `_hermitian_eigen(W::AbstractMatrix{Complex{T}})` for `T <: Real` outside `BlasReal`
  is a hand-written cyclic Jacobi method. `GeometricSolvers` plans a Hermitian `jacobi_eigen`,
  which is not released. The local copy stays until that release, and then goes. It raises a
  `DomainError` for a non-finite entry and an error after `maxsweeps` sweeps without
  convergence.
- **kind:** upstream
- **found:** #43
