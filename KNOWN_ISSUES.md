# Known issues

## KI-1 — No test catches a wrong RK4 stage

Kind: missing test. `mutate.jl <pkg> src/integrators.jl 'k4 = vectorfield(f, û .+ Δt .* k3)'
'k4 = vectorfield(f, û .+ Δt .* k2)'` SURVIVED against `integrators.jl`, and against
`equations/kdv.jl`, `metriplecticflows.jl` and `miura.jl` together. The fix is an RK4
order-of-convergence check.

## KI-3 — A redundant `import SparseArrays` in `test/integrators.jl`

Kind: dead code. The top-level `import SparseArrays` makes the `import SparseArrays` inside the
"mixed formulation" testset redundant.

## KI-4 — Weak mutant coverage in `test/algebras.jl`

Kind: missing test. A `-` → `+` change in the `so_n` commutator of `src/algebras.jl` survives
`test/algebras.jl`.

## KI-5 — A stale test file name in `src/metriplectic.jl`

Kind: docs. `src/metriplectic.jl:27` names `test/metriplectic_tests.jl`, which is now
`test/metriplectic.jl`. A test migration does not change `src/`.
