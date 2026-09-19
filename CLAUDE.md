# PoissonBrackets

`Packages/CLAUDE.md` governs this repository. This file holds the two rules that are specific to it.

## SymPy lives in `scripts/`, never in the package

`SymPyPythonCall` is declared in `scripts/Project.toml` and **must not** appear in the package's
`Project.toml`, its tests, or CI. The matrix is 5 Julia versions × 3 operating systems with no
Python provisioning, so a Python dependency in the package breaks every job.

`CondaPkg` auto-provisions sympy because `SymPyPythonCall` ships its own `CondaPkg.toml`. No
`scripts/CondaPkg.toml` is needed. Symbolics.jl does not substitute: it has no `dsolve` and a weak
`solve`.

Where a symbolic helper reaches for SymPy only to do exact rational arithmetic — the shifted
Legendre integrals are the case — use `Rational{BigInt}` instead and keep it out of `scripts/`
entirely.

`scripts/check.jl` reproduces the retired Python `common.py` output format byte for byte, which is
what makes a comparison a plain `diff`. Import it as `using .Checks: header, check, summary` — a
bare `using` leaves `summary` ambiguous with `Base.summary`.

## so(3) is not a usable Jacobi control

An earlier docstring claimed that every 3-dimensional antisymmetric bracket satisfies Jacobi
identically. It does not: 200 of 200 random antisymmetric `c` in dimension 3 fail. so(3) passes
only because it lies in the six-parameter Bianchi class A family, and it keeps passing under the
obvious perturbations, so a control built on it cannot fail.

Use se(3) as the positive control and a random antisymmetric `c` in dimension five as the negative
one. `test/algebras_tests.jl` pins this. The full statement is in
`Knowledge/Discrete Poisson Brackets/so(3) is a degenerate Jacobi control only because it lies in Bianchi class A.md`.
