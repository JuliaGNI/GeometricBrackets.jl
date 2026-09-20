# GeometricBrackets

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/GeometricBrackets.jl/stable/)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/GeometricBrackets.jl/latest/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Build Status](https://github.com/JuliaGNI/GeometricBrackets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/GeometricBrackets.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/GeometricBrackets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/GeometricBrackets.jl)

Structure-preserving discretisations of the Poisson brackets of one-dimensional Hamiltonian
partial differential equations, with the discrete brackets, Hamiltonians, Casimirs and
geometric integrators kept as separate, composable objects.

The package collects the prototype discretisations and numerical experiments behind three
manuscripts on discrete Poisson brackets, and turns the structural claims they make — exact
antisymmetry, the Jacobi identity or its failure, conservation of Hamiltonians and Casimirs —
into a test suite and twenty-two verification scripts.

Three equations are covered:

| equation | bracket | basis |
|:--|:--|:--|
| Burgers | `J(u) = g(u) K g(u)` with `g = √u`, made constant by `ū = √u` | periodic nodal Lagrange |
| KdV | the bi-Hamiltonian pair `P¹` (constant, Poisson) and `P²(û)` (antisymmetric, not Jacobi) | periodic B-spline |
| Camassa-Holm | the standard bi-Hamiltonian pair — *prototype* | periodic B-spline |

and, for the second manuscript and its Dirac companion, the finite-dimensional machinery those
brackets are special cases of:

| topic | what is here |
|:--|:--|
| Lie-Poisson brackets | structure constants `c[m,i,j]` and the closure condition; `se(3)`, `so(N)`, Zeitlin's sine algebra, the truncated Witt, polynomial and torus algebras |
| Dirac reduction | Schur complement, Dirac projector and tensor, and the tensorial transport of the Jacobiator that makes reduction faithful but never curative |
| the broken hierarchical space | the DG realisation in which `[V₁, V₁] ⊆ V₁ ⊕ V₂` holds exactly, assembled over `Rational{BigInt}` with no CAS |
| exact linear algebra | `rref`, `exact_rank` and `kernel` over ℚ, where the SVD-based `rank` and `nullspace` do not run |

The third manuscript works in the continuum instead, on the periodic two-torus, and asks which
**four**-brackets reduce to a Poisson bracket when a fixed entropy is fed into two of their four
slots:

| topic | what is here |
|:--|:--|
| the torus | `TorusGrid` with spectral or 8th-order finite-difference differentiation, stored as one matrix, and the canonical bracket `[f,k] = f_x k_y − f_y k_x` |
| four-brackets | the Gardner and symmetric families with their weighted variants, both reductions to `∫u[A_u,B_u]` — reached exactly, not asymptotically — the Plücker relation that collapses the Jacobi obstruction, and the two antisymmetry conditions that each family satisfies exactly one of |
| metriplectic | the Kulkarni-Nomizu product and the positivity of the two-bracket it induces |

The [documentation](https://JuliaGNI.github.io/GeometricBrackets.jl/latest/) carries the theory:
what each diagnostic measures, the backward error analysis of the implicit midpoint rule, the
singularity of the log-entropy weight, and the no-go results for aliasing and for antisymmetric
three-brackets.

## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
