```@meta
CurrentModule = GeometricBrackets
```

# Antisymmetric three-brackets

If the second KdV structure cannot be made Poisson, perhaps the bi-Hamiltonian pair should be
replaced by something else. A Nambu-style **three**-bracket ``\{\,\cdot\,,\,\cdot\,,\,\cdot\,\}``
that is fully antisymmetric and reproduces ``u_t = \{u, H_1, H_2\}`` would be such a thing.

This page is the answer, which is a sequence of no-go results with one small positive residue at
the end. It is §7 of the KdV notes and `scripts/verify_kdv_nambu.jl`.

## The pair ``(H_1, H_2)`` cannot work

Both ``\delta H_1/\delta u = -3u^2 - u_{xx}`` and ``\delta H_2/\delta u = u`` vanish at
``u = 0``, so any bracket built from them has **no term of first order in ``u``** — while the
right-hand side of KdV has one, ``-u_{xxx}``. The obstruction is that simple, and it survives the
two freedoms one would reach for: adding a Casimir to either Hamiltonian, and exploiting Galilean
invariance. The script shows it symbolically and again on a finite-dimensional model.

## The Vandermonde lemma

The next question is what a local antisymmetric three-bracket *can* look like. Full antisymmetry
forces the symbol to be the Vandermonde determinant in the three wavenumbers times a symmetric
factor. Evaluated at ``(k, -k, 0)`` that is ``k^3`` times an **even** polynomial in ``k``, so a
local bracket delivers

```math
\partial_x^3, \; \partial_x^5, \; \partial_x^7, \; \dots
```

in the Casimir slot, and **never** ``\partial_x``. Any local three-bracket therefore carries at
least three derivatives where the equation needs one. This is a statement about antisymmetry
alone, independent of which Hamiltonians are chosen.

## The third slot must be the Casimir

Given the above, the triple that can work is ``(H_1, C_0)`` with the mass ``C_0 = \int u\,dx`` in
the third slot. Two explicit brackets realise it, and they agree wherever the consistency
condition pins them — which is the check that the construction is determined rather than fitted.

## The discrete wedge tensor, and what it is not

On the spline basis the natural candidate is

```math
S_{ijk} = \frac{\mathbb{P}^1_{ij} + \mathbb{P}^1_{jk} + \mathbb{P}^1_{ki}}{L} ,
\qquad \sum_k S_{ijk}\, g_k = \mathbb{P}^1_{ij} ,
```

with ``g_i = \int\phi_i``. It is fully antisymmetric and it does contract back onto the first
bracket against the mass gradient. But that is also its limitation: **its flow is the
``\mathbb{P}^1`` flow.** It repackages the two-bracket theory in three slots and adds nothing to
it. Worth constructing, because the repackaging is not obvious until it is done, and worth
labelling, because it is easy to mistake for a result.

## The one genuinely new bracket

``\{u, H_2, C_0\}`` is not a repackaging. Its structure is mixed:

  - the **dispersive part is local** — half the Wronskian;
  - the **nonlinear part provably is not**, and the proof is not an appeal to failed attempts;
  - it carries a **zero-mode anomaly**: the weight at the mean of ``u`` must be ``3/2`` times the
    weight at every other mode.

So there is a three-bracket, it is genuinely new, and it is nonlocal — which is the same price
the [Miura construction](@ref "The Miura map") exacts for the Jacobi identity, arrived at from
an unrelated direction.

## The fundamental identity is out of reach by theorem

The Nambu analogue of the Jacobi identity is the *fundamental identity*, and one might hope to
recover it by working harder. One cannot. A Nambu-Poisson tensor of order ``\ge 3`` satisfying
the fundamental identity must be **decomposable** (Gautheron; Alekseevsky-Guha), hence of rank
three. Nothing of rank three can carry a field discretised on ``N`` degrees of freedom.

This is worth stating in the same breath as the rest: the other results on this page say that a
particular construction fails, and could in principle be overturned by a cleverer construction.
This one says that no construction exists. It is the same kind of statement as the closedness of
the structure-constant condition on the [Diagnostics](@ref) page — a claim about the shape of the
space being searched, not about the search.
