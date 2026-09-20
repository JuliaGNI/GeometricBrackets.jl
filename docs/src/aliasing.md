```@meta
CurrentModule = GeometricBrackets
```

# Aliasing and the zero-mode theorem

Zeitlin's truncation of the two-dimensional Euler equations closes into ``\mathfrak{su}(N)``
exactly — see [Zeitlin's escape](@ref). The obvious question is whether the same trick repairs
the second KdV bracket, whose Jacobi residual is of order one and
flat under refinement — see [The Jacobi identity](@ref).

**It does not**, and this page is why. Everything here is `scripts/verify_kdv_aliasing.jl`.

## Aliasing needs a grading, and splines have none

Zeitlin's construction replaces a Fourier truncation of the sine-Euler bracket by structure
constants that are **periodic in the grading**,

```math
c_{mn}^{\,m+n} = \frac{N}{2\pi}\sin\!\left(\frac{2\pi}{N}\, m\times n\right) ,
\qquad m+n \bmod N ,
```

so that reducing an index mod ``N`` is invisible: ``\sin`` is unchanged by ``m\times n \to
m\times n + N``.

That presupposes a *multiplicative grading* — ``\phi_m\phi_n \in \operatorname{span}
\{\phi_{m+n}\}`` — which Fourier modes have and B-splines do not. The question is therefore not
even well posed in the basis this package uses for KdV; it has to be asked in a Fourier basis,
where the answer turns out to be no for reasons that have nothing to do with the basis.

## In a Fourier basis the second bracket is exactly Virasoro

```math
\mathbb{P}_{mn} = (m-n)\,u_{m+n} + c\,m^3\,\delta_{m+n,0} ,
```

the Witt Lie-Poisson part plus the central cocycle carried by ``-\partial_x^3``. Its structure
constants are ``m-n``: **linear and unbounded**. No ``N``-periodic function agrees with ``m-n``
across a window of width ``N`` without an ``O(N)`` jump at the seam — and that jump *is* the
aliasing error. Where Zeitlin's ``\sin`` matches its target smoothly across the wrap, the linear
target cannot be matched at all.

## The zero-mode theorem

This is the sharp statement, and it rules out every ``\mathbb{Z}_N``-graded repair at once. For
an ansatz ``[L_m, L_n] = f(m,n) L_{m+n}`` with indices in ``\mathbb{Z}_N``, setting ``m = 0`` in
the graded Jacobi condition gives, identically,

```math
J(0,n,p) = f(n,p)\,\big[\, \varphi(n) + \varphi(p) - \varphi(n+p) \,\big] ,
\qquad \varphi(n) := f(0,n) .
```

So ``\varphi`` must be **additive** wherever the bracket is nonzero. If ``f(n,p) \ne 0`` for all
``n \ne p`` — which consistency with ``m-n`` forces — then for ``N \ge 5`` additivity over
``\mathbb{Z}_N`` forces ``\varphi \equiv 0``, i.e. ``L_0`` **central**; whereas consistency
demands ``\varphi(n) \to -n``.

The sole exception is ``N = 3``, whose nondegenerate solution has a nondegenerate Killing form
and is therefore ``\mathfrak{sl}(2,\mathbb{C})``, the unique three-dimensional simple Lie
algebra — and three modes cannot converge to anything.

### Why Zeitlin escapes it

The sine algebra is graded by ``\mathbb{Z}_N^2 \setminus \{0\}``: **the zero mode is simply
absent**, and legitimately so, because constants are already central in
``C^\infty(\mathbb{T}^2)``. Its ``\varphi`` vanishes identically and the theorem has nothing to
bite on.

Witt's ``L_0`` cannot be dropped the same way. It is the *grading element*, and
``[L_n, L_{-n}] = 2n L_0`` puts it in the image of the bracket: removing it does not give a
subalgebra. The asymmetry between the two cases is structural, not a matter of effort.

## A second, independent obstruction

Even setting the grading aside, the ``-\partial_x^3`` term is the **Virasoro cocycle**, and it is
not a coboundary. Every candidate target here is semisimple, and a semisimple Lie algebra has
``H^2 = 0`` by Whitehead's lemma — so it admits no nontrivial central extension for the cocycle
to land in. Two obstructions, neither implying the other, and either alone is fatal.

## What does work

The repair that succeeds is not algebraic but geometric: push the *first* bracket forward along
the Miura map. See [The Miura map](@ref). It gives an exactly Poisson discrete second structure
at every resolution, and the price is locality rather than the Jacobi identity.
