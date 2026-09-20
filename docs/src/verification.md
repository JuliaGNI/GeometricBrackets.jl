```@meta
CurrentModule = GeometricBrackets
```

# Verification

This page records what the package has been checked against, claim by claim, and what came
out of it. What each script *checks*, as opposed to what it agreed with, is indexed in
[The verification scripts](@ref). The references are the manuscripts —
`discrete-kdv-brackets{,-notes}.tex`, `discrete-lie-poisson-brackets{,-notes}.tex` and its
companion `discrete-lie-poisson-dirac-brackets.tex` — and the Python prototypes in their
`Scripts/` directories, which the Julia is a port of.

It is written so that the *absence* of an entry is informative: everything examined is listed,
including the things that turned out to be correct and the one system that has nothing to be
checked against.

## Method

  - Julia: `julia --project=. -e 'using Pkg; Pkg.test()'`, then
    `julia --project=scripts scripts/run_all.jl`, which runs all twenty-seven verification
    scripts and exits nonzero if any check fails. `search_dirac_variants.jl` is exploratory
    and slow, and is excluded from that driver as it was from `run_all.sh`; run it by hand.
  - Figures: `julia --project=scripts scripts/kdv.jl` and `scripts/kdv_bea_sweep.jl`, both of
    which take `--outdir=PATH`.
  - Python, *while it existed*: `./run_all.sh` in both `Scripts/` directories, plus
    `plot_kdv_energies.py` for the drift tables, which existed only on stdout. Those scripts
    have since been converted and retired; the agreement established before they were removed
    is recorded below.
  - Both manuscripts recompile with XeTeX — not pdfTeX or LuaTeX; `fontspec` rules out the
    first and a `\boundary` clash with a LuaTeX primitive in `definitions_math.tex` the
    second.

## The Runge-Kutta 4 question

The RK4 results for the `cos` and single-soliton runs were suspected of disagreeing with the
Python. **They do not.** Every quantity above round-off agrees; the columns that differ are
round-off-dominated, and the mass column is the built-in control that shows it.

The step size is identical to full precision, which rules out the obvious explanation first:

| case | ``\Delta t_{\mathrm{RK4}}`` (Python) | steps | Julia |
|:--|:--|--:|:--|
| `cos` | `1.3041539392303622e-4` | 766781 | identical |
| `soliton1` | `4.195777470400295e-3` | 23833 | identical |

The drift tables, Python against Julia:

| `cos` | ``\lvert\Delta H_1\rvert`` | ``\lvert\Delta H_2\rvert`` | ``\lvert\Delta C_0\rvert`` |
|:--|:--|:--|:--|
| RK4, flow 1 | 1.1e-11 / 6.2e-12 | 1.1e-10 / **1.1e-10** | 7.7e-12 / 4.7e-12 |
| RK4, flow 2 | 3.4e-07 / **3.4e-07** | 1.3e-13 / 2.3e-13 | 2.2e-13 / 6.3e-13 |

| `soliton1` | ``\lvert\Delta H_1\rvert`` | ``\lvert\Delta H_2\rvert`` | ``\lvert\Delta C_0\rvert`` |
|:--|:--|:--|:--|
| RK4, flow 1 | 5.9e-08 / **5.9e-08** | 3.7e-08 / **3.6e-08** | 3.9e-12 / 6.5e-12 |
| RK4, flow 2 | 4.4e-07 / **4.4e-07** | 3.8e-08 / **3.8e-08** | 1.3e-14 / 1.5e-14 |

The bold entries are the genuine dynamical drifts, and they agree to every digit the tables
print. What differs — `cos` flow 1's ``\lvert\Delta H_1\rvert``, and the mass column
throughout — sits at ``10^{-11}`` or below.

That those are round-off is not an assumption. ``C_0`` is conserved *exactly* in exact
arithmetic, by [`MassCasimir`](@ref), so its measured drift is a pure meter of accumulated
round-off: 7.7e-12 in Python and 4.7e-12 in Julia over 766781 steps of the cosine run. Any
``\lvert\Delta H\rvert`` reported at that same order is therefore arithmetic, not dynamics —
which is exactly what `cos` RK4 flow 1's 1.1e-11 against 6.2e-12 is. The Python's own closing
note makes the same point: a large growth ratio on an error still at round-off is "only the
random walk of the arithmetic".

## Three independent sources agree on the spectral radius

The explicit step size of every run in the manuscripts comes from
``\Delta t \le 2\sqrt{2}/\rho``, so ``\rho`` is the single number the whole RK4 comparison
rests on. It is now pinned from three directions:

| ``N`` | notes §5 | `kdvsim.py` | `stability_limit` |
|--:|--:|--:|--:|
| 12 | 115 | 115.4 | 115.4 |
| 16 | 274 | 274.2 | 274.2 |
| 24 | 907 | 907.2 | 907.2 |
| 32 | 2170 | 2170.0 | 2170.0 |

at ``p = 3`` on the first flow, from an ``O(1)`` single-mode initial condition. The value is
sensitive to the amplitude — ``u_0 = 0.1\sin x`` gives 112.5, 270.0, 900.2, 2160.1, off by two
per cent — which is why the notes now state their initial datum. Both the Julia test suite and
`kdvsim.py` assert these four figures, so a change in either assembly fails a test rather than
drifting silently away from the manuscript.

## Divergences found, and fixed

  - **`integrate` dropped the tail of every run.** `nout = nsteps ÷ stride` ran
    `nout * stride` steps while the tables went on reporting `nsteps`: 766400 of a claimed
    766781 for the cosine run, ending at ``T = 99.95`` rather than 100. Now `cld` with a final
    short window, as `plot_kdv_energies.py` does. The reported numbers do not move at the
    printed precision, but they were not measuring what they said.
  - **The explicit step size was clamped.** `scripts/kdv.jl` used
    `min(dt_rk4, case.dt)` where the Python uses `dt_rk4` outright. Invisible on four of the
    six cases, because there the stability limit is the smaller; on `soliton5` it forced
    0.0625 instead of 0.065 and hid the point the benchmark makes, which is that a wide,
    small-amplitude field costs an explicit method no step-count penalty at all.
  - **Window rounding.** `NT ÷ NOUT` against the Python's `round(NT / NOUT)`.
  - **`casimir_gradient` was exported and never defined.** It appeared in the export list of
    `src/GeometricBrackets.jl` with no method anywhere in `src/`, which under
    `checkdocs = :exports` is a latent documentation failure as well as a dangling name. The
    export is removed; nothing referenced it.

## Errata found in the Python

  - **`verify_burgers_discretisation.py` integrated four times too slowly.** Section 4
    transformed to ``\bar{u} = 2\sqrt{u}`` but used the bracket ``\mathbb{K}/4`` that belongs
    to ``\bar{u} = \sqrt{u}``. The transformation and the factor move together:
    ``\sqrt{u}`` with ``\mathbb{K}/4``, or ``2\sqrt{u}`` with ``\mathbb{K}`` and no factor.

    Nothing in the script could see it, and this is the interesting part: a constant rescaling
    of a Poisson vector field preserves every Casimir, every energy bound and every
    convergence *rate*, so all three of its checks passed. Only a comparison against the
    pushforward of the ``u``-space field catches it. That comparison is now a test here — see
    the `LagrangeSpace` Burgers testset — and a check there.

    The package did not inherit the error, its dynamics being carried in ``u``; but
    `to_sqrt_variables` was in the ``2\sqrt{u}`` convention while the docstring beside it
    displayed ``\mathbb{K}/4``. Both are now the manuscript's ``\bar{u} = \sqrt{u}``.
  - A stale comment in `kdvsim.py` put the explicit cosine run at "two million steps"; it is
    766781.

## The Python prototypes, converted and retired

All eighteen Python scripts were converted to Julia and each was run against its original
before the Python was removed. The harness `scripts/check.jl` reproduces `common.py`'s output
format byte for byte — the same two leading spaces, the same three before the detail, the same
`summary` wording and exit codes — so the comparison was a plain `diff` rather than a reading
of two tables side by side.

Two standards of agreement apply, and which one is available depends on the script:

  - **Deterministic sections** — fixed initial data, fixed meshes, symbolic computation —
    were compared digit for digit.
  - **Random-probe sections** cannot be: Julia's random stream is not Python's, and no attempt
    was made to make it so. There the verdicts and the orders of magnitude were compared, and
    wherever the Python used a random probe merely as a witness the Julia prefers the
    deterministic algebras it already provides — ``\mathfrak{se}(3)``, ``\mathfrak{so}(N)`` —
    which *do* agree exactly.

Each entry is a Julia script under `scripts/`; see [The verification scripts](@ref) for what
each one establishes and where the theory behind it is written down.

| script | agreement |
|:--|:--|
| `verify_kdv_continuous` | identical but for `**` vs `^` in two printed expressions |
| `verify_kdv_discrete` | 46 PASS / 0 FAIL both; support widths, seam jumps and the order-6 contracted Jacobiator identical |
| `verify_kdv_semidiscrete` | 240 PASS / 0 FAIL both; convergence tables to 3–4 s.f.; the ``n_q`` sweep reproduces exactly, with ``\lvert K_2 + K_2^\top\rvert`` identically zero |
| `verify_kdv_timedisc` | 73 PASS / 0 FAIL both; the Ge-Marsden table to 3 s.f. on every deterministic row |
| `verify_kdv_jacobi_family` | all verdicts identical; the flat ``0.4212`` / ``0.4665`` table exact. See the residual-convention erratum below |
| `verify_kdv_aliasing` | identical but for `**` vs `^`, numpy's `np.float64()` repr, and round-off |
| `verify_kdv_miura` | 72 PASS / 0 FAIL both; ``\operatorname{cond}(DM)``, the Hill eigenvalue ``-0.3287`` and the whole drift table match |
| `verify_kdv_nambu` | 48 PASS / 0 FAIL both **at conversion**; **byte-identical** on every deterministic detail, including both symbolic no-go systems (127 and 268 equations, 0 solutions). The Julia now runs 76: section 7, Nambu's Liouville criterion, was added after the Python was retired and so has no counterpart to compare against |
| `verify_kdv_bea` | 64 PASS / 0 FAIL both; the order test (3.00, 4.99) and the dispersion table identical, exponents 6.07 / 8.11 / 10.22 against ``2p+2`` |
| `verify_burgers_jacobi_family` | labels and verdicts identical |
| `verify_burgers_leibniz` | identical but for `**` vs `^` |
| `verify_burgers_entropy_casimir` | identical but for `**` vs `^` |
| `verify_burgers_discretisation` | **every deterministic digit identical** — error tables, all convergence rates, energy rates |
| `verify_liepoisson_structure_constants` | the headline flat ``\approx 0.73`` / ``0.71`` violation identical at every refinement |
| `verify_liepoisson_4bracket` | all verdicts and all zero/nonzero patterns identical |
| `verify_gardner_4bracket` | identical but for `True`/`False` capitalisation; every rank matches |
| `verify_dirac_reduction` | 110 PASS / 0 FAIL both; every structural column identical. See the fixed-point cross-check below |
| `search_dirac_variants` | 15 PASS / 0 FAIL both; B1a, B1b and B3 byte-identical (**46 candidates / 12 hits** at K=1, **834 / 0** at K=2) |

### The Dirac machinery, cross-checked at a fixed point

`verify_dirac_reduction`'s numbers ride on a random surface point, so a diff of it only
establishes matching verdicts. Agreement was therefore established a second way: at a **fixed
rational point with no RNG on either side**, comparing `src/dirac.jl` and `src/hierarchical.jl`
directly against `diractools.py` and `femtools.py`. Every value matched digit for digit —
the reduced Jacobiator ``16134943910670/117497863``, ``\det C = 10490880625/50176``,
``\text{structure-constant residual} = 10368/49``, and the broken-hierarchical mass entry
``M_{11} = 4/15``. That last one pins the exact shifted-Legendre and Lagrange assembly against
the SymPy version it replaces. It is now a regression test in `test/dirac_tests.jl`.

## Further errata found in the Python

  - **``\mathfrak{so}(3)``'s degeneracy was attributed to the wrong cause.**
    `verify_kdv_jacobi_family.py` and `common.py` both assert that *every* three-dimensional
    antisymmetric bracket satisfies the Jacobi identity identically. It does not: 200 of 200
    random antisymmetric ``c`` in dimension three fail the structure-constant residual.

    The real reason ``\mathfrak{so}(3)`` is a useless positive control is narrower and more
    interesting. It lies in the six-parameter family ``c_{ij}^k = \epsilon_{ijl} n^{lk}`` with
    ``n`` **symmetric** — Bianchi class A — every member of which satisfies Jacobi, and the
    perturbations one reaches for stay inside it: rescaling a generator, or rescaling a single
    structure constant, both keep ``n`` symmetric and diagonal. So it goes on passing after it
    looks broken. Drop the symmetry of ``n`` and 200 of 200 fail.

    The conclusion the Python drew is right; the stated reason is not. Corrected in the
    docstrings of `so3` and `structure_constant_residual`, and pinned by `test/algebras_tests.jl`.
  - **`common.py`'s two structure-constant residuals are not the same function.** The exact
    `structure_constant_residual` and `structure_constant_residual_float` agree precisely when
    ``C`` is antisymmetric in ``(i,j)``; off that locus one is the other evaluated at ``C``
    transposed. `verify_kdv_jacobi_family.py`'s ``\beta/\alpha`` scan leaves that locus
    deliberately, so its off-axis numbers are this package's mirrored about ``\beta/\alpha =
    1/2``, because ``C(1,t)^\top = -C(1,1-t)`` in that family.

    Nothing turns on it: only ``\alpha = 2\beta`` gives an antisymmetric ``C``, so only there
    is the tensor a candidate set of structure constants at all, and there both conventions
    give ``0.4212``, with the family minimum ``0.4184`` either way. The package keeps the exact
    convention, which every Lie-Poisson script uses and which is verified against
    ``\mathfrak{se}(3)`` and ``\mathfrak{so}(N)``.
  - **A threshold on a round-off-limited quantity.** `verify_kdv_timedisc.py` checks that the
    Gonzalez correction vanishes for a quadratic Hamiltonian by testing the bare
    ``\big((H(y)-H(x)) - g(\bar{x})\cdot\Delta x\big) / \lvert\Delta x\rvert^2`` against
    ``10^{-14}``. The numerator is exactly zero in exact arithmetic, so that quantity is
    round-off *divided by* ``\lvert\Delta x\rvert^2`` — its size depends on how large a step
    happens to be drawn, and it varies by an order of magnitude across seeds. The Python reports
    exactly `0.0`, which it is not entitled to. This package measures the rank-one term
    ``\text{corr} \cdot \Delta x`` **relative to the gradient it corrects**, which sits at
    ``10^{-15}`` across seeds, and averages over five draws.
  - **A no-op line.** `verify_dirac_reduction.py:147` computes `ok = all(... or True ...)`,
    which is unconditionally `True`, and is immediately overwritten by the next line. Only the
    second line was ever meaningful; the Julia keeps only that.
  - **A missing suffix.** `verify_kdv_bea.py` ends with `summary("verify_kdv_bea")` where every
    other script passes its own filename.

## What the port improved

  - **`femtools.py` needed SymPy only for exact integrals of shifted Legendre products.** Those
    polynomials have *integer* coefficients and ``\int_0^1 \xi^n = 1/(n+1)``, so the whole
    assembly is rational arithmetic on coefficient vectors. `src/hierarchical.jl` does it over
    ``\mathbb{Q}`` with no CAS, which is what keeps a symbolic dependency out of this package
    altogether. The Python also routed its Lagrange shape functions through `float64` and
    `limit_denominator`; the Julia is exact end to end, and `M_{11} = 4/15` above shows the two
    agree anyway.
  - **The exact and floating-point twins collapsed.** `verify_dirac_reduction.py` carried a
    second copy of the Dirac machinery in numpy, and `femtools.py` a second copy of the
    hierarchical assembly with an SVD for the ``V_2`` complement. Both are one generic
    implementation here, parametric in the element type. After `galerkin_c` was rewritten from
    a direct ``O(n^5)`` sum to two staged contractions, the largest size the refinement study
    needs assembles **exactly** in 1.7 s, so the floating-point path is not needed at all.
  - **`sine_algebra` returned a dense ``d^3`` tensor holding only ``d^2`` nonzeros.** At
    ``N = 41`` that is 75 GB. `sine_coefficient(N, m, n)` gives the single nonzero coefficient
    directly. Doing so also exposed that the ``N^{-2}`` law is asymptotic in
    ``N \gg 2\pi(m \times n)``: at ``N = 11`` the observed order is 0.8, at ``N = 161`` it is
    1.98.
  - **One `KdVSys`, where the Python had three.** `kdvsim.Sys`, `verify_kdv_timedisc.Sys` and
    `verify_kdv_miura.Chart` were near-identical copies whose drift under the sign-convention
    change is recorded below. `scripts/kdvtools.jl` holds one.

## Julia-specific traps met on the way

Recorded because each produced a wrong answer or a failure to load, and the first is the only
one that did **not** announce itself:

  - `LinearAlgebra.dot(a, b)` conjugates its **first** argument, so numpy's `v.conj() @ (A @ v)`
    is `dot(v, A, v)` — the three-argument form, which also avoids materialising `A*v`. Writing
    `dot(conj(v), A*v)` conjugates twice and silently returns a plausible real number: it gave
    dispersion errors of `1.5e+02` where the correct form gives `1.8e-10`, and turned the
    ``2p+2`` exponents into ``-2``.
  - Broadcast orientation. ``\Phi`` is ``(n_b \times n_q)`` and the weights are ``(n_q,)``, so
    numpy's `W * P[d]`, which scales the *quadrature* axis, is `Φ .* transpose(W)`. This bit
    three times, always as a loud `DimensionMismatch` — but it would be silent if
    ``n_b = n_q``.
  - `let a, b = f()` does not destructure: it declares `a` uninitialised and binds the whole
    tuple to `b`. Write `let (a, b) = f()`.
  - Name collisions, all of which forced a rename in `scripts/`: `fld` with `Base.fld`, `Sys`
    with `Base.Sys`, `integrate` with the re-exported `GeometricBase.integrate`, and
    `drift`/`growth` with this package's own.
  - SymPy through `SymPyPythonCall`: `symbols("g[k,l]")` splits on the comma and returns a
    *tuple* (use `sympy.Symbol`); `Sym(name)` takes no assumptions (use
    `symbols(name, positive = true)`); and a `Vector` of `Eq` objects passed to `solve` trips a
    Matrix deprecation, so pass bare expressions, which sympy reads as ``= 0``.

## The sign convention was flipped, and both sides agree afterwards

The package and both manuscripts now use the textbook ``u_t + 6uu_x + u_{xxx} = 0``, so the
``\operatorname{sech}^2`` solitons are elevations. The older ``u_t = 6uu_x - u_{xxx}`` is
reached by ``u \to -u``.

The one step that cannot be guessed is the second structure: it does **not** simply change
sign. ``4u\partial_x + 2u_x - \partial_x^3`` becomes ``-4u\partial_x - 2u_x -
\partial_x^3`` — the transport part flips, the third derivative does not. That was derived
and checked in `verify_kdv_continuous.py`, which works in jet variables, *before* any prose
or code was touched; in the package it is the single character of `AffineBracket`'s `scale`,
from `2` to `-2`, `K0` being the ``-\partial_x^3`` block and keeping its own sign.

Consequences, all verified:

  - ``H_1 = \tfrac12\int(u_x^2 - 2u^3)``, ``\delta H_1/\delta u = -3u^2 - u_{xx}``;
    ``H_2`` unchanged; ``H_3 = \int(\tfrac52 u^4 - 5uu_x^2 + \tfrac12 u_{xx}^2)``.
  - The Magri rung ``\mathbb{P}^2 g = 2\mathbb{P}^1 \partial H_2/\partial\hat{u}`` becomes
    ``-2``, for the same reason.
  - The Miura map becomes ``u = -(v^2 + v_x)``, its Jacobian ``-(2v + \partial_x)``. The
    overall sign cancels in ``\mathbb{L}\mathbb{P}^1\mathbb{L}^T``, so the bracket is
    unchanged; the chart is not, and the Hill potential becomes ``-u_h``.
  - The image of the Miura map moves to ``C_{0,d} \le 0`` — and the solitons move with it, so
    they are excluded exactly as before. **The sign convention was never what restricted the
    chart.** The obstruction is spectral: a soliton is a reflectionless potential *with* a
    bound state, and the chart covers the fields whose Hill operator has none.
  - The Hill threshold for ``u = c + \sin x + 0.4\cos 2x`` at ``N = 20``, ``p = 3`` moves
    from ``+0.470894`` to ``-0.328741``, the admissible side becoming the negative one. The
    magnitude differs because ``\pm(\sin x + 0.4\cos 2x)`` are not translates of one another.
    Measured by bisection it is ``-0.32893``, four digits rather than six: at the threshold
    ``\psi`` acquires a zero and ``v = \psi_x/\psi`` is unbounded, so the Newton iteration
    and not the criterion sets the resolution.

**Julia and Python agree after the flip**, on `soliton1` at ``N = 128``:
``H_1 = -6.39998180``, ``H_2 = +2.66666660``, ``C_0 = +4.00000000`` in both, and
``\rho = 115.4, 274.2, 907.2, 2170.0`` in both. ``H_1`` and ``H_2`` are the values the *old*
convention reported on the *old* data, since each Hamiltonian carries the matching sign; only
``C_0`` flipped.

## The Miura chart now covers every example

The restriction was never the sign convention, so flipping it did not help — and the fix is
not a shift bolted on but the spectral parameter the Riccati substitution always had:
``u = -(v^2 + v_x) - \lambda`` turns into ``(-\partial_x^2 - u)\psi = \lambda\psi``, so
``\lambda`` is an eigenvalue parameter of the Hill operator that decides invertibility and a
preimage exists exactly when ``\lambda < \lambda_0``.

Measured, at ``p = 3``, with `miura_lambda` a tenth below each threshold:

| example | ``C_{0,d}`` | ``\lambda_0`` | at ``\lambda = 0`` | at ``\lambda`` | round trip |
|:--|--:|--:|:--|:--|--:|
| ``\cos x``, ``N = 20`` | 0.000 | -0.3785 | none | reached | 1.2e-15 |
| ``\sin x + 0.4\cos 2x``, ``N = 20`` | 0.000 | -0.3287 | none | reached | 1.3e-15 |
| single soliton, ``N = 64`` | +4.000 | -0.9999 | none | reached | 1.1e-15 |
| two-soliton, ``N = 64`` | +8.000 | -1.4394 | none | reached | 1.6e-15 |

and the pushforward stays *exactly* antisymmetric with its Jacobiator at ``10^{-16}`` for
``\lambda = -0.5, -2, -5``. Both the Julia test suite and `verify_kdv_miura.py` §6b assert
this.

Two caveats worth stating rather than discovering:

  - ``\mathbb{L}\mathbb{P}^1\mathbb{L}^T`` is **not** the Galerkin ``\mathbb{P}^2 -
    4\lambda\mathbb{P}^1``; measured, they differ by 30-57 %. That difference *is* the Jacobi
    defect the Miura construction exists to remove, so the identity must not be asserted.
  - In the ``\hat{v}`` chart ``C_{0,d}`` is the mKdV momentum, not a Casimir, so a Miura run
    loses it — 2.1e-6 on the single soliton. The two methods exchange roles exactly as at
    ``\lambda = 0``: ``\tilde{H}`` is quartic in ``\hat{v}`` and ``C_{0,d}`` quadratic, so the
    discrete gradient holds the energy (1.3e-13) and the symplectic midpoint rule the mass.

### What the flip cost: four duplicated formulas

Each of these reimplements something for speed and had to be changed in step. Every one
produced a plausible-looking wrong answer rather than an error, and each was caught by a
test rather than by reading:

  - `invariants(::KdVSystem, …)` and `invariants(::MiuraSystem, …)` in `src/diagnostics.jl`,
    which assemble ``H_1`` from one field evaluation instead of calling
    `hamiltonian(KdVHamiltonian1(), …)`. Left stale, they made implicit midpoint report a
    373 % drift in a quantity the flow conserves exactly.
  - `kdvsim.Sys.invariants` and `kdvsim.Sys.H1`, the same duplication on the Python side.
  - `kdvsim.MiuraSys.Mmap`, which reimplements `splinetools.miura_map`.
  - Local ``H_1`` lambdas in `verify_kdv_semidiscrete.py` and `verify_kdv_timedisc.py`.

The lesson is the same as the Burgers factor above: a sign that moves in one copy and not
the other is invisible to every structural test and shows up only as a drift.

## The notes' own error list, dispositioned

`discrete-kdv-brackets-notes.tex` carries a standing list of known defects. Every item is
accounted for here.

| item | in the package? | disposition |
|:--|:--|:--|
| Errors 1 — mass-Casimir index and factor | no | paper-only. `MassCasimir` sidesteps it, arguing from ``g^T \mathbb{P}^2 \partial H_2/\partial\hat{u} = 0`` pointwise rather than ``g^T\mathbb{P}^2 = 0``, which is the correct weaker claim. |
| Errors 2 — dangling RK4 cross-reference | n/a | **closed.** The measurement was missing, not the reference; §5 now carries it, with the ``\rho`` table above. |
| Errors 3 — strict ``C_{0,d} > 0`` | **yes**, six sites | **fixed** in both, to ``C_{0,d} \ge 0`` with the equality case named. |
| Errors 4 — zero-mode theorem skips ``N = 4`` | no | paper-only; no counterpart here. |
| Numbers 1-3 | no | paper-only; each needs a measurement in the notes' own sections. |
| Numbers 4 — unstated initial data | partly | the time-discretisation datum is now stated. The disagreement between the *Miura* runs' data is still open. |
| Overstatements 1, 5 | no | paper-only. |
| Overstatements 2 — critical set of ``D\mathcal{M}_h`` | **yes** | **fixed.** Condition numbers are now given as evidence, and the exact statement that does hold is added: ``\int v_h`` is a Casimir of ``\mathbb{P}^1`` in this chart, so the flow cannot reach the degeneracy from data off it. Now a test. |
| Overstatements 3 — Ge-Marsden hypotheses | **yes**, four sites | **fixed** in both; the theorem is now the reason to expect the trade-off, not a proof of it here. |
| Overstatements 4 — modified Hamiltonian | no | paper-only. |
| Wording 1-5 | no | paper-only. |

## Confirmed correct

Checked and reproduced, and worth recording so that their absence above is not read as
neglect:

  - the Burgers bracket ``\mathbb{J} = \sqrt{u}\,\mathbb{K}\sqrt{u}`` reproducing
    ``u_t = 3uu_x`` at second order, and its exact Casimir ``2\int\sqrt{u}``;
  - the even-rank argument and the spurious sawtooth Casimir at even ``N``, with the
    odd-``N`` advice;
  - the whole Miura chain: ``C_{0,d} = \int v_h^2`` exactly, the sharp criterion
    ``\lambda_0(-\partial_x^2 + u_h) > 0``, two Floquet preimages with opposite ``\int v``,
    the threshold ``-\lambda_0 = 0.470894``, and that none of the standard examples is in the
    image;
  - both KdV flows reproducing the right-hand side of the KdV equation, and the Magri
    hierarchy step;
  - the Jacobiator of the Galerkin second bracket flat at ``\approx 0.42`` under refinement,
    against round-off for the Miura bracket.

## Two latent faults found by making the tabulation sparse

Neither was reachable before, which is why neither had been noticed.

  - **`project!` never worked on a `LagrangeSpace`.** `mass_factorization` returned a bare
    dense `Cholesky`, and `mass_solve!` — which the generic `project!` calls — has methods only
    for a `MassOperator`. Every in-place projection onto a Lagrange space was a `MethodError`.
    Nothing in the suite exercised it. It works now, and is tested.
  - **`AffineBracket` densified the tabulation it was handed.** Its `Ψ` field was declared
    `Matrix{T}` and its constructor called `Matrix(Ψ)`, so the *spline* space's already-sparse
    table was thrown away for the second KdV bracket and the first Camassa-Holm one — both of
    which contract it on every Newton iteration. Measured at ``N = 384``, one `jacobian` of
    the second flow: 7.24 ms before, 1.26 ms after, identical results. `bracket_directional`
    alone is 25 times faster; the `LagrangeSpace` assemblies about 30 times.

    The two ``O(N^3)`` tensor routines densify locally and on purpose — they random-access the
    table rather than contracting it — and are off the time loop by construction.

## The four-bracket manuscript's scripts, folded in

The `verify_fourbracket_*` scripts are the odd ones out in the table above, because there is no
row for them: they had **no Python ancestor**. `poisson-brackets-from-four-brackets.tex` arrived
with a Julia suite of its own — a `BracketChecks.jl` carrying spectral and 8th-order
finite-difference differentiation on the two-torus, a `CheckSet` harness, four drivers and a
`run_all.jl`, stdlib-only and with no `Project.toml`. So the standard of agreement here is not
"matches a retired prototype" but "matches itself after being refactored onto the package".

Where each piece went:

| was | is now |
|:--|:--|
| `BracketChecks.jl` grids and differentiation | `src/torus.jl` |
| its `cbracket`, `gardner_x`, `gardner_y`, and the bracket forms the drivers defined inline | `src/fourbrackets.jl` |
| `03`'s `metric_bracket` | `src/metriplectic.jl`, with `kulkarni_nomizu` alongside it |
| its `CheckSet` / `check!` / `report` harness | `scripts/check.jl`, as `check_exact` and `check_refined` |
| the test fields, carried in three copies | `scripts/torustools.jl` |
| `01`–`04` and `run_all.jl` | the four `verify_fourbracket_*.jl`, and rows in `scripts/run_all.jl` |

Every check was run before and after. **Every finite-difference residual and every observed rate
reproduced to all printed digits** — `1.52e-08` at `204x`, `6.30e-11` at `251x`, the twenty
Monte-Carlo minima of Proposition 2.1 from `7.303e-09` up to `1.747e+01`, the whole refinement
table of the log-entropy section. The spectral residuals moved, in the band `1e-14` to `1e-16`,
and uniformly *downward*: `4.50e-15` to `3.05e-15` for cyclicity, `2.23e-14` to `9.21e-15` for
the Gardner total-derivative identity, `1.07e-15` to `5.28e-16` for Section 6. That is the one
place a difference was expected. The original evaluated each spectral derivative by a naive DFT
and its inverse per call, at ``O(N^4)``; `src/torus.jl` assembles the differentiation matrix once
and applies it, so fewer floating-point operations accumulate and roundoff is smaller. Nothing
about the identities changed.

One residual went the other way, from `0.00e+00` to `2.40e-16` — the alternative pairing of
Section 6. An exact zero became a roundoff-level nonzero, which is the same claim.

### What the fold-in turned up

Three defects, none affecting a verdict:

  - `04_log_entropy_weight.jl` defined a `weighted_4bracket`, gave it a docstring, and never
    called it; the refinement loop below re-inlined its body by hand. There is now one
    definition, in `src/fourbrackets.jl`, and both sites call it.
  - `02_convergence.jl` defined `resolutions = (32, 64, 128, 256)` and then invoked `main` with
    a duplicated literal tuple, so editing the constant would have changed nothing.
  - The test fields `Au`, `Bu`, `Cu`, `chi`, `uu` were copy-pasted across three drivers, and
    `Cu` a fourth time as a local `c3` inside one study of `02`. `04`'s `ushift` was
    character-for-character `01`'s `uu` under a second name. All of it is now
    `scripts/torustools.jl`.

Two pieces of dead weight went with them: `BracketChecks.jl` imported `Random` and
`LinearAlgebra` and used neither, and `03_metriplectic_positivity.jl` imported `LinearAlgebra`
without calling anything from it.

### What the fold-in improved

`BracketChecks.Grid` stored its two derivative operators as `Dx::Function` and `Dy::Function`,
which leaves the field type abstract at every call site, and computed a spectral derivative by
transforming and inverse-transforming with explicit quadruple loops — ``O(N^4)`` per derivative,
which is why the original could only afford `spectral(16)`. [`TorusGrid`](@ref) stores one
differentiation matrix: dense for the spectral scheme, sparse circulant for the stencil. A
derivative is `D * f` or `f * transpose(D)`, so both schemes share a type instead of forming a
hierarchy, and the matrix is antisymmetric — exactly for the centred stencil — which is where
the antisymmetry of [`canonical_bracket`](@ref) now comes from rather than being asserted
separately.

The structural fact the refactoring exposed, and which the original's copies obscured: a
two-bracket **is** its four-bracket with the entropy in the second and fourth slots.
[`gardner_2bracket`](@ref) is defined as `gardner_4bracket(g, a, s, b, s)` and
[`symmetric_2bracket`](@ref) likewise, rather than written out a second time; and the weighted
brackets are the symmetric densities times the weight. `test/fourbrackets_tests.jl` asserts both
identifications with `==`, not `≈`.

One claim was added that the original did not make. `metriplectic_bracket` evaluates equation
(2.6) in closed form at ``O(n^2)``; `kulkarni_nomizu` now builds the four-index tensor from the
definition so that the closed form can be checked against what it is supposed to implement.
Without that, the positivity sweep would be evidence about a rearrangement rather than about the
bracket. Worst relative difference over dimensions two to five: `1.12e-14`.

## What cannot be verified

**Camassa-Holm has no reference.** Grep for `camassa`, `peakon`, `holm` or `helmholtz` over
both `Scripts/` directories returns nothing, and neither manuscript covers it. The
implementation here is a *prototype* in the strict sense: it is internally consistent and its
structural tests pass, but there is no independent statement of what it should reproduce. It
should not be read as verified.
