# The doctests of this package, in its docstrings and in the manual under `docs/src`, as the
# `Doctests` job of `CI.yml` runs them.
#
# Documenter evaluates a page's `@meta` block in `Main`, and a `@safetestset` file runs in a module
# of its own, so `GeometricBrackets` is imported into `Main` first.

using GeometricBrackets
using Documenter: DocMeta, doctest

@eval Main import GeometricBrackets

DocMeta.setdocmeta!(GeometricBrackets, :DocTestSetup, :(using GeometricBrackets); recursive = true)

doctest(GeometricBrackets)
