"""
    _apply_P_h!(Pf, f, h, ci, li, h₁, h₂)

Write Arakawa's Jacobian `[f, h]` into `Pf`, matrix-free, on a doubly periodic grid with
`CartesianIndices` `ci`, `LinearIndices` `li` and spacings `h₁` and `h₂`. The vectors `Pf`,
`f` and `h` hold one value per node, in the order of `li`.
"""
function _apply_P_h!(
        Pf::AbstractVector, f::AbstractVector, h::AbstractVector, ci, li, h₁, h₂)
    nx, ny = size(ci)
    length(Pf) == length(f) == length(h) || throw(DimensionMismatch())
    @inbounds for ij in li
        i, j = Tuple(ci[ij])
        i₋ = mod1(i-1, nx)
        j₋ = mod1(j-1, ny)
        i₊ = mod1(i+1, nx)
        j₊ = mod1(j+1, ny)
        jpp = ((f[li[i₊, j]] - f[li[i₋, j]]) * (h[li[i, j₊]] - h[li[i, j₋]])
               -
               (f[li[i, j₊]] - f[li[i, j₋]]) * (h[li[i₊, j]] - h[li[i₋, j]])
        )
        jpc = (f[li[i₊, j]] * (h[li[i₊, j₊]] - h[li[i₊, j₋]])
               -
               f[li[i₋, j]] * (h[li[i₋, j₊]] - h[li[i₋, j₋]])
               -
               f[li[i, j₊]] * (h[li[i₊, j₊]] - h[li[i₋, j₊]])
               +
               f[li[i, j₋]] * (h[li[i₊, j₋]] - h[li[i₋, j₋]])
        )
        jcp = (f[li[i₊, j₊]] * (h[li[i, j₊]] - h[li[i₊, j]])
               -
               f[li[i₋, j₋]] * (h[li[i₋, j]] - h[li[i, j₋]])
               -
               f[li[i₋, j₊]] * (h[li[i, j₊]] - h[li[i₋, j]])
               +
               f[li[i₊, j₋]] * (h[li[i₊, j]] - h[li[i, j₋]])
        )
        Pf[ij] = 1/(12 * h₁ * h₂) * (jpp + jpc + jcp)
    end
end

"""
    _apply_P_ϕ!(Pf, f, v, ϕ, ci, li, h₁, h₂)

Write `[f, v²/2 + ϕ]` into `Pf`, as `_apply_P_h!` does for the Hamiltonian
`h[i, j] = ϕ[i] + v[j]^2 / 2`, from the velocity nodes `v` of length `nv` and the potential `ϕ`
of length `nx`, without forming `h`.
"""
function _apply_P_ϕ!(Pf::AbstractVector, f::AbstractVector,
        v::AbstractVector, ϕ::AbstractVector, ci, li, h₁, h₂)
    nx, nv = size(ci)
    length(Pf) == length(f) == nx*nv || throw(DimensionMismatch())
    length(v) == nv || throw(DimensionMismatch())
    length(ϕ) == nx || throw(DimensionMismatch())
    @inbounds for ij in li
        i, j = Tuple(ci[ij])
        i₋ = mod1(i-1, nx)
        j₋ = mod1(j-1, nv)
        i₊ = mod1(i+1, nx)
        j₊ = mod1(j+1, nv)
        jpp = ((f[li[i₊, j]] - f[li[i₋, j]]) * 0.5 * (v[j₊]^2 - v[j₋]^2)
               -
               (f[li[i, j₊]] - f[li[i, j₋]]) * (ϕ[i₊] - ϕ[i₋])
        )
        jpc = (f[li[i₊, j]] * 0.5 * (v[j₊]^2 - v[j₋]^2)
               -
               f[li[i₋, j]] * 0.5 * (v[j₊]^2 - v[j₋]^2)
               -
               f[li[i, j₊]] * (ϕ[i₊] - ϕ[i₋])
               +
               f[li[i, j₋]] * (ϕ[i₊] - ϕ[i₋])
        )
        jcp = (f[li[i₊, j₊]] * (ϕ[i] + 0.5 * v[j₊]^2 - ϕ[i₊] - 0.5 * v[j]^2)
               -
               f[li[i₋, j₋]] * (ϕ[i₋] + 0.5 * v[j]^2 - ϕ[i] - 0.5 * v[j₋]^2)
               -
               f[li[i₋, j₊]] * (ϕ[i] + 0.5 * v[j₊]^2 - ϕ[i₋] - 0.5 * v[j]^2)
               +
               f[li[i₊, j₋]] * (ϕ[i₊] + 0.5 * v[j]^2 - ϕ[i] - 0.5 * v[j₋]^2)
        )
        Pf[ij] = 1/(12 * h₁ * h₂) * (jpp + jpc + jcp)
    end
end
