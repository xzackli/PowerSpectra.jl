
## SpectralArray and SpectralVector

This package wraps outputs in a custom [`SpectralArray`](@ref) (based on [OffsetArray](https://github.com/JuliaArrays/OffsetArrays.jl)), which provides arbitrary indexing but by default makes an array 0-indexed. This is useful for manipulating angular spectra, as although Julia's indices start at 1, multipoles start with the monopole ``\ell = 0``. The type [`SpectralVector`](@ref) is an alias for a one-dimensional SpectralArray, i.e., `SpectralArray{T,1}`. 

```julia-repl
julia> using PowerSpectra

julia> cl = SpectralVector([1,2,3,4])
4-element SpectralVector{Int64, Vector{Int64}} with indices 0:3:
 1
 2
 3
 4

julia> cl[0]
1
```

You can also specify arbitrary indices, like OffsetArray. In the next example, we index the rows by the range `0:1` and the columns by `5:8`.

```julia-repl
julia> A = SpectralArray(ones(2,4), 0:1, 5:8)
2×4 SpectralArray{Float64, 2, Matrix{Float64}} with indices 0:1×5:8:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0

julia> A[0, 8]
1.0
```
Slicing a [`SpectralArray`](@ref) makes that sliced dimension become 1-indexed, which loses the index information. For example, slicing a `SpectraVector` just produces a Vector. If you want to produce a SpectralArray that preserves the indices, you can use `IdentityRange` from IdentityRanges.jl.

```julia-repl
julia> x = SpectralVector(ones(4), 0:3)
4-element SpectralVector{Float64, Vector{Float64}} with indices 0:3:
 1.0
 1.0
 1.0
 1.0

julia> x[2:3]
2-element Vector{Float64}:
 1.0
 1.0

julia> using IdentityRanges

julia> x[IdentityRange(2:3)]
2-element SpectralVector{Float64, Vector{Float64}} with indices 2:3:
 1.0
 1.0

```

The one major difference is that matrix multiplication and linear solve operator `\` are specialized for `SpectralArray` to ignore the monopole and dipole, as pseudo-``C_{\ell}`` methods do not handle those multipoles very well.

You can wrap an array `A` without copying by just calling `SpectralArray(A)`.
