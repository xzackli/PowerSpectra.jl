```@meta
CurrentModule = PowerSpectra
```

# PowerSpectra

[PowerSpectra.jl](https://github.com/xzackli/PowerSpectra.jl) is a package for power spectrum analysis on the sphere. It computes mode-coupling matrices and covariance matrices for TT, TE, and EE spectra, using pseudo-``C_{\ell}`` methods (i.e. [Hivon et al. 2002](https://arxiv.org/abs/astro-ph/0105302), [Efstathiou 2006](https://arxiv.org/abs/astro-ph/0601107), [Hamimeche and Lewis 2008](https://arxiv.org/abs/0801.0554)). It can also compute  beam matrices in the QuickPol formalism ([Hivon et al. 2017](https://arxiv.org/abs/1608.08833)).

This package makes use of a special array type, which provides an indexing convention. We provide an introduction and some examples here.

## Convention: SpectralArray and SpectralVector

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

The SpectralArray has special operations defined on it, for the manipulation and application of mode-coupling matrices. For the majority of tasks, you will want to have ``\ell_{\mathrm{min}}=0``, so it's sufficient to just wrap your array without any other arguments, i.e. `SpectralArray(A)` or `SpectralVector(v)`. For advanced use, take a look at [SpectralArray and SpectralVector](@ref).
