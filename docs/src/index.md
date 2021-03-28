```@meta
CurrentModule = AngularPowerSpectra
```

# AngularPowerSpectra

[AngularPowerSpectra.jl](https://github.com/xzackli/AngularPowerSpectra.jl) is a package for power spectrum analysis on the sphere. It computes mode-coupling matrices and covariance matrices for TT, TE, and EE spectra, using pseudo-``C_{\ell}`` methods (i.e. [Hivon et al. 2002](https://arxiv.org/abs/astro-ph/0105302), [Efstathiou 2006](https://arxiv.org/abs/astro-ph/0601107), [Hamimeche and Lewis 2008](https://arxiv.org/abs/0801.0554)). It can also compute  beam matrices in the QuickPol formalism ([Hivon et al. 2017](https://arxiv.org/abs/1608.08833)).

### Conventions: SpectralArray and SpectralVector

This package wraps outputs in a custom `SpectralArray`, which is a simple array type that just makes the array 0-indexed. This is very useful for manipulating angular spectra, as multipoles start with the monopole ``\ell = 0``. The type `SpectralVector` is an alias for a one-dimensional array, `SpectralArray{T,1}`. The one major difference is that matrix multiplication and linear solve operator `\` are specialized for `SpectralArray` to ignore the monopole and dipole, as pseudo-``C_{\ell}`` methods do not handle those multipoles very well.

You can wrap an array `A` without copying by just calling `SpectralArray(A)`.