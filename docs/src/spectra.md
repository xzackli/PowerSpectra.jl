```@meta
CurrentModule = AngularPowerSpectra
```

# Spectral Analysis
In this section, we describe how one can estimate unbiased cross-spectra from masked maps using this package. 

## Mode-Coupling Matrices

If you compute the cross-spectrum of masked maps, the mask will couple together different modes. This biased estimate of the true spectrum is termed the pseudo-spectrum ``\widetilde{C}_{\ell}``, 
```math
\widetilde{C}_{\ell} = \frac{1}{2\ell+1} \sum_m \mathsf{m}^{i,X}_{\ell m} \mathsf{m}^{j,Y}_{\ell m},
```
where ``\mathsf{m}^{i,X}_{\ell m}`` are the spherical harmonic coefficients of the masked map ``i`` of channel ``X \in \{T, E\}``. In the pseudo-``C_{\ell}`` estimator, one seeks an estimate ``\hat{C}_{\ell}`` of the true spectrum that is related to the pseudo-spectrum by a linear operator,
```math
   \widetilde{C}_{\ell} = \mathbf{M}^{XY}(i,j)_{\ell_1 \ell_2} \hat{C}_{\ell}
```
where ``\mathbf{M}^{XY}(i,j)_{\ell_1 \ell_2}`` is the mode-coupling matrix between fields ``i`` and ``j`` for spectrum ``XY \in \{TT, TE, EE\}``. Applying the inverse of the mode-coupling matrix to the pseudo-spectrum ``\widetilde{C}_{\ell}`` yields an unbiased and nearly optimal estimate ``\hat{C}_{\ell}`` of the true spectrum. To compute the mode-coupling matrix, one needs

* ``XY \in \{TT, TE, EE\}``, the desired spectrum
* ``\mathsf{m}^{i,X}_{\ell m}``, spherical harmonic coefficients of the mask for map ``i``, mode ``X``
* ``\mathsf{m}^{j,Y}_{\ell m}``, spherical harmonic coefficients of the mask for map ``j``, mode ``Y``

A basic functionality of this package is to compute this matrix. Let's look at a basic example of the cross-spectrum between two intensity maps.

```julia
# get some example masks
using Healpix, AngularPowerSpectra
mask1 = readMapFromFITS("test/data/mask1_T.fits", 1, Float64)
mask2 = readMapFromFITS("test/data/mask2_T.fits", 1, Float64)

# compute TT mode-coupling matrix from mask harmonic coefficients
M = mcm(:TT, map2alm(mask1), map2alm(mask2))
```

Similarly, one could have specified the symbol `:TE`, `:ET`, or `:EE` for other types of cross-spectra[^1].
The function `mcm` returns a `SpectralArray{T,2}`, which is just a simple array wrapper that makes the array 0-indexed. That means `M[ℓ₁, ℓ₂]` corresponds to the mode-coupling matrix entry ``\mathbf{M}_{\ell_1, \ell_2}``. If you want to access the underlying array, you can use `mcm.parent`. One can optionally truncate the computation with the `lmax` keyword, i.e. `mcm(:TT, mask1, mask2; lmax=10)`. 

[^1]: You can combine symbols, in cases where you're looping over combinations of spectra, by using `Symbol`.
    ```julia-repl
    julia> Symbol(:T, :T)
    :TT
    ```

Now one can apply a linear solve to decouple the mask.
```julia
# generate two uniform maps
nside = mask1.resolution.nside
npix = nside2npix(nside)
map1 = Map{Float64, RingOrder}(ones(npix))
map2 = Map{Float64, RingOrder}(ones(npix))

# mask the maps with different masks
map1.pixels .*= mask1.pixels
map2.pixels .*= mask2.pixels

# compute the pseudo-spectrum, and wrap it in a SpectralVector
alm1, alm2 = map2alm(map1), map2alm(map2)
pCl = SpectralVector(alm2cl(alm1, alm2))

# decouple the spectrum
Cl = M \ pCl
```
**Note that this performs the linear solve starting at** ``\mathbf{\ell = 2}``, setting ``\hat{C}_{\ell < 2} = 0``. The linear solve operator on `SpectralArrays` is specialized to do this. Most of the time you want to avoid the monopole and dipole, which tend to be very large relative to anisotropies. **You should subtract the monopole and dipole from your maps.** If you really want to perform the linear solve on the monopole and dipole, you can use the underlying arrays to decouple the spectrum, i.e. `Cl_starting_from_zero = M.parent \ pCl.parent`. 

## API

```@docs
mcm
```
