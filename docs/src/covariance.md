```@meta
CurrentModule = PowerSpectra
```

# Covariance Estimation

The covariance between two spectra ``\textrm{Cov}(\hat{C}_{\ell}^{XY,i,j}, \hat{C}_{\ell}^{WZ,p,q})`` for channels ``X,Y,W,Z \in \{T, E\}`` obtained from masked maps can be expressed using
* spherical harmonic coefficients of the masks of the four maps ``i, j, p, q`` involved in the covariance
* assumed signal spectrum ``C_{\ell,\mathrm{tot}}^{XY}``, for example ``C_{\ell}^{\mathrm{th}} + C_{\ell}^{\mathrm{fg},i,j}``
* pixel variance maps ``\sigma_p^{XX}`` for ``XX \in \{II, QQ, UU\}`` for the four maps involved in the covariance

However, these are only sufficient for a description of a homogeneous survey with white noise and sufficient mask apodization. Two additional corrections are required for ``\sim1\%`` covariance determinations.
* noise power spectra ``\hat{N}_{\ell}^{XY,i,j}`` for the involved channels
* corrections to the diagonals of the covariance matrices from insufficient apodization around point sources

The basic calculation is essentially a mode-coupling calculation, and mode-coupling matrices are themselves used to correct the covariance matrix at the end. The methods in this package were written to match the analysis of the *Planck* satellite, and we provide a more detailed description of these methods in Li et al. 2020 (in prep). The derivation of these covariance matrices, in the limit of uniform noise, are available in Thibaut Louis's excellent [notes](https://pspy.readthedocs.io/en/latest/scientific_doc.pdf).

The expressions for the covariance matrix tend to reuse the same expressions many times, and one tends also to compute several different related covariance matrices (i.e. TTTT, TETE, TTTE) on the same maps. The covariance calculation in PowerSpectra.jl is centered around the [`CovarianceWorkspace`](@ref), which caches the various quantities that are re-used during covariance estimation.

## Computing the Covariance

First, let's set up the required data -- masks and variances. The variance is a [`Healpix.PolarizedMap`](https://ziotom78.github.io/Healpix.jl/dev/mapfunc/#Healpix.PolarizedMap) containing the fields `i`, `q`, `u`. In this example, we read the masks from disk, but set the variances for everything to 1.
```julia
using Healpix, PowerSpectra
mask1_T = readMapFromFITS("test/data/mask1_T.fits", 1, Float64)
mask2_T = readMapFromFITS("test/data/mask2_T.fits", 1, Float64)
mask1_P = readMapFromFITS("test/data/mask1_T.fits", 1, Float64)
mask2_P = readMapFromFITS("test/data/mask2_T.fits", 1, Float64)

# for this example, pixel variance = 1
nside = mask1_T.resolution.nside
unit_var = PolarizedMap{Float64, RingOrder}(nside)
unit_var.i .= 1.0
unit_var.q .= 1.0
unit_var.u .= 1.0
```
Once you have the masks and variances, you can create a [`CovField`](@ref) for each field involved. This
structure also has an associated name, which is used for the signal spectra.
```julia
# set up CovField, we're computing the variance of a spectrum on (f1 × f2)
f1 = CovField("143_hm1", mask1_T, mask1_P, unit_var)
f2 = CovField("143_hm2", mask2_T, mask2_P, unit_var)
f3 = f1
f4 = f2

# compute covariance between the (f1 × f2) spectrum and (f3 × f4) spectrum  
workspace = CovarianceWorkspace(f1, f2, f3, f4)
```
A covariance matrix calculation needs an assumed signal spectrum for each channel you want. 
You need to generate a dictionary that maps the names of various cross-spectra to [`SpectralVector`](@ref).
```julia
cl_th = SpectralVector(ones(nside2lmax(nside)+1))

spectra = Dict{SpectrumName, SpectralVector{Float64, Vector{Float64}}}(
    (:TT, "143_hm1", "143_hm1") => cl_th, (:TT, "143_hm1", "143_hm2") => cl_th,
    (:TT, "143_hm2", "143_hm1") => cl_th, (:TT, "143_hm2", "143_hm2") => cl_th,

    (:EE, "143_hm1", "143_hm1") => cl_th, (:EE, "143_hm1", "143_hm2") => cl_th,
    (:EE, "143_hm2", "143_hm1") => cl_th, (:EE, "143_hm2", "143_hm2") => cl_th ,

    (:TE, "143_hm1", "143_hm1") => cl_th, (:TE, "143_hm1", "143_hm2") => cl_th,
    (:TE, "143_hm2", "143_hm1") => cl_th, (:TE, "143_hm2", "143_hm2") => cl_th)
```

Now all that remains is to compute the coupled covmat.

```julia
C = coupledcov(:TT, :TT, workspace, spectra)
```

## API

```@docs
CovField
CovarianceWorkspace
coupledcov
``` 
