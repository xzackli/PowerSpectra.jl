```@meta
CurrentModule = AngularPowerSpectra
```

# Spectral Analysis

```julia
using Healpix
using AngularPowerSpectra

# make up a trivial mask of ones
nside = 256
mask = readMapFromFITS("test/data/example_mask_1.fits", 1, Float64)
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )

# pretend we are computing 143 GHz fields between two half-missions, hm1 and hm2
m1 = PolarizedField("143_hm1",  flat_mask, flat_mask)
m2 = PolarizedField("143_hm2", flat_mask, flat_mask)
workspace = SpectralWorkspace(m1, m2)

# compute the mode-coupling matrix
M = mcm(workspace, TT, "143_hm1", "143_hm2")
Cl_hat = spectra_from_masked_maps(map1 * mask, map1 * mask, lu(M.parent), flat_beam, flat_beam)
```
