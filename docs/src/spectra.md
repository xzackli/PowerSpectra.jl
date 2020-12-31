```@meta
CurrentModule = AngularPowerSpectra
```

# Spectral Analysis

```julia
using Healpix
using AngularPowerSpectra

# load up a mask
nside = 256
mask = readMapFromFITS("test/data/example_mask_1.fits", 1, Float64)

# pretend we are computing 143 GHz fields between two half-missions, hm1 and hm2
m1 = PolarizedField("143_hm1", mask, mask)
m2 = PolarizedField("143_hm2", mask, mask)
workspace = SpectralWorkspace(m1, m2)

# compute the mode-coupling matrix
M = mcm(workspace, "TT", "143_hm1", "143_hm2")
```
