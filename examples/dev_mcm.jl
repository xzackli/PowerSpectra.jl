# get some example masks
using Healpix, AngularPowerSpectra
mask1 = readMapFromFITS("data/mask1_T.fits", 1, Float64)
mask2 = readMapFromFITS("data/mask2_T.fits", 1, Float64)

##

# compute TT mode-coupling matrix from mask harmonic coefficients
a1 = map2alm(mask1)
a2 = map2alm(mask2)
M = mcm(:TE, a1, a2)

##



##
