# get some example masks

using Healpix, AngularPowerSpectra
using LinearAlgebra
# mask1 = readMapFromFITS("data/mask1_T.fits", 1, Float64)
# mask2 = readMapFromFITS("data/mask2_T.fits", 1, Float64)

nside = 256
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )

# compute TT mode-coupling matrix from mask harmonic coefficients
a1 = map2alm(flat_mask)
a2 = map2alm(flat_mask)
M = mcm(:TT, a1, a2; lmin=2)

##
pCl = SpectralVector(alm2cl(a1, a2))

##
lu(M) \ pCl

##
M \ pCl

##
