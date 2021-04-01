# get some example masks

using Healpix, AngularPowerSpectra
using LinearAlgebra
cd("test")

mask1 = readMapFromFITS("data/mask1_T.fits", 1, Float64)
mask2 = readMapFromFITS("data/mask2_T.fits", 1, Float64)

nside = 256
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )

# compute TT mode-coupling matrix from mask harmonic coefficients
a1 = map2alm(mask1)
a2 = map2alm(mask2)
M = mcm(:TT, a1, a2)
pCl = SpectralVector(alm2cl(a1, a2))

##
F = (lu(M))
F \ BlockSpectralMatrix(pCl)

##
M \ pCl

##
BlockSpectralMatrix(M)

##
heatmap(2:20, 2:20, ((inv(M)[2:20, 2:20])),
    aspectratio=1, size=(500,500), xticks=0:10, yticks=0:10, yflip=true)

##
heatmap(2:30, 2:30, (inv(M[2:30,2:30]) .- (inv(M)[2:30, 2:30])) ./ (inv(M)[2:30, 2:30]),
    aspectratio=1, size=(500,500), xticks=0:10, yticks=0:10, yflip=true, clim=(-0.01, 0.01))

##

##
s = [pCl; pCl]

##
lu(M)


##
lu(M) \ pCl

##
M \ pCl

##

##
M⁻⁻ = mcm(:M⁻⁻, a1, a1; lmin=2)
M⁺⁺ = mcm(:M⁺⁺, a1, a1; lmin=2)

M_EE_BB = [ M⁺⁺  M⁻⁻;
            M⁻⁻  M⁺⁺ ]

# plot([diag(parent(M⁻⁻)), diag(parent(M⁺⁺))]; ylim=(-1,2))
# heatmap(M_EE_BB, aspectratio=1)

##

using IdentityRanges
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m1 = CovField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m2 = CovField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    M = mcm(:M⁺⁺, m1.maskP, m2.maskP)
    # factorized_mcm12 = lu(parent(M))
    reference = readdlm("data/mcm_EE_diag.txt")
    @test all(reference .≈ diag(parent(M))[3:767])

# plot([reference_spectrum ./ parent(Cl_hat)], labels=["ref/Cl"], ylim=(0,2))
