using Healpix
using PowerSpectra
using Plots

# retrieve the downgraded planck maps
nside = 256
map1 = 1e6 * PowerSpectra.planck256_map("100", "hm1", "I_STOKES")
map2 = 1e6 * PowerSpectra.planck256_map("100", "hm2", "I_STOKES")
plot(map1, clim=(-200,200))

mask1 = PowerSpectra.planck256_mask("100", "hm1", :T)
mask2 = PowerSpectra.planck256_mask("100", "hm2", :T)
plot(mask1)

# compute the cross-spectra of the masked maps
masked_alm_1 = map2alm(map1 * mask1)
masked_alm_2 = map2alm(map2 * mask2)
pCl = SpectralVector(alm2cl(masked_alm_1, masked_alm_2))

# compute the mode coupling matrix
M_TT = mcm(:TT, map2alm(mask1), map2alm(mask2))

# perform a linear solve to undo the effects of mode-coupling
Cl = M_TT \ pCl

# get lmax from nyquist frequency
lmax = nside2lmax(nside)

# get the planck instrumental beam and pixel window
Wl = PowerSpectra.planck_beam_Wl("100", "hm1", "100", "hm2", :TT, :TT; lmax=lmax)
pixwinT = SpectralVector(pixwin(nside)[1:(lmax+1)])

# plot our spectra
ell = eachindex(Cl)
prefactor = ell .* (ell .+ 1) ./ (2π)
plot(prefactor .*  Cl ./ (Wl .* pixwinT.^2), label="\$D_{\\ell}\$", xlim=(0,2nside) )

# compare it to theory
theory = PowerSpectra.planck_theory_Dl()  # returns a Dict of Dl indexed with :TT, :EE, ...
plot!(theory[:TT], label="theory TT")


##
using Healpix
using PowerSpectra
using Plots

nside = 256
m₁ = PowerSpectra.planck256_polmap("100", "hm1")
m₂ = PowerSpectra.planck256_polmap("100", "hm2")
maskT₁ = PowerSpectra.planck256_mask("100", "hm1", :T)
maskP₁ = PowerSpectra.planck256_mask("100", "hm1", :P)
maskT₂ = PowerSpectra.planck256_mask("100", "hm2", :T)
maskP₂ = PowerSpectra.planck256_mask("100", "hm2", :P)

# convert to μK
scale!(m₁, 1e6)
scale!(m₂, 1e6)

lmax = nside2lmax(256)
# utility function for doing TEB mode decoupling
Cl = master(m₁, maskT₁, maskP₁, 
            m₂, maskT₂, maskP₂)
lmax = nside2lmax(256)
print(keys(Cl))
spec = :TE
Wl = PowerSpectra.planck_beam_Wl("100", "hm1", "100", "hm2", spec, spec; lmax=lmax)
pixwinT, pixwinP = pixwin(nside; pol=true)
pixwinT = SpectralVector(pixwinT[1:(lmax+1)])
pixwinP = SpectralVector(pixwinP[1:(lmax+1)])

ℓ = eachindex(Wl)
plot( (ℓ.^2 / (2π)) .*  Cl[spec] ./ (Wl .* pixwinT .* pixwinP), 
    label="\$D_{\\ell}\$", xlim=(0,2nside) )
theory = PowerSpectra.planck_theory_Dl()
plot!(theory[spec], label="theory $(spec)")
##

spec = :EE
Wl = PowerSpectra.planck_beam_Wl("100", "hm1", "100", "hm2", spec, spec; lmax=lmax)
ℓ = eachindex(Wl)
plot( (ℓ.^2 / (2π)) .*  Cl[spec] ./ (Wl .* pixwinP.^2), 
    label="\$D_{\\ell}\$", xlim=(0,2nside) )
theory = PowerSpectra.planck_theory_Dl()
plot!(theory[spec], label="theory $(spec)", ylim=(-10,50))
##
