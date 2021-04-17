


```@example quickstart
using Healpix
using PowerSpectra

nside = 256
m₁ = 1e6 * PowerSpectra.planck256_map("100", "hm1", "I_STOKES")
m₂ = 1e6 * PowerSpectra.planck256_map("100", "hm2", "I_STOKES")

using Plots
using Plots.PlotMeasures: mm
plot(m₁, clim=(-200,200), rightmargin=5mm)
```


```@example quickstart
mask₁ = PowerSpectra.planck256_maskT("100", "hm1")
mask₂ = PowerSpectra.planck256_maskT("100", "hm2")
plot(mask₁, rightmargin=5mm)
```


```@example quickstart
M_TT = mcm(:TT, map2alm(mask₁), map2alm(mask₂))
pCl = SpectralVector(alm2cl(map2alm(m₁ * mask₁), map2alm(m₂ * mask₂)))
Cl = M_TT \ pCl
```


```@example quickstart
lmax = nside2lmax(256)
bl = PowerSpectra.planck_beam_bl("100", "hm1", "100", "hm2", :TT, :TT; lmax=lmax)
ℓ = eachindex(Cl)
plot( (ℓ.^2 / (2π)) .*  Cl ./ bl.^2, label="\$D_{\\ell}\$", xlim=(0,2nside) )

theory = PowerSpectra.planck_theory_Dl()
plot!(theory[:TT], label="theory")
```