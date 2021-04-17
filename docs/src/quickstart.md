```@meta
CurrentModule = PowerSpectra
```

# Quickstart

Let's compute some power spectra! This package includes real Planck 2018 frequency maps,
scaled down to ``n_{\mathrm{side}} = 256``. These are not exported by default, so you have
to call them with `PowerSpectra.planck...` names. For example, let's show the 100 GHz 
half-mission maps. We plot the first half-mission.

```@example quickstart
using Healpix
using PowerSpectra

nside = 256
m₁ = 1e6 * PowerSpectra.planck256_map("100", "hm1", "I_STOKES")
m₂ = 1e6 * PowerSpectra.planck256_map("100", "hm2", "I_STOKES")

using Plots
plot(m₁, clim=(-200,200))
```

Looks good! The point of this package is that we don't want things like the galaxy and 
point sources getting in the way. Let's load in the scaled-down likelihood masks used in the 
Planck 2018 cosmological analysis. Let's get those masks too.

```@example quickstart
mask₁ = PowerSpectra.planck256_maskT("100", "hm1")
mask₂ = PowerSpectra.planck256_maskT("100", "hm2")
plot(mask₁)
```

This mask will be multiplied with our maps to form pseudo-spectra. That will bias our 
estimate of the spectrum, so we'll have to compute a mode-coupling matrix. Let's do that!

* The mode-coupling matrix involves the spherical harmonics of the masks. 
* The pseudo-spectrum depends on the spherical harmonics of the masked maps. 
* We wrap the spectrum from `Healpix.alm2cl` in a `SpectralVector` in order to tell the 
    package that this vector represents a power spectrum. Also, it makes it 0-indexed.
* The important part: we undo the effect of the mask by performing a linear solve.

```@example quickstart
M_TT = mcm(:TT, map2alm(mask₁), map2alm(mask₂))
pCl = SpectralVector(alm2cl(map2alm(m₁ * mask₁), map2alm(m₂ * mask₂)))
Cl = M_TT \ pCl
```

Now let's plot our spectrum and compare it to the official Planck 2018 bestfit theory.
Note that the instrumental beam is also in these maps. This package also provides the 
Planck beams as a utility function. We plot these in the convenient scaling,
``D_{\ell} \equiv \ell(\ell+1) C_{\ell} / 2\pi``.

```@example quickstart
lmax = nside2lmax(256)
bl = PowerSpectra.planck_beam_bl("100", "hm1", "100", "hm2", :TT, :TT; lmax=lmax)
ell = eachindex(Cl)
prefactor = ell .* (ell .+ 1) ./ (2π)
plot( prefactor .*  Cl ./ bl.^2, label="\$D_{\\ell}\$", xlim=(0,2nside) )

theory = PowerSpectra.planck_theory_Dl()  # returns a Dict of Dl indexed with :TT, :EE, ...
plot!(theory[:TT], label="theory")
```

Looks pretty good!
