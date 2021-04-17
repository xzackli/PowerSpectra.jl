```@meta
CurrentModule = PowerSpectra
```

# Quickstart

Let's compute some power spectra! For testing convenience, this package includes the 
Planck 2018 frequency maps, scaled down to ``n_{\mathrm{side}} = 256``. These are not 
exported by default, so you have to call them with `PowerSpectra.planck...` names. In this
example, we will use the 100 GHz half-mission temperature maps. 

We plot the first half-mission.

```@example quickstart
using Healpix
using PowerSpectra
using Plots

# retrieve the downgraded planck maps
nside = 256
map1 = 1e6 * PowerSpectra.planck256_map("100", "hm1", "I_STOKES")
map2 = 1e6 * PowerSpectra.planck256_map("100", "hm2", "I_STOKES")
plot(map1, clim=(-200,200))
```

We don't want things like the Galaxy and point sources getting in the way. Let's load in the 
scaled-down likelihood masks used in the Planck 2018 cosmological analysis. 

```@example quickstart
mask1 = PowerSpectra.planck256_mask("100", "hm1", :T)
mask2 = PowerSpectra.planck256_mask("100", "hm2", :T)
plot(mask1)
```

This mask will be multiplied with our maps to form pseudo-spectra. This removes the Galaxy
and the point sources. However, it will bias our 
estimate of the spectrum, so we'll have to compute a mode-coupling matrix. Let's do that!

* The mode-coupling matrix involves the spherical harmonics of the masks. 
* The pseudo-spectrum depends on the spherical harmonics of the masked maps. 
* We wrap the spectrum from `Healpix.alm2cl` in a `SpectralVector` in order to tell the 
    package that this vector represents a power spectrum. Also, it makes it 0-indexed.
* The important part: we undo the effect of the mask by performing a linear solve.

```@example quickstart
# compute the cross-spectra of the masked maps
masked_alm_1 = map2alm(map1 * mask1)
masked_alm_2 = map2alm(map2 * mask2)
pCl = SpectralVector(alm2cl(masked_alm_1, masked_alm_2))

# compute the mode coupling matrix
M_TT = mcm(:TT, map2alm(mask1), map2alm(mask2))

# perform a linear solve to undo the effects of mode-coupling
Cl = M_TT \ pCl
```

Now let's plot our spectrum and compare it to the official Planck 2018 bestfit theory.
Note that the instrumental beam is also in these maps. This package also provides the 
Planck beams as a utility function. We plot these in the convenient scaling,
``D_{\ell} \equiv \ell(\ell+1) C_{\ell} / 2\pi``.

```@example quickstart
# get lmax from nyquist frequency
lmax = nside2lmax(256)

# get the planck instrumental beam
bl = PowerSpectra.planck_beam_bl("100", "hm1", "100", "hm2", :TT, :TT; lmax=lmax)

# plot our spectra
ell = eachindex(Cl)
prefactor = ell .* (ell .+ 1) ./ (2π)
plot( prefactor .*  Cl ./ bl.^2, label="\$D_{\\ell}\$", xlim=(0,2nside) )

# compare it to theory
theory = PowerSpectra.planck_theory_Dl()  # returns a Dict of Dl indexed with :TT, :EE, ...
plot!(theory[:TT], label="theory")
```

Looks pretty good!
