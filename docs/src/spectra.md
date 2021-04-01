```@meta
CurrentModule = AngularPowerSpectra
```

# Spectral Analysis
In this section, we describe how one can estimate unbiased cross-spectra from masked maps using this package. We expand fluctuations on the sphere in terms of spherial harmonics, with coefficients 

```math
a_{\ell m} = \iint \Theta(\mathbf{\hat{n}}) Y_{\ell m}^* (\mathbf{\hat{n}}) \, d\Omega.
```

We then define the power spectrum ``C_{\ell}`` of these fluctuations,

```math
\langle a_{\ell m}^X a_{\ell^\prime m^\prime}^{Y*} \rangle = \delta_{\ell \ell^\prime} \delta_{m m^\prime} C_{\ell}.
```

## Mode Coupling for TT, TE, TB

If you compute the cross-spectrum of masked maps, the mask will couple together different modes ``\ell_1, \ell_2``. This biased estimate of the true spectrum is termed the pseudo-spectrum ``\widetilde{C}_{\ell}``, 
```math
\widetilde{C}_{\ell} = \frac{1}{2\ell+1} \sum_m \mathsf{m}^{i,X}_{\ell m} \mathsf{m}^{j,Y}_{\ell m},
```
where ``\mathsf{m}^{i,X}_{\ell m}`` are the spherical harmonic coefficients of the masked map ``i`` of channel ``X \in \{T, E, B\}``. In the pseudo-``C_{\ell}`` method, we seek an estimate ``\hat{C}_{\ell}`` of the true spectrum that is related to the pseudo-spectrum by a linear operator,
```math
   \langle\widetilde{C}_{\ell}\rangle = \mathbf{M}^{XY}(i,j)_{\ell_1 \ell_2} \langle C_{\ell} \rangle,
```
where ``\mathbf{M}^{XY}(i,j)_{\ell_1 \ell_2}`` is the mode-coupling matrix between fields ``i`` and ``j`` for spectrum ``XY``. The expectation value ``\langle \cdots \rangle`` in this expression is over all realizations of ``a_{\ell m}``, since the mask is not isotropic. Applying the inverse of the mode-coupling matrix to the pseudo-spectrum ``\widetilde{C}_{\ell}`` yields an unbiased and nearly optimal estimate ``\hat{C}_{\ell}`` of the true spectrum. To compute the mode-coupling matrix, one needs

* ``XY``, the desired spectrum, i.e. ``TE``
* ``\mathsf{m}^{i,X}_{\ell m}``, spherical harmonic coefficients of the mask for map ``i``, mode ``X``
* ``\mathsf{m}^{j,Y}_{\ell m}``, spherical harmonic coefficients of the mask for map ``j``, mode ``Y``

A basic functionality of this package is to compute this matrix. Let's look at a basic example of the cross-spectrum between two intensity maps.

```julia
# get some example masks
using Healpix, AngularPowerSpectra
mask1 = readMapFromFITS("test/data/mask1_T.fits", 1, Float64)
mask2 = readMapFromFITS("test/data/mask2_T.fits", 1, Float64)

# compute TT mode-coupling matrix from mask harmonic coefficients
M = mcm(:TT, map2alm(mask1), map2alm(mask2))
```

Similarly, one could have specified the symbol `:TE`, `:TE`, or `:ET` for other types of cross-spectra[^1].
The function `mcm` returns a `SpectralArray{T,2}`, which is an array type that contains elements in ``\ell_{\mathrm{min}} \leq \ell_1, \ell_2 \leq \ell_{\mathrm{max}}``. The important thing about `SpectralArray` is that indices correspond to ``\ell``, such that `M[ℓ₁, ℓ₂]` corresponds to the mode-coupling matrix entry ``\mathbf{M}_{\ell_1, \ell_2}``. If you want to access the underlying array, you can use `parent(mcm).`. One can optionally truncate the computation with the `lmax` keyword, i.e. `mcm(:TT, mask1, mask2; lmin=2, lmax=10)`. 

[^1]: You can combine symbols, in cases where you're looping over combinations of spectra, by using `Symbol`.
    ```julia-repl
    julia> Symbol(:T, :T)
    :TT
    ```

Now one can apply a linear solve to decouple the mask. We define a special operator `Cl = M \ₘ pCl` to perform mode decoupling on `SpectralArray` and `SpectralVector`. This operation performs a linear solve on ``\ell_{\mathrm{min}} \leq \ell \leq \ell_{\mathrm{max}}``, the minimum and maximum indices of the mode-coupling matrix. It leaves the elements outside of those ``\ell``-ranges untouched in `pCl`.

Here's an example that uses the mode-coupling matrix from above to obtain spectra from masked maps.

```julia
# generate two uniform maps
nside = mask1.resolution.nside
npix = nside2npix(nside)
map1 = Map{Float64, RingOrder}(ones(npix))
map2 = Map{Float64, RingOrder}(ones(npix))

# mask the maps with different masks
map1.pixels .*= mask1.pixels
map2.pixels .*= mask2.pixels

# compute the pseudo-spectrum, and wrap it in a SpectralVector
alm1, alm2 = map2alm(map1), map2alm(map2)
pCl = SpectralVector(alm2cl(alm1, alm2))

# decouple the spectrum
Cl = M \ₘ pCl
```
**Note that this performs the linear solve starting at** ``\mathbf{\ell = 2}``, setting ``\hat{C}_{\ell < 2} = 0``. The linear solve operator on `SpectralArrays` is specialized to do this. Most of the time you want to avoid the monopole and dipole, which tend to be very large relative to anisotropies. **You should subtract the monopole and dipole from your maps.** If you really want to perform the linear solve on the monopole and dipole, you can use the underlying arrays to decouple the spectrum, i.e. `Cl_starting_from_zero = parent(M) \ parent(pCl)`. 

## Mode Coupling for EE, EB, BB

The mode coupling on spin-2 ``\times`` spin-2 (`:EE`, `:EB`, `:BB`) is slightly more complicated. For a more detailed description, please see Thibaut Louis's [notes](https://pspy.readthedocs.io/en/latest/scientific_doc.pdf).

```math
\tiny
 \begin{bmatrix} 
 \langle \widetilde{C}^{T_{\nu_{1}}T_{\nu_{2}}}_{\ell_1} \rangle \cr
  \langle \widetilde{C}^{T_{\nu_{1}}E_{\nu_{2}}}_{\ell_1} \rangle \cr 
  \langle \widetilde{C}^{T_{\nu_{1}}B_{\nu_{2}}}_{\ell_1} \rangle  \cr 
  \langle \widetilde{C}^{E_{\nu_{1}}T_{\nu_{2}}}_{\ell_1} \rangle  \cr 
  \langle \widetilde{C}^{B_{\nu_{1}}T_{\nu_{2}}}_{\ell_1} \rangle  \cr 
  \langle \widetilde{C}^{E_{\nu_{1}}E_{\nu_{2}}}_{\ell_1} \rangle  \cr 
  \langle \widetilde{C}^{B_{\nu_{1}}B_{\nu_{2}}}_{\ell_1} \rangle \cr
  \langle \widetilde{C}^{E_{\nu_{1}}B_{\nu_{2}}}_{\ell_1} \rangle \cr  
  \langle \widetilde{C}^{B_{\nu_{1}}E_{\nu_{2}}}_{\ell_1} \rangle 
  \end{bmatrix} = \sum_{\ell_{2}}
\begin{bmatrix} 
\mathbf{M}^{\nu_{1}\nu_{2}00}_{\ell_1 \ell_{2}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 &
\cr
0 & \mathbf{M}^{\nu_{1}\nu_{2}02}_{\ell_1 \ell_{2}} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 
\cr
0 & 0 & \mathbf{M}^{\nu_{1}\nu_{2}02}_{\ell_1 \ell_{2}} & 0 & 0 & 0 & 0 & 0 & 0 &
\cr
0 & 0 & 0 & \mathbf{M}^{\nu_{1}\nu_{2}02}_{\ell_1 \ell_{2}} & 0 & 0 & 0 & 0 & 0 &
\cr
0 & 0 &  0 & 0 & \mathbf{M}^{\nu_{1}\nu_{2}02}_{\ell_1 \ell_{2}} & 0 & 0 & 0 & 0 &
\cr
0 & 0 & 0 & 0 & 0 & \mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}} & \mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}} & 0 & 0 &
\cr
0 & 0 & 0 & 0 & 0 & \mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}} & \mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}} & 0 & 0 &
\cr
0 &0 &0 &0 & 0 & 0 & 0 & \mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}}  & -\mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}}  &
\cr
0 & 0 & 0 & 0 & 0 & 0 & 0 & -\mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}} & \mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}} &
\end{bmatrix}
\begin{bmatrix} \langle C^{T_{\nu_{1}}T_{\nu_{2}}}_{\ell_{2}} \rangle  \cr \langle C^{T_{\nu_{1}}E_{\nu_{2}}}_{\ell_{2}} \rangle  \cr \langle C^{T_{\nu_{1}}B_{\nu_{2}}}_{\ell_{2}} \rangle  \cr \langle C^{E_{\nu_{1}}T_{\nu_{2}}}_{\ell_{2}} \rangle  \cr \langle C^{B_{\nu_{1}}T_{\nu_{2}}}_{\ell_{2}} \rangle  \cr 
\langle C^{E_{\nu_{1}}E_{\nu_{2}}}_{\ell_{2}} \rangle  \cr 
\langle C^{B_{\nu_{1}}B_{\nu_{2}}}_{\ell_{2}} \rangle \cr
\langle C^{E_{\nu_{1}}B_{\nu_{2}}}_{\ell_{2}} \rangle \cr  
\langle C^{B_{\nu_{1}}E_{\nu_{2}}}_{\ell_{2}} \rangle  
\end{bmatrix}
```

Note that the ``(0,0)``, ``(0,2)``, and ``(2,0)`` combinations from the previous section are block-diagonal. Thus we define 

```math
\begin{aligned}
    \mathbf{M}^{\nu_1 \nu_2 TT}_{\ell_1 \ell_2} &= \mathbf{M}^{\nu_1 \nu_2 00}_{\ell_1 \ell_2} \\
    \mathbf{M}^{\nu_1 \nu_2 TE}_{\ell_1 \ell_2} = \mathbf{M}^{\nu_1 \nu_2 TB}_{\ell_1 \ell_2} &= \mathbf{M}^{\nu_1 \nu_2 02}_{\ell_1 \ell_2} \\
    \mathbf{M}^{\nu_1 \nu_2 ET}_{\ell_1 \ell_2} = \mathbf{M}^{\nu_1 \nu_2 BT}_{\ell_1 \ell_2} &= \mathbf{M}^{\nu_1 \nu_2 20}_{\ell_1 \ell_2}
\end{aligned}
```
The previous section showed how to compute these matrices, by passing `:TT`, `:TE`, `:TB`, `:ET`, or `:BT` to [`mcm`](@ref). We now define two additional block matrices,

```math
\mathbf{M}^{\nu_1 \nu_2 EE,BB}_{\ell_1 \ell_2} = \left[
\begin{array}{cc}
\mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}} & \mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}} \\
 \mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}} & \mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}} \\
\end{array} \right], \qquad
\mathbf{M}^{\nu_1 \nu_2 EB,BE}_{\ell_1 \ell_2} = \left[
\begin{array}{cc}
\mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}}  & -\mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}} \\
-\mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}} & \mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}}  \\
\end{array}
\right].
```

These matrices are defined such that

```math
\left[
\begin{array}{c}
\langle \widetilde{C}^{E_{\nu_{1}}E_{\nu_{2}}}_{\ell_1} \rangle  \\
\langle \widetilde{C}^{B_{\nu_{1}}B_{\nu_{2}}}_{\ell_1} \rangle \\
\end{array}
\right] = \sum_{\ell_2} \mathbf{M}^{\nu_1 \nu_2 EE,BB}_{\ell_1 \ell_2} \left[
\begin{array}{c}
\langle C^{E_{\nu_{1}}E_{\nu_{2}}}_{\ell_{2}} \rangle  \\
\langle C^{B_{\nu_{1}}B_{\nu_{2}}}_{\ell_{2}} \rangle \\
\end{array}
\right],
```
```math
\left[
\begin{array}{c}
\langle \widetilde{C}^{E_{\nu_{1}}B_{\nu_{2}}}_{\ell_1} \rangle  \\
\langle \widetilde{C}^{B_{\nu_{1}}E_{\nu_{2}}}_{\ell_1} \rangle \\
\end{array}
\right] = \sum_{\ell_2} \mathbf{M}^{\nu_1 \nu_2 EB,BE}_{\ell_1 \ell_2} \left[
\begin{array}{c}
\langle C^{E_{\nu_{1}}B_{\nu_{2}}}_{\ell_{2}} \rangle  \\
\langle C^{B_{\nu_{1}}E_{\nu_{2}}}_{\ell_{2}} \rangle \\
\end{array}
\right].
```
You can compute these matrices by passing `:EE_BB` and `:EB_BE` as the first argument to [`mcm`](@ref). You can produce both matrices at once by passing a Tuple, `(:EE_BB, :EB_BE)` and get back a tuple containing the two matrices, which can be efficient since the these two block matrices share the same blocks. You can also obtain the sub-blocks ``\mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}}`` and ``\mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}}`` by passing to [`mcm`](@ref) the symbols `:M⁺⁺` and `:M⁻⁻` (note the Unicode superscripts). 



## API

```@docs
mcm
```
