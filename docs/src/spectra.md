```@meta
CurrentModule = PowerSpectra
```

# Spectral Analysis
In this section, we describe how one can estimate unbiased cross-spectra from masked maps using this package. We expand fluctuations on the sphere in terms of spherical harmonics, with coefficients 

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
using Healpix, PowerSpectra
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

Now one can apply a linear solve to decouple the mask. We apply the linear solve operator `Cl = M \ pCl` to perform mode decoupling on `SpectralArray` and `SpectralVector`.
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
Cl = M \ pCl
```

### Custom Multipole Ranges
The majority of the time, you want ``\ell_{\mathrm{min}}=0``, and you should subtract the monopole and dipole from your maps. Note that you can pass `lmin` to [`mcm`](@ref). Most other mode-coupling codes start the mode-coupling calculation at ``\ell_{\mathrm{min}} = 2``. In order to imitate this behavior, you must specify `lmin=2` and truncate the `SpectralVector` to remove the monopole and dipole.

```julia
using IdentityRanges  # range for preserving SpectralArrays index info in slices
pCl = SpectralVector(alm2cl(alm1, alm2))[IdentityRange(2:end)]  # start at dipole
M = mcm(:TT, map2alm(mask1), map2alm(mask2); lmin=2)            # start at dipole
Cl = M \ pCl  # SpectralArray with indices 2:end
```

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
You can compute these matrices by passing `:EE_BB` and `:EB_BE` as the first argument to [`mcm`](@ref). 
For these coupled channels, the [`@spectra`](@ref) macro can be helpful for writing clear and concise code. It unpacks the blocks of the resulting block-vector[^2] after mode decoupling. The matrix syntax in Julia performs concatenation when the inputs are arrays, so `[pCl_EE; pCl_BB]` stacks the coupled spectra vectors vertically.

```julia
# compute stacked EE,BB mode-coupling matrix from mask alm
M_EE_BB = mcm(:EE_BB, alm1, alm2)

# make up some coupled pseudo-spectra
pCl_EE, pCl_BB = pCl, pCl

# apply the 2×2 block mode-coupling matrix to the stacked EE and BB spectra
@spectra Cl_EE, Cl_BB = M_EE_BB \ [pCl_EE; pCl_BB]
```
In this case, `M_EE_BB` is a big matrix with blocks corresponding to ``\mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}}`` and ``\mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}}``. `mcm` wraps that matrix in a special `Array` type that keeps tracks of indices and blocks, which is used to unpack the results.

You can produce both matrices at once by passing a Tuple, `(:EE_BB, :EB_BE)` and get back a tuple containing the two matrices, which can be efficient since the these two block matrices share the same blocks. 

```julia
M_EE_BB, M_EB_BE = mcm((:EE_BB, :EB_BE), alm1, alm2)
```

You can also obtain the sub-blocks ``\mathbf{M}^{\nu_{1}\nu_{2}++}_{\ell_1 \ell_{2}}`` and ``\mathbf{M}^{\nu_{1}\nu_{2}--}_{\ell_1 \ell_{2}}`` by passing to [`mcm`](@ref) the symbols `:M⁺⁺` and `:M⁻⁻` (note the Unicode superscripts). 



[^2]: The `@spectra` macro used there is equivalent to
    ```julia
    Cl = M_EE_BB \ [pCl_EE; pCl_BB]
    Cl_EE, Cl_BB = getblock(Cl, 1), getblock(Cl, 2)
    ```

## API

```@docs
mcm
```
