using PyCall
using Healpix
hp = pyimport("healpy")


"""
    fill_single_alm!(𝐦::Map{T}, ℓ, m) where T

Fills a map with a single spherical harmonic.

# Arguments:
- `𝐦::Map{T}`: map to fill
- `ℓ`: quantum number
- `m`: quantum number
"""
function fill_single_alm!(𝐦::Map{T}, ℓ, m) where T
    for i in 1:nside2npix(𝐦.resolution.nside)
        θ, ϕ = pix2ang(𝐦, i)
        𝐦.pixels[i] = sphevaluate(θ, ϕ, ℓ, m)
    end
    if m != 0
        fact = (-1)^m #* √2
        𝐦.pixels .*= fact
    end
    return 𝐦
end

##
nside = 256
m = Map{Float64,RingOrder}(nside)
# m.pixels .= 1.0
fill_single_alm!(m, 1, 1)
w = readMapFromFITS("test/data/example_mask_1.fits", 1, Float64) 
m.pixels .= w.pixels

##
mb = Map{BigFloat,RingOrder}(nside)
mb.pixels .= w.pixels
# fill_single_alm!(mb, 1, 1)

##
using AngularPowerSpectra
using BenchmarkTools
using StaticArrays
using ReferenceImplementations
using LinearAlgebra


# @refimpl fitdipole fitdipole(mb, mb*0+1)

##
fitdipole(mb, mb*0+1)

##
fitdipole(m, m*0+1)

##
@refimpl fitdipole(m, m*0+1)

##
hp.pixelfunc.fit_dipole(m.pixels)

##
plot(m)

##
hp.pixelfunc.fit_dipole(m.pixels)
##

# a = map2alm(m)
# a.alm[almIndex(a, 1, 0)]
