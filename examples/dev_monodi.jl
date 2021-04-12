using PyCall
using Healpix
using FastTransforms
hp = pyimport("healpy")


##
nside = 256
m = Map{Float64,RingOrder}(nside)
# m.pixels .= 1.0
fillalm!(m, 1, 1)
w = readMapFromFITS("test/data/example_mask_1.fits", 1, Float64) 

##
mb = Map{BigFloat,RingOrder}(nside)
fillalm!(mb, 1, 1)

##
using BenchmarkTools
using StaticArrays
using ReferenceImplementations
using LinearAlgebra


@refimpl fitdipole(mb, mb*0+1)

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

a = map2alm(m)
a.alm[almIndex(a, 1, 0)]

##


#

