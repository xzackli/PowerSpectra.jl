using AngularPowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
using BenchmarkTools
hp = pyimport("healpy")
nmt = pyimport("pymaster")

##
fake_σ = readMapFromFITS("/media/data/wmap/ring/wmap_band_imap_r9_9yr_W_v5.fits", 1, Float64)

clf()
hp.mollview(fake_σ.pixels, min=-0.5, max=0.5)
gcf()


##
wmap_W

