## load in the data
using PowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
using BenchmarkTools
ENV["OMP_NUM_THREADS"] = 16
hp = pyimport("healpy")
nmt = pyimport("pymaster")

data_dir = "/home/zequnl/.julia/dev/PowerSpectra/notebooks/data/"
# mask = readMapFromFITS(data_dir * "mask.fits", 1, Float64)
nside = 256
lmax = 3 * nside - 1

##
mask_arr = zeros(hp.nside2npix(nside))
θ, ϕ = hp.pix2ang(nside, 0:(hp.nside2npix(nside)-1))
ϕ[ϕ .> π] .-= 2π
mask_arr[
    (θ .* (1 .+ 0.1 .* sin.(ϕ .* 3.1 .+ 0.1)) .> 1.7) .|
    (θ .* (1 .+ 0.1 .* sin.(ϕ .* 2.0 .+ 0.1)) .< 1.25)
] .= 1.0
mask_arr = nmt.mask_apodization(mask_arr, 20.0, apotype="C2")

ps_mask = ones(hp.nside2npix(nside))
ps_mask[ rand(length(mask_arr)) .> 0.9999 ] .= 0.0
ps_mask = nmt.mask_apodization(ps_mask, 4.0, apotype="C2")

mask = Map{Float64, RingOrder}(mask_arr) 
saveToFITS(mask, "test/mask1_P.fits")
mask.pixels .= mask_arr .* ps_mask
saveToFITS(mask, "test/mask1_T.fits")

##
mask_arr = zeros(hp.nside2npix(nside))
θ, ϕ = hp.pix2ang(nside, 0:(hp.nside2npix(nside)-1))
ϕ[ϕ .> π] .-= 2π
mask_arr[
    (θ .* (1 .+ 0.1 .* sin.(ϕ .* (2.1+rand()) .+ 0.1 .+ (rand()*0.1) )) .> 1.7) .|
    (θ .* (1 .+ 0.1 .* sin.(ϕ .* (1 + rand()) .+ 0.1 .+ (rand()*0.1) )) .< 1.25)
] .= 1.0
mask_arr = nmt.mask_apodization(mask_arr, 20.0, apotype="C2")

ps_mask = ones(hp.nside2npix(nside))
ps_mask[ rand(length(mask_arr)) .> 0.9999 ] .= 0.0
ps_mask = nmt.mask_apodization(ps_mask, 4.0, apotype="C2")

mask = Map{Float64, RingOrder}(mask_arr) 
saveToFITS(mask, "test/mask2_P.fits")
mask.pixels .= mask_arr .* ps_mask
saveToFITS(mask, "test/mask2_T.fits")

clf(); hp.mollview(mask.pixels); gcf()

##

using CSV 
using DataFrames

data_dir = "/home/zequnl/.julia/dev/PowerSpectra/notebooks/data/"
theory = CSV.File(data_dir * "theory.csv") |> DataFrame  
m0 = hp.synfast(theory.cltt, nside=nside, verbose=false, pixwin=true, new=true)
saveToFITS(Map{Float64, RingOrder}(m0), "test/example_map.fits")




##