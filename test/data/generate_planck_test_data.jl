## load in the data
ENV["OMP_NUM_THREADS"] = 16
using PowerSpectra
using Healpix
# using PyCall, PyPlot
# using CSV, DataFrames, LinearAlgebra
# using BenchmarkTools
# hp = pyimport("healpy")
using PyCall
using DelimitedFiles
nmt = pyimport("pymaster")

nside = 256
lmax = nside2lmax(nside)

maskT₁ = PowerSpectra.planck256_mask("100", "hm1", "T")
maskP₁ = PowerSpectra.planck256_mask("100", "hm1", "P")
maskT₂ = PowerSpectra.planck256_mask("100", "hm2", "T")
maskP₂ = PowerSpectra.planck256_mask("100", "hm2", "P")
flat_map = Map{Float64, RingOrder}(ones(nside2npix(nside)) )

##
f₁_0 = nmt.NmtField(maskT₁.pixels, [flat_map.pixels])
f₁_2 = nmt.NmtField(maskP₁.pixels, [flat_map.pixels, flat_map.pixels])
f₂_0 = nmt.NmtField(maskT₂.pixels, [flat_map.pixels])
f₂_2 = nmt.NmtField(maskP₂.pixels, [flat_map.pixels, flat_map.pixels])

b = nmt.NmtBin.from_nside_linear(nside, 1);

##
using JLD2
w = nmt.NmtWorkspace()
@time w.compute_coupling_matrix(f₁_0, f₂_0, b)
mcm00 = w.get_coupling_matrix()
@save "test/data/mcm00.jld2" {compress=true} mcm00

# writedlm("test/mcm_TT_diag.txt", diag(w.get_coupling_matrix()[1:lmax, 1:lmax])[3:767])

##
w = nmt.NmtWorkspace()
@time w.compute_coupling_matrix(f₁_2, f₂_2, b)
mcm22 = w.get_coupling_matrix()
@save "test/data/mcm22.jld2" {compress=true} mcm22
# writedlm("test/mcm_EE_diag.txt", diag(w.get_coupling_matrix()[1:4:4*lmax, 1:4:4*lmax])[3:767])

##
w = nmt.NmtWorkspace()
@time w.compute_coupling_matrix(f₁_0, f₂_2, b)
mcm02 = w.get_coupling_matrix()
@save "test/data/mcm02.jld2" {compress=true} mcm02

##
w = nmt.NmtWorkspace()
@time w.compute_coupling_matrix(f₁_2, f₂_0, b)
mcm20 = w.get_coupling_matrix()
@save "test/data/mcm20.jld2" {compress=true} mcm20


# writedlm("test/mcm_TE_diag.txt", diag(w.get_coupling_matrix()[1:2:2*lmax, 1:2:2*lmax])[3:767])

##
using BenchmarkTools
# @btime mcm(:TE, map2alm(maskT₁), map2alm(maskT₂))
@btime mcm((:EE_BB, :EB_BE), map2alm(maskT₁), map2alm(maskT₂))

##
@btime map2alm(maskT₁), map2alm(maskT₂)

##

# flat_beam = SpectralVector(ones(3*nside))
# flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )


# mask = readMapFromFITS("test/example_mask_1.fits", 1, Float64)
# clf()
# hp.mollview(mask.pixels, title="Simple Mask")
# gcf()
