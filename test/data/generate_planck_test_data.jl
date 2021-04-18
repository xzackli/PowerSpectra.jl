## load in the data
ENV["OMP_NUM_THREADS"] = 16
using PowerSpectra
using Healpix
using PyCall
using DelimitedFiles
nmt = pyimport("pymaster")

nside = 256
lmax = nside2lmax(nside)

m₁ = PowerSpectra.planck256_polmap("100", "hm1")
m₂ = PowerSpectra.planck256_polmap("100", "hm2")
maskT₁ = PowerSpectra.planck256_mask("100", "hm1", :T)
maskP₁ = PowerSpectra.planck256_mask("100", "hm1", :P)
maskT₂ = PowerSpectra.planck256_mask("100", "hm2", :T)
maskP₂ = PowerSpectra.planck256_mask("100", "hm2", :P)
flat_map = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
scale!(m₁, 1e6)
scale!(m₂, 1e6)

##
f₁_0 = nmt.NmtField(maskT₁.pixels, [m₁.i.pixels])
f₁_2 = nmt.NmtField(maskP₁.pixels, [m₁.q.pixels, m₁.u.pixels])
f₂_0 = nmt.NmtField(maskT₂.pixels, [m₂.i.pixels])
f₂_2 = nmt.NmtField(maskP₂.pixels, [m₂.q.pixels, m₂.u.pixels])

b = nmt.NmtBin.from_nside_linear(nside, 1);

##
# Compute MASTER estimator
# spin-0 x spin-0
cl_00 = nmt.compute_full_master(f₁_0, f₂_0, b)
# spin-0 x spin-2
cl_02 = nmt.compute_full_master(f₁_0, f₂_2, b)
# spin-0 x spin-2
cl_20 = nmt.compute_full_master(f₁_2, f₂_0, b)
# spin-2 x spin-2
cl_22 = nmt.compute_full_master(f₁_2, f₂_2, b)

##
using JLD2
@save "test/data/planck_spec.jld2" {compress=true} cl_00 cl_02 cl_20 cl_22


##
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
