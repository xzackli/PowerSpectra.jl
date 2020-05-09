## load in the data
using AngularPowerSpectra
using Healpix
using PyCall
using CSV, DataFrames
hp = pyimport("healpy")

##

data_dir = "/home/zequnl/.julia/dev/AngularPowerSpectra/notebooks/data/"
# mask = readMapFromFITS(data_dir * "mask.fits", 1, Float64)
nside = 256

mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )  # FLAT MASK
θ = [pix2ang(mask, i)[1] for i in 1:nside2npix(nside)]
ϕ = [pix2ang(mask, i)[2] for i in 1:nside2npix(nside)]
mask.pixels .*= sin.(5θ).^2


flatspec = (1:(3*nside)).^(-2)
flatspec[1:2] .= 0.0
m0 = hp.synfast(
    flatspec, 
    nside=nside, verbose=false, pixwin=false, new=true)

m = Map{Float64, RingOrder}(nside)
m.pixels .= m0;

##

nw = pyimport("nawrapper")
nm1 = nw.namap_hp(
    maps=(m.pixels), masks=mask.pixels, verbose=false, unpixwin=false)

##
using PyPlot

##

m_143_hm1 = Field("143_hm1", mask, mask)
m_143_hm2 = Field("143_hm2", mask, mask)
workspace = SpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)


# M = AngularPowerSpectra.compute_mcm!(workspace, "143_hm1", "143_hm2")


theory = CSV.read(data_dir * "theory.csv")


# map = readMapFromFITS(data_dir * "map.fits", 1, Float64)
M = AngularPowerSpectra.compute_mcm!(workspace, "143_hm1", "143_hm1")
# M .*= sqrt( sum(mask.pixels) / nside2npix(nside) )

using LinearAlgebra

m.pixels .*= mask.pixels
Cl_hat = (alm2cl(map2alm(m; niter=3)))
Cl_hat = inv(Symmetric(M.parent)) * Cl_hat

# println(sum(Cl_hat[100:300]) / 200)
##
clf()
axhline(1)
plot(Cl_hat .* (1:length(Cl_hat)).^2 , alpha=0.5)
# yscale("log")
gcf()

##
clf()
hp.mollview(m.pixels)
gcf()


##
using LinearAlgebra
using PyPlot
plt.clf()
plt.plot(diag(M.parent))
plt.gcf()
##

AngularPowerSpectra.effective_weights!(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
AngularPowerSpectra.W_spectra!(workspace)


using BenchmarkTools
@btime cov($workspace, $m_143_hm1, $m_143_hm2, $m_143_hm1, $m_143_hm2)
##