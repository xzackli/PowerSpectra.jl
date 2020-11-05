## load in the data
using AngularPowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
using BenchmarkTools
hp = pyimport("healpy")
nmt = pyimport("pymaster")

data_dir = "/home/zequnl/.julia/dev/AngularPowerSpectra/notebooks/data/"
# mask = readMapFromFITS(data_dir * "mask.fits", 1, Float64)
nside = 256

mask_arr = zeros(hp.nside2npix(nside))
θ, ϕ = hp.pix2ang(nside, 0:(hp.nside2npix(nside)-1))
ϕ[ϕ .> π] .-= 2π
mask_arr[
    (θ .* (1 .+ 0.1 .* sin.(ϕ .* 3.1 .+ 0.1)) .> 1.7) .|
    (θ .* (1 .+ 0.1 .* sin.(ϕ .* 2.0 .+ 0.1)) .< 1.25)
] .= 1.0
mask_arr = nmt.mask_apodization(mask_arr, 20.0, apotype="C2")

mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) ) 
mask.pixels .= mask_arr

clf()
hp.mollview(mask.pixels, title="Simple Mask")
gcf()

##
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
theory = CSV.read(data_dir * "theory.csv"; limit=2000)

clf()
plot(theory.cltt .* theory.ell.^2, "-")
plot(1e-2 .* theory.ell.^2, "-")
gcf()

##
m0 = hp.synfast(theory.cltt, nside=nside, verbose=false, pixwin=true, new=true)
m = Map{Float64, RingOrder}(nside)
m.pixels .= m0;
# map = readMapFromFITS(data_dir * "map.fits", 1, Float64)

wl = SpectralVector(hp.pixwin(nside))
m_143_hm1 = Field("143_hm1", mask, flat_mask, wl)
m_143_hm2 = Field("143_hm2", mask, flat_mask, wl)
workspace = SpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)

@time mcm = compute_mcm(workspace, "143_hm1", "143_hm1")
@time factorized_mcm = cholesky(Hermitian(mcm.parent));


##
import AngularPowerSpectra: TT

cltt = SpectralVector(convert(Vector, theory.cltt))
spectra = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
    (TT, "143_hm1", "143_hm1") => cltt,
    (TT, "143_hm2", "143_hm2") => cltt,
    (TT, "143_hm1", "143_hm2") => cltt,
    (TT, "143_hm2", "143_hm1") => cltt)
@time C = compute_covmat(workspace, spectra, factorized_mcm, factorized_mcm,
                         m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);


##
@time Clhat = compute_spectra(m * mask, m * mask, factorized_mcm, wl, wl)
ells = collect(0:(length(Clhat)-1));

##
using JLD2
function generate_sim_array(spec, nsims, nside, mask_, fact_mcm)

    result = Array{Float64, 2}(undef, (3 * nside, nsims))
    for i in 1:nsims
        m0 = hp.synfast(spec, nside=nside, verbose=false, pixwin=false, new=true)
        norm = mean(m0)
        m0 .-= norm
        m = Map{Float64, RingOrder}(nside)
        m.pixels .= m0
        result[:,i] .= compute_spectra(m * mask_, m * mask_, factorized_mcm, wl, wl)
    end

    return result
end
# sims = generate_sim_array(theory.cltt, 100, nside, mask, factorized_mcm)
# @save "sims.jld2" sims
@load "sims.jld2" sims

##

clf()
plot(theory.cltt[1:(3*nside)] ./ Clhat, "-")
ylim(0,2)
xlim(0, 2nside)
ylabel(raw"$\hat{C}_{\ell} / C_{\ell}^{\mathrm{theory}}$")
xlabel(raw"Multipole moment, $\ell$")
# plot(1e-2 .* theory.ell.^2, "-")
gcf()

##


##

using Statistics
clf()
plot( (Statistics.var(sims, dims=2) ./ diag(C.parent))[2:2nside] )
# plot( 1 ./ wl.parent .^ 4)
ylabel(raw"$\mathrm{Var}^{\mathrm{sim}}(C_{\ell}) / \mathrm{Var}^{\mathrm{analytic}}(C_{\ell})$")
xlabel(raw"Multipole moment, $\ell$")
# yscale("log")
gcf()



## 
nmt = pyimport("pymaster")
f_0 = nmt.NmtField(mask.pixels, [mask.pixels])
b = nmt.NmtBin.from_nside_linear(nside, 1)
w = nmt.NmtWorkspace()
##

@time w.compute_coupling_matrix(f_0, f_0, b)
##

##

cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f_0, f_0, f_0, f_0)
n_ell = length(b.get_effective_ells())
covar_00_00 = nmt.gaussian_covariance(cw,
                                      0, 0, 0, 0,  # Spins of the 4 fields
                                      [cltt],  # TT
                                      [cltt],  # TT
                                      [cltt],  # TT
                                      [cltt],  # TT
                                      w, wb=w)

##

using Statistics
clf()

# plot( ( diag(covar_00_00) ./ diag(C.parent)[3:(3nside)])[2:2nside], 
#     label=raw"$\mathrm{Var}^{\mathrm{nmt}}(C_{\ell}) / \mathrm{Var}^{\mathrm{planck}}(C_{\ell})$")

plot( (Statistics.var(sims, dims=2) ./ diag(C.parent))[1:2nside], 
    label=raw"$\mathrm{Var}^{\mathrm{sim}}(C_{\ell}) / \mathrm{Var}^{\mathrm{planck}}(C_{\ell})$")
plot( 2:(2nside+1), (Statistics.var(sims, dims=2)[3:(3nside)] ./ diag(covar_00_00))[1:2nside], 
    label=raw"$\mathrm{Var}^{\mathrm{sim}}(C_{\ell}) / \mathrm{Var}^{\mathrm{nmt}}(C_{\ell})$")
legend()
xlabel(raw"Multipole moment, $\ell$")
ylim(0.5,1.5)
# xlim(0,50)
# yscale("log")
gcf()


##
# M .*= sqrt( sum(mask.pixels) / nside2npix(nside) )

# using LinearAlgebra

# m.pixels .*= mask.pixels
# Cl_hat = (alm2cl(map2alm(m; niter=3)))
# Cl_hat = inv(Symmetric(M.parent)) * Cl_hat

# println(sum(Cl_hat[100:300]) / 200)
##
using PyPlot
clf()
axhline(1)
plot(Cl_hat .* (1:length(Cl_hat)).^2 , alpha=0.5, color="C2")
ylim(0,2)
gcf()

##
clf()
hp.mollview(m.pixels)
gcf()


##
using LinearAlgebra
using PyPlot
plt.clf()
plt.plot(diag(M.parent,4))
plt.gcf()
##

AngularPowerSpectra.effective_weights!(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
AngularPowerSpectra.W_spectra!(workspace)

using BenchmarkTools
@btime cov($workspace, $m_143_hm1, $m_143_hm2, $m_143_hm1, $m_143_hm2)
##
