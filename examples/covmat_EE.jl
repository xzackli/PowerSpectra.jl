using AngularPowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
using BenchmarkTools
using JLD2
hp = pyimport("healpy")
nmt = pyimport("pymaster")

nside = 256
mask1 = readMapFromFITS("notebooks/data/mask1.fits", 1, Float64)
mask2 = readMapFromFITS("notebooks/data/mask2.fits", 1, Float64)

Bl1 = CSV.read("notebooks/data/beam1.csv";).Bl
Bl2 = CSV.read("notebooks/data/beam2.csv";).Bl
beam1 = SpectralVector(Bl1 .* hp.pixwin(nside))
beam2 = SpectralVector(Bl2 .* hp.pixwin(nside))

theory = CSV.read("notebooks/data/theory.csv")
noise = CSV.read("notebooks/data/noise.csv")

fake_σ = readMapFromFITS("/media/data/wmap/ring/wmap_band_imap_r9_9yr_W_v5.fits", 1, Float64)

zero_var = Map{Float64, RingOrder}(1e-1 .* ones(nside2npix(nside)))
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
m_143_hm1 = PolarizedField("143_hm1", mask1, mask1, zero_var, zero_var, zero_var, beam1, beam1)
m_143_hm2 = PolarizedField("143_hm2", mask2, mask2, zero_var, zero_var, zero_var, beam2, beam2)
workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
@time mcm = compute_mcm_EE(workspace, "143_hm1", "143_hm2")
@time factorized_mcm = lu(mcm.parent);

##
import AngularPowerSpectra: EE, PP, QQ, UU
# i, j, p, q = "143_hm1", "143_hm2", "143_hm1", "143_hm2"
# AngularPowerSpectra.effective_weights_w!(workspace, m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
# AngularPowerSpectra.window_function_W!(workspace, PP, PP, i, p, PP, j, q, PP)


clee = SpectralVector(convert(Vector, theory.clee))
nlee = SpectralVector(convert(Vector, noise.nlee))

spectra = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
    (EE, "143_hm1", "143_hm1") => clee .+ nlee,
    (EE, "143_hm1", "143_hm2") => clee,
    (EE, "143_hm2", "143_hm1") => clee,
    (EE, "143_hm2", "143_hm2") => clee .+ nlee)
@time C = compute_covmat_EE(workspace, spectra, factorized_mcm, factorized_mcm,
                         m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);


                         
##
# hp = pyimport("healpy")
# nmt = pyimport("pymaster")
# f2_1 = nmt.NmtField(mask1.pixels, [flat_mask.pixels, flat_mask.pixels])
# f2_2 = nmt.NmtField(mask2.pixels, [flat_mask.pixels, flat_mask.pixels])

# b = nmt.NmtBin.from_nside_linear(nside, 1)
# w22 = nmt.NmtWorkspace()
# w22.compute_coupling_matrix(f2_1, f2_2, b)

# cw = nmt.NmtCovarianceWorkspace()
# cw.compute_coupling_coefficients(f2_1, f2_2, f2_1, f2_2)
# n_ell = length(b.get_effective_ells())
# cl_ee = theory.clee
# cl_eb = zeros(length(theory.clee))
# cl_bb = theory.clbb
# @time covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
#                                       [cl_ee .+ noise.nlee, cl_eb,
#                                        cl_eb, cl_bb .+ noise.nlee],  # EE, EB, BE, BB
#                                       [cl_ee, cl_eb,
#                                        cl_eb, cl_bb],  # EE, EB, BE, BB
#                                       [cl_ee, cl_eb,
#                                        cl_eb, cl_bb],  # EE, EB, BE, BB
#                                       [cl_ee .+ noise.nlee, cl_eb,
#                                        cl_eb, cl_bb .+ noise.nlee],  # EE, EB, BE, BB
#                                       w22, wb=w22)
# np = pyimport("numpy")
# cov = np.reshape(covar_22_22, (n_ell, 4, n_ell, 4))
# covar_EE_EE = cov[:, 1, :, 1]
# @save "covarEE.jld2"  covar_EE_EE 
@load "covarEE.jld2"  covar_EE_EE 

##
clf()
plt.plot(diag(covar_EE_EE), "-", label="NaMaster")
plt.plot(diag(C.parent)[3:768], "--", label="bespoke code")
plt.legend()

plt.ylabel(raw" Covmat Diagonal")
plt.xlabel(raw"Multipole Moment, $\ell$")
yscale("log")
gcf()

##
clf()
plt.plot( diag(covar_EE_EE) ./  (diag(C.parent)[3:768] ), "-", label="NaMaster / Bespoke Code" )
plt.legend()
plt.ylabel(raw"Ratio of Covmat Diagonal")
plt.xlabel(raw"Multipole Moment, $\ell$")
gcf()

##


##
using JLD2
# sims = generate_sim_array(100)
# @save "sims.jld2" sims
@load "sims.jld2" sims


##
using Statistics
clf()
plot( (Statistics.var(sims, dims=2) ./ diag(C.parent))[2:2nside] )
# plot( 1 ./ wl.parent .^ 4)
ylabel(raw"$\mathrm{Var}^{\mathrm{sim}}(C_{\ell}) / \mathrm{Var}^{\mathrm{analytic}}(C_{\ell})$")
xlabel(raw"Multipole moment, $\ell$")
# yscale("log")
ylim(0,2)
gcf()


##
clf(); plot(nltt); gcf()
##

using Statistics
clf()
plot( (Statistics.mean(sims, dims=2) ./ theory.cltt[1:768])[2:2nside] )
# plot( 1 ./ wl.parent .^ 4)
ylabel(raw"$\mathrm{Var}^{\mathrm{sim}}(C_{\ell}) / \mathrm{Var}^{\mathrm{analytic}}(C_{\ell})$")
xlabel(raw"Multipole moment, $\ell$")
# yscale("log")
gcf()

