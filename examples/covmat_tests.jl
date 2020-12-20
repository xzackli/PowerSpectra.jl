ENV["OMP_NUM_THREADS"] = 6

using AngularPowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
using BenchmarkTools, DelimitedFiles
hp = pyimport("healpy")
nmt = pyimport("pymaster")
np = pyimport("numpy")

nside = 256
mask1 = readMapFromFITS("test/example_mask_1.fits", 1, Float64)
mask2 = readMapFromFITS("test/example_mask_2.fits", 1, Float64)
zero_var = Map{Float64, RingOrder}(zeros(nside2npix(nside)))
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )


Bl1 = CSV.read("notebooks/data/beam1.csv", DataFrame).Bl
Bl2 = CSV.read("notebooks/data/beam2.csv", DataFrame).Bl
beam1 = SpectralVector(Bl1 .* hp.pixwin(nside))
beam2 = SpectralVector(Bl2 .* hp.pixwin(nside))
theory = CSV.read("notebooks/data/theory.csv", DataFrame)
noise = CSV.read("notebooks/data/noise.csv", DataFrame)

##
f2_1 = nmt.NmtField(mask1.pixels, [flat_mask.pixels, flat_mask.pixels])
f2_2 = nmt.NmtField(mask2.pixels, [flat_mask.pixels, flat_mask.pixels])

b = nmt.NmtBin.from_nside_linear(nside, 1)
w22 = nmt.NmtWorkspace()
w22.compute_coupling_matrix(f2_1, f2_2, b)

cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f2_1, f2_2, f2_1, f2_2)
n_ell = length(b.get_effective_ells())
cl_tt = theory.cltt
cl_ee = theory.clee
nl_ee = noise.nlee
zero_cl = zeros(length(theory.clee))
cl_bb = theory.clbb

@time covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                      [cl_ee .+ nl_ee, zero_cl,
                                      zero_cl, nl_ee],  # EE, EB, BE, BB
                                      [cl_ee, zero_cl,
                                      zero_cl, zero_cl],  # EE, EB, BE, BB
                                      [cl_ee, zero_cl,
                                      zero_cl, zero_cl],  # EE, EB, BE, BB
                                      [cl_ee .+ nl_ee, zero_cl,
                                      zero_cl, nl_ee],  # EE, EB, BE, BB
                                      w22, wb=w22, coupled=true)
cov = np.reshape(covar_22_22, (3nside, 4, 3nside, 4))
covar_coupled_EEEE = cov[:, 1, :, 1]

##
@time covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                      [cl_ee .+ nl_ee, zero_cl,
                                      zero_cl, nl_ee],  # EE, EB, BE, BB
                                      [cl_ee, zero_cl,
                                      zero_cl, zero_cl],  # EE, EB, BE, BB
                                      [cl_ee, zero_cl,
                                      zero_cl, zero_cl],  # EE, EB, BE, BB
                                      [cl_ee .+ nl_ee, zero_cl,
                                      zero_cl, nl_ee],  # EE, EB, BE, BB
                                      w22, wb=w22, coupled=false)
covar_EE_EE = (np.reshape(covar_22_22, (n_ell, 4, n_ell, 4)))[:, 1, :, 1]

##
mcm_nmt = w22.get_coupling_matrix()[1:4:end, 1:4:end];

##



##
clf()
bespoke = (mcm_nmt[3:end, 3:end]) * covar_EE_EE * (mcm_nmt[3:end, 3:end]')
plt.plot(diag(covar_coupled_EEEE)[3:end])
plt.plot(diag(bespoke))
xlim(0, 10)
yscale("log")
gcf()

##
clf()
plt.plot( (diag(bespoke) ./ (diag(covar_coupled_EEEE)[3:end]))[30:end]  )
gcf()

##
import AngularPowerSpectra: TT, EE, PP, QQ, UU

identity_spectrum = SpectralVector(ones(3nside));
r_coeff = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
    (EE, "143_hm1", "143_hm1") => identity_spectrum,
    (EE, "143_hm1", "143_hm2") => identity_spectrum,
    (EE, "143_hm2", "143_hm1") => identity_spectrum,
    (EE, "143_hm2", "143_hm2") => identity_spectrum,
    (TT, "143_hm1", "143_hm1") => identity_spectrum,
    (TT, "143_hm1", "143_hm2") => identity_spectrum,
    (TT, "143_hm2", "143_hm1") => identity_spectrum,
    (TT, "143_hm2", "143_hm2") => identity_spectrum
)

cltt = SpectralVector(convert(Vector, theory.cltt))
clee = SpectralVector(convert(Vector, theory.clee))
nlee = SpectralVector(convert(Vector, noise.nlee))
nltt = SpectralVector(convert(Vector, noise.nltt))

spectra = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
    (EE, "143_hm1", "143_hm1") => clee .+ nlee,
    (EE, "143_hm1", "143_hm2") => clee,
    (EE, "143_hm2", "143_hm1") => clee,
    (EE, "143_hm2", "143_hm2") => clee .+ nlee,
    
    (TT, "143_hm1", "143_hm1") => cltt .+ nltt,
    (TT, "143_hm1", "143_hm2") => cltt,
    (TT, "143_hm2", "143_hm1") => cltt,
    (TT, "143_hm2", "143_hm2") => cltt .+ nltt
)

##
m_143_hm1 = PolarizedField("143_hm1", mask1, mask1, zero_var, zero_var, zero_var, beam1, beam1)
m_143_hm2 = PolarizedField("143_hm2", mask2, mask2, zero_var, zero_var, zero_var, beam2, beam2)
workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
@time mcm = compute_mcm_EE(workspace, "143_hm1", "143_hm2")
@time factorized_mcm = lu(mcm.parent');
@time C_EEEE = AngularPowerSpectra.compute_coupled_covmat_EEEE(workspace, spectra, r_coeff,
                         m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);

##
clf()
plt.plot(diag(covar_coupled_EEEE))
plt.plot(diag(C_EEEE.parent), "-")
yscale("log")
plt.xlim(0,40)
gcf()

##
clf()
ℓ₀ = 11
plt.plot((ℓ₀):3nside, -(diag(C_EEEE.parent) .- diag(covar_coupled_EEEE))[ℓ₀:end])
plt.plot()
# ylim(0.999, 1.001)
# yscale("log")
gcf()


##
















## TT TEST


##
b = nmt.NmtBin.from_nside_linear(nside, 1)
n_ell = length(b.get_effective_ells())
f0_1 = nmt.NmtField(mask1.pixels, [flat_mask.pixels])
f0_2 = nmt.NmtField(mask2.pixels, [flat_mask.pixels])
w00 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(f0_1, f0_2, b)
cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0_1, f0_2, f0_1, f0_2)
covar_00_00 = nmt.gaussian_covariance(cw,
                                      0, 0, 0, 0,  # Spins of the 4 fields
                                      [theory.cltt .+ noise.nltt],  # TT
                                      [theory.cltt],  # TT
                                      [theory.cltt],  # TT
                                      [theory.cltt .+ noise.nltt],  # TT
                                      w00, wb=w00)

covar_00_00 = np.reshape(covar_00_00, (n_ell, 1, n_ell, 1))
covar_TT_TT = covar_00_00[:, 1, :, 1]

##
@time mcm = compute_mcm_TT(workspace, "143_hm1", "143_hm2")
@time factorized_mcm = lu(mcm.parent');
@time C_TTTT = compute_covmat_TTTT(workspace, spectra, r_coeff, factorized_mcm, factorized_mcm,
                         m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);

##
clf()
plt.plot(diag(covar_TT_TT))
plt.plot(diag(C_TTTT.parent)[3:end], "-")
yscale("log")
gcf()

##
clf()
ℓ₀ = 2
plt.plot((ℓ₀ + 2):3nside, (diag(covar_TT_TT) ./ diag(C_TTTT.parent)[3:end])[ℓ₀:end])
gcf()

##