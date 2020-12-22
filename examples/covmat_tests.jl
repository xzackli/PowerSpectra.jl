ENV["OMP_NUM_THREADS"] = 6

using AngularPowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
using BenchmarkTools, DelimitedFiles
using NPZ
hp = pyimport("healpy")
nmt = pyimport("pymaster")
np = pyimport("numpy")

nside = 256
mask1_T = readMapFromFITS("test/data/mask1_T.fits", 1, Float64) 
mask2_T = readMapFromFITS("test/data/mask2_T.fits", 1, Float64)

# mask2_T.pixels .= 0.9
mask1_P = deepcopy(mask1_T)#readMapFromFITS("test/data/mask1_P.fits", 1, Float64)
# mask1_P.pixels .= 0.6
mask2_P = deepcopy(mask1_T)#readMapFromFITS("test/data/mask2_P.fits", 1, Float64)
# mask2_P.pixels .= 0.7
var = Map{Float64, RingOrder}(ones(nside2npix(nside)))
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )

beam1 = SpectralVector(ones(3nside))
beam2 = SpectralVector(ones(3nside))
theory = CSV.read("test/data/theory.csv", DataFrame)
noise = CSV.read("test/data/noise.csv", DataFrame)


f0_1 = nmt.NmtField(mask1_T.pixels, [flat_mask.pixels])
f0_2 = nmt.NmtField(mask2_T.pixels, [flat_mask.pixels])
f2_1 = nmt.NmtField(mask1_P.pixels, [flat_mask.pixels, flat_mask.pixels])
f2_2 = nmt.NmtField(mask2_P.pixels, [flat_mask.pixels, flat_mask.pixels])

b = nmt.NmtBin.from_nside_linear(nside, 1)
w00 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(f0_1, f0_2, b)
w02 = nmt.NmtWorkspace()
w02.compute_coupling_matrix(f0_1, f2_2, b)
w22 = nmt.NmtWorkspace()
w22.compute_coupling_matrix(f2_1, f2_2, b)

cw = nmt.NmtCovarianceWorkspace()
cw.compute_coupling_coefficients(f0_1, f0_2, f0_1, f0_2)
n_ell = 3*nside
cl_tt = theory.cltt
cl_te = theory.clte
cl_ee = theory.clee
nl_ee = noise.nlee
nl_tt = noise.nltt
zero_cl = zeros(length(theory.clee))
cl_bb = zero_cl
cl_tb = zero_cl


@time covar_00_00 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_tt .+ nl_tt],
                                      [cl_tt],
                                      [cl_tt],
                                      [cl_tt .+ nl_tt],
                                      w00, wb=w00, coupled=true)
covar_coupled_TTTT = (np.reshape(covar_00_00, (n_ell, 1, n_ell, 1)))[:, 1, :, 1]

# @time covar_02_22 = nmt.gaussian_covariance(cw, 0, 2, 2, 2,  # Spins of the 4 fields
#                                       [cl_te, cl_tb],
#                                       [cl_te, cl_tb],
#                                       [cl_ee, cl_te,
#                                       cl_te, cl_bb],  # EE, EB, BE, BB
#                                       [cl_ee .+ nl_ee, cl_te,
#                                       cl_te, cl_bb],  # EE, EB, BE, BB
#                                       w02, wb=w22, coupled=true)
# covar_coupled_TEEE = (np.reshape(covar_02_22, (n_ell, 2, n_ell, 4)))[:, 1, :, 1]


import AngularPowerSpectra: TT, TE, EE, PP, QQ, UU
import DataStructures: DefaultDict

N_white = 4π / nside2npix(nside)
identity_spectrum = SpectralVector(ones(3nside));

cltt = SpectralVector(convert(Vector, theory.cltt))
clte = SpectralVector(convert(Vector, theory.clte))
clee = SpectralVector(convert(Vector, theory.clee))
nltt = SpectralVector(convert(Vector, noise.nltt))
nlee = SpectralVector(convert(Vector, noise.nlee))

r_coeff = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
    (TT, "143_hm1", "143_hm1") => sqrt.(nltt ./ N_white),
    (TT, "143_hm1", "143_hm2") => identity_spectrum,
    (TT, "143_hm2", "143_hm1") => identity_spectrum,
    (TT, "143_hm2", "143_hm2") => sqrt.(nltt ./ N_white),

    (EE, "143_hm1", "143_hm1") => sqrt.(nlee ./ N_white),
    (EE, "143_hm1", "143_hm2") => identity_spectrum,
    (EE, "143_hm2", "143_hm1") => identity_spectrum,
    (EE, "143_hm2", "143_hm2") => sqrt.(nlee ./ N_white)
)

spectra = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
    (TT, "143_hm1", "143_hm1") => cltt,
    (TT, "143_hm1", "143_hm2") => cltt,
    (TT, "143_hm2", "143_hm1") => cltt,
    (TT, "143_hm2", "143_hm2") => cltt,

    (EE, "143_hm1", "143_hm1") => clee,
    (EE, "143_hm1", "143_hm2") => clee,
    (EE, "143_hm2", "143_hm1") => clee,
    (EE, "143_hm2", "143_hm2") => clee,
    
    (TE, "143_hm1", "143_hm1") => clte,
    (TE, "143_hm1", "143_hm2") => clte,
    (TE, "143_hm2", "143_hm1") => clte,
    (TE, "143_hm2", "143_hm2") => clte,
)


m_143_hm1 = PolarizedField("143_hm1", mask1_T, mask1_P, var, var, var, beam1, beam1)
m_143_hm2 = PolarizedField("143_hm2", mask2_T, mask2_P, var, var, var, beam2, beam2)
workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
@time C = AngularPowerSpectra.compute_coupled_covmat_TTTT(workspace, spectra, r_coeff,
                         m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);
reference_covar = covar_coupled_TTTT


clf()
plt.plot( (diag(C.parent) ./ diag(reference_covar) )[30:end])
plt.plot()
ylim(0.99, 1.01)
# yscale("log")
gcf()

##
clf()
for k in [0, 10, 20, 30, 40, 50]

    plt.plot(
        diag(reference_covar[3:2nside,3:2nside],k)
        ./
        diag(C.parent[3:2nside, 3:2nside],k), 
        "-", label=k)
end
legend()
ylim(0.9, 1.1)
gcf()

##
clf()
k = 50
plt.plot(
    diag(reference_covar[3:2nside,3:2nside],k),
    "-")
plt.plot(
    diag(C.parent[3:2nside, 3:2nside],k), 
    "--")
yscale("log")
gcf()

##


##

# @time mcm = compute_mcm_EE(workspace, "143_hm1", "143_hm2")
# @time mcm = compute_mcm_TE(workspace, "143_hm1", "143_hm2")
# @time factorized_mcm = lu(mcm.parent');
# @time C_EEEE = AngularPowerSpectra.compute_coupled_covmat_EEEE(workspace, spectra, r_coeff,
#                          m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);

##
# using NPZ
# reference_covar_EE_EE = npzread("test/data/covar_TE_EE.npy")

# using Test
# @test all((diag(C_EEEE.parent) .≈ diag(reference_covar_EE_EE))[3:end])

# ##
# clf()
# plt.plot(diag(reference_covar_EE_EE), "-")
# plt.plot(diag(C_EEEE.parent), "--")
# yscale("log")
# gcf()


##


##


##



##
clf()
plt.plot( (diag(C_EEEE.parent) ./ diag(reference_covar_EE_EE) .- 1 ))
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