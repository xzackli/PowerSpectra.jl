using AngularPowerSpectra
using Healpix
using CSV
using Test
using LinearAlgebra
using DataFrames
using DelimitedFiles
using NPZ



@testset "Basic Covmat Mode Decoupling" begin
    A = [1.0 0.2 0.3; 0.4 2.0 0.15; 0.3 0.1 1.44]
    B1 = [1.2 0.6 0.1; 0.3 1.4 0.5; 0.44 0.2 1.3]
    B2 = [1.6 0.4 0.1; 0.3 1.4 0.9; 0.45 0.8 1.7]
    Cref = inv(B1) * (A) * (inv(B2)')
    C = decouple_covmat(SpectralArray(deepcopy(A)), SpectralArray(B1), SpectralArray(B2);
        lmin1=0, lmin2=0)
    @test all(Cref .≈ parent(C))

    A₀ = deepcopy(A)
    A₀[1,1] = 0.0
    Cref = inv(B1) * A₀ * (inv(B2)')
    C = decouple_covmat(SpectralArray(deepcopy(A)), SpectralArray(B1), SpectralArray(B2);
        lmin1=1, lmin2=1)
    @test all(Cref .≈ parent(C))
end

##
@testset "Covariance Matrix Diagonal in the Isotropic Noise Limit" begin
    nside = 256
    mask1_T = readMapFromFITS("data/mask1_T.fits", 1, Float64)
    mask2_T = readMapFromFITS("data/mask2_T.fits", 1, Float64)
    mask1_P = readMapFromFITS("data/mask1_P.fits", 1, Float64)
    mask2_P = readMapFromFITS("data/mask2_P.fits", 1, Float64)
    unit_var = Map{Float64, RingOrder}(ones(nside2npix(nside)))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    beam1 = SpectralVector(ones(3nside))
    beam2 = SpectralVector(ones(3nside))
    theory = CSV.read("data/theory.csv", DataFrame)
    noise = CSV.read("data/noise.csv", DataFrame)
    identity_spectrum = SpectralVector(ones(3nside));

    cltt = SpectralVector(convert(Vector, theory.cltt))
    clte = SpectralVector(convert(Vector, theory.clte))
    clee = SpectralVector(convert(Vector, theory.clee))
    nlee = SpectralVector(convert(Vector, noise.nlee))
    nltt = SpectralVector(convert(Vector, noise.nltt))

    # this test specifies a map with unit variance. the corresponding white noise level is divided out in r_coeff
    N_white = 4π / nside2npix(nside)
    r_coeff = Dict{AngularPowerSpectra.SpectrumName, SpectralVector{Float64, Vector{Float64}}}(
        (:TT, "143_hm1", "143_hm1") => sqrt.(nltt ./ N_white),
        (:TT, "143_hm1", "143_hm2") => identity_spectrum,
        (:TT, "143_hm2", "143_hm1") => identity_spectrum,
        (:TT, "143_hm2", "143_hm2") => sqrt.(nltt ./ N_white),

        (:EE, "143_hm1", "143_hm1") => sqrt.(nlee ./ N_white),
        (:EE, "143_hm1", "143_hm2") => identity_spectrum,
        (:EE, "143_hm2", "143_hm1") => identity_spectrum,
        (:EE, "143_hm2", "143_hm2") => sqrt.(nlee ./ N_white))

    spectra = Dict{AngularPowerSpectra.SpectrumName, SpectralVector{Float64, Vector{Float64}}}(
        (:TT, "143_hm1", "143_hm1") => cltt,
        (:TT, "143_hm1", "143_hm2") => cltt,
        (:TT, "143_hm2", "143_hm1") => cltt,
        (:TT, "143_hm2", "143_hm2") => cltt,

        (:EE, "143_hm1", "143_hm1") => clee,
        (:EE, "143_hm1", "143_hm2") => clee,
        (:EE, "143_hm2", "143_hm1") => clee,
        (:EE, "143_hm2", "143_hm2") => clee ,

        (:TE, "143_hm1", "143_hm1") => clte,
        (:TE, "143_hm1", "143_hm2") => clte,
        (:TE, "143_hm2", "143_hm1") => clte,
        (:TE, "143_hm2", "143_hm2") => clte,
    )

    m1 = CovField("143_hm1", mask1_T, mask1_P, unit_var, unit_var, unit_var, beam1, beam1)
    m2 = CovField("143_hm2", mask2_T, mask2_P, unit_var, unit_var, unit_var, beam2, beam2)
    workspace = CovarianceWorkspace(m1, m2, m1, m2)

    C = AngularPowerSpectra.compute_coupledcov_TTTT(workspace, spectra, r_coeff);
    C_ref = npzread("data/covar_TT_TT.npy")
    @test isapprox(diag(parent(C))[3:end], diag(C_ref)[3:end])

    C = AngularPowerSpectra.compute_coupledcov_TTTE(workspace, spectra, r_coeff);
    C_ref = npzread("data/covar_TT_TE.npy")
    @test isapprox(diag(parent(C))[3:end], diag(C_ref)[3:end])

    C = AngularPowerSpectra.compute_coupledcov_TETE(workspace, spectra, r_coeff);
    C_ref = npzread("data/covar_TE_TE.npy")
    @test isapprox(diag(parent(C))[3:end], diag(C_ref)[3:end])

    C = AngularPowerSpectra.compute_coupledcov_TTEE(workspace, spectra, r_coeff);
    C_ref = npzread("data/covar_TT_EE.npy")
    @test isapprox(diag(parent(C))[3:end], diag(C_ref)[3:end])

    C = AngularPowerSpectra.compute_coupledcov_TEEE(workspace, spectra, r_coeff; planck=false);
    C_ref = npzread("data/covar_TE_EE.npy")
    @test isapprox(diag(parent(C))[3:end], diag(C_ref)[3:end])

    C = AngularPowerSpectra.compute_coupledcov_EEEE(workspace, spectra, r_coeff);
    C_ref = npzread("data/covar_EE_EE.npy")
    @test isapprox(diag(parent(C))[3:end], diag(C_ref)[3:end])

    # test that planck approx is kind of close at high ell
    C = AngularPowerSpectra.compute_coupledcov_TEEE(workspace, spectra, r_coeff; planck=true);
    C_ref = npzread("data/covar_TE_EE.npy")
    @test isapprox(diag(parent(C))[30:end], diag(C_ref)[30:end], rtol=0.01)


    # test decoupling
    𝐌 = mcm(:EE, m1.maskP, m2.maskP)
    C_decoupled = decouple_covmat(C, 𝐌, 𝐌; lmin1=2, lmin2=2)

    # # by default, decouple has lmin=2
    @test isapprox(C[2:end,2:end],
        𝐌[2:end,2:end] * C_decoupled[2:end,2:end] * (𝐌[2:end,2:end]') )
end
