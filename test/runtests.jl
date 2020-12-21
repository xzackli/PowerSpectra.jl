using AngularPowerSpectra
using Healpix
using CSV
using Test
using LinearAlgebra
using DataFrames
using DelimitedFiles
using NPZ
import AngularPowerSpectra: TT, TE, EE

##
@testset "Mode Coupling Matrix TT" begin
    nside = 256
    mask = readMapFromFITS("test/data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
    @time mcm = compute_mcm_TT(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = lu(mcm.parent)

    reference = readdlm("test/data/mcm_TT_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])

    map1 = readMapFromFITS("test/data/example_map.fits", 1, Float64)
    Cl_hat = compute_spectra(map1 * mask, map1 * mask, factorized_mcm, flat_beam, flat_beam)
    reference_spectrum = readdlm("test/data/example_TT_spectrum.txt")
end

##
@testset "Mode Coupling Matrix EE" begin
    nside = 256
    mask = readMapFromFITS("test/data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
    @time mcm = compute_mcm_EE(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = lu(mcm.parent)

    reference = readdlm("test/data/mcm_EE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])
end

##
@testset "Mode Coupling Matrix TE" begin
    nside = 256
    mask = readMapFromFITS("test/data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
    @time mcm = compute_mcm_TE(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = lu(mcm.parent)
    reference = readdlm("test/data/mcm_TE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])

    @time mcm = compute_mcm_ET(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = lu(mcm.parent)
    reference = readdlm("test/data/mcm_TE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])
end

##

@testset "Covariance Matrix Homogenous Noise" begin
    nside = 256
    mask1_T = readMapFromFITS("test/data/mask1_T.fits", 1, Float64)
    mask2_T = readMapFromFITS("test/data/mask2_T.fits", 1, Float64)
    mask1_P = readMapFromFITS("test/data/mask1_P.fits", 1, Float64)
    mask2_P = readMapFromFITS("test/data/mask2_P.fits", 1, Float64)
    unit_var = Map{Float64, RingOrder}(ones(nside2npix(nside)))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    beam1 = SpectralVector(ones(3nside))
    beam2 = SpectralVector(ones(3nside))
    theory = CSV.read("test/data/theory.csv", DataFrame)
    noise = CSV.read("test/data/noise.csv", DataFrame)
    identity_spectrum = SpectralVector(ones(3nside));

    cltt = SpectralVector(convert(Vector, theory.cltt))
    clte = SpectralVector(convert(Vector, theory.clte))
    clee = SpectralVector(convert(Vector, theory.clee))
    nlee = SpectralVector(convert(Vector, noise.nlee))
    nltt = SpectralVector(convert(Vector, noise.nltt))

    # this test specifies a map with unit variance. the corresponding white noise level is divided out in r_coeff
    N_white = 4π / nside2npix(nside)
    r_coeff = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
        (TT, "143_hm1", "143_hm1") => sqrt.(nltt ./ N_white),
        (TT, "143_hm1", "143_hm2") => identity_spectrum,
        (TT, "143_hm2", "143_hm1") => identity_spectrum,
        (TT, "143_hm2", "143_hm2") => sqrt.(nltt ./ N_white),

        (EE, "143_hm1", "143_hm1") => sqrt.(nlee ./ N_white),
        (EE, "143_hm1", "143_hm2") => identity_spectrum,
        (EE, "143_hm2", "143_hm1") => identity_spectrum,
        (EE, "143_hm2", "143_hm2") => sqrt.(nlee ./ N_white))

    spectra = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
        (TT, "143_hm1", "143_hm1") => cltt,
        (TT, "143_hm1", "143_hm2") => cltt,
        (TT, "143_hm2", "143_hm1") => cltt,
        (TT, "143_hm2", "143_hm2") => cltt,

        (EE, "143_hm1", "143_hm1") => clee,
        (EE, "143_hm1", "143_hm2") => clee,
        (EE, "143_hm2", "143_hm1") => clee,
        (EE, "143_hm2", "143_hm2") => clee ,
        
        (TE, "143_hm1", "143_hm1") => clte,
        (TE, "143_hm1", "143_hm2") => clte,
        (TE, "143_hm2", "143_hm1") => clte,
        (TE, "143_hm2", "143_hm2") => clte,
    )

    m_143_hm1 = PolarizedField("143_hm1", mask1_T, mask1_P, unit_var, unit_var, unit_var, beam1, beam1)
    m_143_hm2 = PolarizedField("143_hm2", mask2_T, mask2_P, unit_var, unit_var, unit_var, beam2, beam2)
    workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)

    @time C = AngularPowerSpectra.compute_coupled_covmat_TTTT(workspace, spectra, r_coeff,
        m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);
    reference_covar = npzread("test/data/covar_TT_TT.npy")
    @test all((diag(C.parent) .≈ diag(reference_covar))[3:end])

    @time C = AngularPowerSpectra.compute_coupled_covmat_TTTE(workspace, spectra, r_coeff,
        m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);
    reference_covar = npzread("test/data/covar_TT_TE.npy")
    @test all((diag(C.parent) .≈ diag(reference_covar))[3:end])

    @time C = AngularPowerSpectra.compute_coupled_covmat_TETE(workspace, spectra, r_coeff,
        m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);
    reference_covar = npzread("test/data/covar_TE_TE.npy")
    @test all((diag(C.parent) .≈ diag(reference_covar))[3:end])

    @time C = AngularPowerSpectra.compute_coupled_covmat_TTEE(workspace, spectra, r_coeff,
        m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);
    reference_covar = npzread("test/data/covar_TT_EE.npy")
    @test all((diag(C.parent) .≈ diag(reference_covar))[3:end])

    @time C = AngularPowerSpectra.compute_coupled_covmat_TEEE(workspace, spectra, r_coeff,
        m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);
    reference_covar = npzread("test/data/covar_TE_EE.npy")
    @test all((diag(C.parent) .≈ diag(reference_covar))[3:end])

    @time C = AngularPowerSpectra.compute_coupled_covmat_EEEE(workspace, spectra, r_coeff,
        m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);
    reference_covar = npzread("test/data/covar_EE_EE.npy")
    @test all((diag(C.parent) .≈ diag(reference_covar))[3:end])
end

##