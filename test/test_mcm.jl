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
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = SpectralWorkspace(m_143_hm1, m_143_hm2)
    mcm = compute_mcm_TT(workspace, "143_hm1", "143_hm2")
    factorized_mcm = lu(mcm.parent)

    reference = readdlm("data/mcm_TT_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])

    map1 = readMapFromFITS("data/example_map.fits", 1, Float64)
    Cl_hat = compute_spectra(map1 * mask, map1 * mask, factorized_mcm, flat_beam, flat_beam)
    reference_spectrum = readdlm("data/example_TT_spectrum.txt")
    @test all(reference_spectrum .≈ Cl_hat[3:end])
end

##
@testset "Mode Coupling Matrix Diag EE" begin
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = SpectralWorkspace(m_143_hm1, m_143_hm2)
    mcm = compute_mcm_EE(workspace, "143_hm1", "143_hm2")
    factorized_mcm = lu(mcm.parent)

    reference = readdlm("data/mcm_EE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])
end

##
@testset "Mode Coupling Matrix Diag TE/ET" begin
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = SpectralWorkspace(m_143_hm1, m_143_hm2)
    mcm = compute_mcm_TE(workspace, "143_hm1", "143_hm2")
    factorized_mcm = lu(mcm.parent)
    reference = readdlm("data/mcm_TE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])

    mcm = compute_mcm_ET(workspace, "143_hm1", "143_hm2")
    factorized_mcm = lu(mcm.parent)
    reference = readdlm("data/mcm_TE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])
end

##
@testset "Full Non-Trivial MCM" begin
    nside = 256
    mask1_T = readMapFromFITS("data/mask1_T.fits", 1, Float64)
    mask2_T = readMapFromFITS("data/mask2_T.fits", 1, Float64)
    mask1_P = readMapFromFITS("data/mask1_P.fits", 1, Float64)
    mask2_P = readMapFromFITS("data/mask2_P.fits", 1, Float64)
    unit_map = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    unit_beam = SpectralVector(ones(3*nside))
    m_143_hm1 = PolarizedField("143_hm1", mask1_T, mask1_P, unit_map, unit_map, unit_map, unit_beam, unit_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask2_T, mask2_P, unit_map, unit_map, unit_map, unit_beam, unit_beam)
    workspace = SpectralWorkspace(m_143_hm1, m_143_hm2)

    mcm = compute_mcm_TT(workspace, "143_hm1", "143_hm2")
    reference_mcm = npzread("data/mcmTT.npy")
    @test all(isapprox(mcm.parent[3:end, 3:end], reference_mcm[3:end, 3:end], atol=1e-11))

    mcm = compute_mcm_TE(workspace, "143_hm1", "143_hm2")
    reference_mcm = npzread("data/mcmTE.npy")
    for k in 0:3nside
        @test all(isapprox(diag(mcm.parent, k)[3:end], diag(reference_mcm, k)[3:end]))
    end

    mcm = compute_mcm_ET(workspace, "143_hm1", "143_hm2")
    reference_mcm = npzread("data/mcmET.npy")

    for k in 0:3nside
        @test all(isapprox(diag(mcm.parent, k)[3:end], diag(reference_mcm, k)[3:end]))
    end

    mcm = compute_mcm_EE(workspace, "143_hm1", "143_hm2")
    reference_mcm = npzread("data/mcmEE.npy")
    for k in 0:3nside
        @test all(isapprox(diag(mcm.parent, k)[3:end], diag(reference_mcm, k)[3:end]))
    end
end
