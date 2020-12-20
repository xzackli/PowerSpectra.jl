using AngularPowerSpectra
using Healpix
using CSV
using Test
using DelimitedFiles
using LinearAlgebra

##
@testset "Mode Coupling Matrix TT" begin
    nside = 256
    mask = readMapFromFITS("test/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
    @time mcm = compute_mcm_TT(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = lu(mcm.parent)

    reference = readdlm("test/mcm_TT_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])

    map1 = readMapFromFITS("test/example_map.fits", 1, Float64)
    Cl_hat = compute_spectra(map1 * mask, map1 * mask, factorized_mcm, flat_beam, flat_beam)
    reference_spectrum = readdlm("test/example_TT_spectrum.txt")
end

##
@testset "Mode Coupling Matrix EE" begin
    nside = 256
    mask = readMapFromFITS("test/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
    @time mcm = compute_mcm_EE(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = lu(mcm.parent)

    reference = readdlm("test/mcm_EE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])
end

##
@testset "Mode Coupling Matrix TE" begin
    nside = 256
    mask = readMapFromFITS("test/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
    @time mcm = compute_mcm_TE(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = lu(mcm.parent)
    reference = readdlm("test/mcm_TE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])

    @time mcm = compute_mcm_ET(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = lu(mcm.parent)
    reference = readdlm("test/mcm_TE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])
end

##


##