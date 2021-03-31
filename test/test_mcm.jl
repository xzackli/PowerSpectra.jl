using AngularPowerSpectra
using Healpix
using CSV
using Test
using LinearAlgebra
using DataFrames
using DelimitedFiles
using NPZ

##
@testset "Mode Coupling Matrix TT" begin
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_map = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    # m1 = CovField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    # m2 = CovField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    M = mcm(:TT, mask, mask)
    reference = readdlm("data/mcm_TT_diag.txt")
    @test all(reference .≈ diag(parent(M))[3:767])
    # map1 = readMapFromFITS("data/example_map.fits", 1, Float64)
    pCl = SpectralVector(alm2cl(map2alm(flat_map)))
    # lu(M)
    # Cl_hat = M \ pCl
    # Cl_hat = map2cl(map1 * mask, map1 * mask, lu(parent(M)), flat_beam, flat_beam)
    # reference_spectrum = readdlm("data/example_TT_spectrum.txt")
    # @test all(reference_spectrum .≈ Cl_hat[3:end])
end

##
@testset "Mode Coupling Matrix Diag EE" begin
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m1 = CovField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m2 = CovField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    M = mcm(:EE, m1.maskP, m2.maskP)
    factorized_mcm12 = lu(parent(M))
    reference = readdlm("data/mcm_EE_diag.txt")
    @test all(reference .≈ diag(parent(M))[3:767])
end

##
@testset "Mode Coupling Matrix Diag TE/ET" begin
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m1 = CovField("143_hm1", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m2 = CovField("143_hm2", mask, mask, flat_mask, flat_mask, flat_mask, flat_beam, flat_beam)
    M = mcm(:TE, m1.maskT, m2.maskP)
    reference = readdlm("data/mcm_TE_diag.txt")
    @test all(reference .≈ diag(parent(M))[3:767])

    M = mcm(:ET, m1.maskP, m2.maskT)
    reference = readdlm("data/mcm_TE_diag.txt")
    @test all(reference .≈ diag(parent(M))[3:767])
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
    m1 = CovField("143_hm1", mask1_T, mask1_P, unit_map, unit_map, unit_map, unit_beam, unit_beam)
    m2 = CovField("143_hm2", mask2_T, mask2_P, unit_map, unit_map, unit_map, unit_beam, unit_beam)

    M = mcm(:TT, m1.maskT, m2.maskT)
    M_ref = npzread("data/mcmTT.npy")
    @test all(isapprox(parent(M)[3:end, 3:end], M_ref[3:end, 3:end], atol=1e-11))

    M = mcm(:TE, m1.maskT, m2.maskP)
    M_ref = npzread("data/mcmTE.npy")
    for k in 0:3nside
        @test all(isapprox(diag(parent(M), k)[3:end], diag(M_ref, k)[3:end]))
    end

    M = mcm(:ET, m1.maskP, m2.maskT)
    M_ref = npzread("data/mcmET.npy")

    for k in 0:3nside
        @test all(isapprox(diag(parent(M), k)[3:end], diag(M_ref, k)[3:end]))
    end

    M = mcm(:EE, m1.maskP, m2.maskP)
    M_ref = npzread("data/mcmEE.npy")
    for k in 0:3nside
        @test all(isapprox(diag(parent(M), k)[3:end], diag(M_ref, k)[3:end]))
    end
end
