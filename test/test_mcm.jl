using PowerSpectra
using Healpix
using CSV
using Test
using LinearAlgebra
using DataFrames
using DelimitedFiles
using NPZ
using IdentityRanges

##
@testset "Mode Coupling Matrix TT" begin
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_map = HealpixMap{Float64, RingOrder}(ones(nside2npix(nside)) )
    M = mcm(:TT, mask, mask; lmin=2)
    reference = readdlm("data/mcm_TT_diag.txt")
    @test all(reference .≈ diag(parent(M))[1:end-1])
    map1 = readMapFromFITS("data/example_map.fits", 1, Float64)
    pCl = SpectralVector(alm2cl(map2alm(map1 * mask)))
    Cl_hat = M \ pCl[IdentityRange(2:end)]
    reference_spectrum = readdlm("data/example_TT_spectrum.txt")
    @test all(reference_spectrum .≈ Cl_hat[2:end])
    @test all(reference_spectrum .≈ Cl_hat[2:end])
end

##
@testset "Mode Coupling Matrix Diag EE" begin
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    flat_mask = HealpixMap{Float64, RingOrder}(ones(nside2npix(nside)) )
    M = mcm(:M⁺⁺, mask, mask)
    # factorized_mcm12 = lu(parent(M))
    reference = readdlm("data/mcm_EE_diag.txt")
    @test all(reference .≈ diag(parent(M))[3:767])
end

##
@testset "Mode Coupling Matrix Diag TE/ET" begin
    nside = 256
    mask = readMapFromFITS("data/example_mask_1.fits", 1, Float64)
    M = mcm(:TE, mask, mask)
    reference = readdlm("data/mcm_TE_diag.txt")
    @test all(reference .≈ diag(parent(M))[3:767])

    M = mcm(:ET, mask, mask)
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
    unit_map = HealpixMap{Float64, RingOrder}(ones(nside2npix(nside)) )
    unit_beam = spectralones(0:(3nside-1))

    M = mcm(:TT, mask1_T, mask2_T)
    M_ref = npzread("data/mcmTT.npy")
    @test all(isapprox(parent(M)[3:end, 3:end], M_ref[3:end, 3:end], atol=1e-11))

    M = mcm(:TE, mask1_T, mask2_P)
    M_ref = npzread("data/mcmTE.npy")
    for k in 0:3nside
        @test all(isapprox(diag(parent(M), k)[3:end], diag(M_ref, k)[3:end]))
    end

    M = mcm(:ET, mask1_P, mask2_T)
    M_ref = npzread("data/mcmET.npy")

    for k in 0:3nside
        @test all(isapprox(diag(parent(M), k)[3:end], diag(M_ref, k)[3:end]))
    end

    M = mcm(:M⁺⁺, mask1_P, mask2_P)
    M_ref = npzread("data/mcmEE.npy")
    for k in 0:3nside
        @test all(isapprox(diag(parent(M), k)[3:end], diag(M_ref, k)[3:end]))
    end
end
