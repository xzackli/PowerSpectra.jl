## load in the data
using Test
using PowerSpectra
using Healpix
using JLD2
using DelimitedFiles


@testset "Planck 100 GHz MCM" begin
    nside = 256
    lmax = nside2lmax(nside)

    maskT₁ = PowerSpectra.planck256_mask("100", "hm1", "T")
    maskP₁ = PowerSpectra.planck256_mask("100", "hm1", "P")
    maskT₂ = PowerSpectra.planck256_mask("100", "hm2", "T")
    maskP₂ = PowerSpectra.planck256_mask("100", "hm2", "P")
    flat_map = HealpixMap{Float64, RingOrder}(ones(nside2npix(nside)) )

    M₀₀ = mcm(:TT, map2alm(maskT₁), map2alm(maskT₂))
    @load "data/mcm00.jld2" mcm00
    @test mcm00 ≈ parent(M₀₀)
    ##
    @load "data/mcm20.jld2" mcm20
    A = mcm20[1:2:2*lmax+1, 1:2:2*lmax+1]
    M₂₀ = mcm(:ET, map2alm(maskP₁), map2alm(maskT₂))
    @test A[3:end, 3:end] ≈ parent(M₂₀)[3:end, 3:end]
    M₂₀ = mcm(:BT, map2alm(maskP₁), map2alm(maskT₂))
    @test A[3:end, 3:end] ≈ parent(M₂₀)[3:end, 3:end]
    @load "data/mcm02.jld2" mcm02
    A = mcm02[1:2:2*lmax+1, 1:2:2*lmax+1]
    M₀₂ = mcm(:TE, map2alm(maskT₁), map2alm(maskP₂))
    @test A[3:end, 3:end] ≈ parent(M₀₂)[3:end, 3:end]
    M₀₂ = mcm(:TB, map2alm(maskT₁), map2alm(maskP₂))
    @test A[3:end, 3:end] ≈ parent(M₀₂)[3:end, 3:end]

    ##
    @load "data/mcm22.jld2" mcm22
    M⁺⁺ = mcm(:M⁺⁺, map2alm(maskP₁), map2alm(maskP₂))
    M⁻⁻ = mcm(:M⁻⁻, map2alm(maskP₁), map2alm(maskP₂))

    ##
    A = mcm22[1:4:4*(lmax+1), 1:4:4*(lmax+1)]
    @test A[3:end, 3:end] ≈ parent(M⁺⁺)[3:(lmax+1), 3:(lmax+1)]

    ##
    Δ₁, Δ₂ = 1, 1
    A = mcm22[1+Δ₁:4:4*(lmax+1)+Δ₁, 1+Δ₂:4:4*(lmax+1)+Δ₂]
    @test A[3:end, 3:end] ≈ parent(M⁺⁺)[3:(lmax+1), 3:(lmax+1)]

    Δ₁, Δ₂ = 2, 2
    A = mcm22[1+Δ₁:4:4*(lmax+1)+Δ₁, 1+Δ₂:4:4*(lmax+1)+Δ₂]
    @test A[3:end, 3:end] ≈ parent(M⁺⁺)[3:(lmax+1), 3:(lmax+1)]

    Δ₁, Δ₂ = 3, 3
    A = mcm22[1+Δ₁:4:4*(lmax+1)+Δ₁, 1+Δ₂:4:4*(lmax+1)+Δ₂]
    @test A[3:end, 3:end] ≈ parent(M⁺⁺)[3:(lmax+1), 3:(lmax+1)]

    Δ₁, Δ₂ = 1, 2
    A = mcm22[1+Δ₁:4:4*(lmax+1)+Δ₁, 1+Δ₂:4:4*(lmax+1)+Δ₂]
    @test A[3:end, 3:end] ≈ parent(-M⁻⁻)[3:(lmax+1), 3:(lmax+1)]
    ##
    Δ₁, Δ₂ = 2, 1
    A = mcm22[1+Δ₁:4:4*(lmax+1)+Δ₁, 1+Δ₂:4:4*(lmax+1)+Δ₂]
    @test A[3:end, 3:end] ≈ parent(-M⁻⁻)[3:(lmax+1), 3:(lmax+1)]

    Δ₁, Δ₂ = 0, 3
    A = mcm22[1+Δ₁:4:4*(lmax+1)+Δ₁, 1+Δ₂:4:4*(lmax+1)+Δ₂]
    @test A[3:end, 3:end] ≈ parent(M⁻⁻)[3:(lmax+1), 3:(lmax+1)]

    Δ₁, Δ₂ = 3, 0
    A = mcm22[1+Δ₁:4:4*(lmax+1)+Δ₁, 1+Δ₂:4:4*(lmax+1)+Δ₂]
    @test A[3:end, 3:end] ≈ parent(M⁻⁻)[3:(lmax+1), 3:(lmax+1)]
end


##
@testset "Planck 100 GHz MCM decoupling" begin
    nside = 256
    m₁ = PowerSpectra.planck256_polmap("100", "hm1")
    m₂ = PowerSpectra.planck256_polmap("100", "hm2")
    maskT₁ = PowerSpectra.planck256_mask("100", "hm1", :T)
    maskP₁ = PowerSpectra.planck256_mask("100", "hm1", :P)
    maskT₂ = PowerSpectra.planck256_mask("100", "hm2", :T)
    maskP₂ = PowerSpectra.planck256_mask("100", "hm2", :P)
    
    # convert to μK
    scale!(m₁, 1e6)
    scale!(m₂, 1e6)
    Cl = master(m₁, maskT₁, maskP₁,
                m₂, maskT₂, maskP₂; lmin=2)

    @load "data/planck_spec.jld2" cl_00 cl_02 cl_20 cl_22

    @test cl_00[1,:] ≈ Cl[:TT][2:end]

    @test cl_02[1,:] ≈ Cl[:TE][2:end]
    @test cl_02[2,:] ≈ Cl[:TB][2:end]
    @test cl_20[1,:] ≈ Cl[:ET][2:end]
    @test cl_20[2,:] ≈ Cl[:BT][2:end]

    @test cl_22[1,:] ≈ Cl[:EE][2:end]
    @test cl_22[2,:] ≈ Cl[:EB][2:end]
    @test cl_22[3,:] ≈ Cl[:BE][2:end]
    @test cl_22[4,:] ≈ Cl[:BB][2:end]
end
