using AngularPowerSpectra
using Healpix
using StableRNGs

@testset "binning matrix" begin
    b = binning_matrix([0,1,2], [0,1,3], ℓ->1; lmax=4)
    ref = [
        1.0  0.0  0.0  0.0
        0.0  1.0  0.0  0.0
        0.0  0.0  0.5  0.5
    ]
    @test all(ref .≈ b)
end

@testset "commented header reader" begin
    t = read_commented_header("data/commented_header.txt")
    @test all(t[!, "a"] .≈ [1.0, 2.0])
    @test all(t[!, "b"] .≈ [2.0, 3.0])
end

##


@testset "synalm" begin
    rng = StableRNG(123)
    nside = 16
    nsims = 100
    C0 = [3.  2.;  2.  5.]
    Cl = repeat(C0, 1, 1, 3nside);
    cls11 = Array{Float64, 2}(undef, (3nside, nsims))
    cls12 = Array{Float64, 2}(undef, (3nside, nsims))
    cls22 = Array{Float64, 2}(undef, (3nside, nsims))
    for i in 1:nsims
        alms = synalm(rng, Cl, nside)
        cls11[:,i] .= alm2cl(alms[1], alms[1])
        cls12[:,i] .= alm2cl(alms[1], alms[2])
        cls22[:,i] .= alm2cl(alms[2], alms[2])
    end
    @test isapprox(sum(cls11) ./ (nsims * (3nside)), 3.0, rtol=0.05)
    @test isapprox(sum(cls12) ./ (nsims * (3nside)), 2.0, rtol=0.05)
    @test isapprox(sum(cls22) ./ (nsims * (3nside)), 5.0, rtol=0.05)
end
