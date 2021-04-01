using Test
using AngularPowerSpectra
using Healpix
using StableRNGs
using IdentityRanges
using LinearAlgebra

##
@testset "spectralones" begin
    x = spectralones(5)
    @test all(parent(x) .== ones(5))

    x = spectralones(31:35)
    @test all(parent(x) .== ones(5))
    @test all(x[35] == 1.0)

    x = spectralones(0:4)
    @test all(parent(x) .== ones(5))
    @test all(x[0] == 1.0)
end

@testset "spectralzeros" begin
    x = spectralzeros(5)
    @test all(parent(x) .== zeros(5))

    x = spectralzeros(31:35)
    @test all(parent(x) .== zeros(5))
    @test all(x[35] == 0.0)

    x = spectralzeros(0:4)
    @test all(parent(x) .== zeros(5))
    @test all(x[0] == 0.0)

    @test all(x .== SpectralVector(zeros(5), -1))
end

##
@testset "UnitRange and AbstractRange slicing" begin
    A = spectralones(5,5)
    @test typeof(A[0:2,2:4]) == typeof(parent(A))
    @test all(A[0:2,2:4] .== ones(3,3))
    @test typeof(A[IdentityRange(0:2),IdentityRange(2:4)]) == typeof(A)
    @test all(parent(A[IdentityRange(0:2),IdentityRange(2:4)]) .== ones(3,3))
end

##
@testset "SpectralArray typing and index style" begin
    A = spectralones(35:39,5:9)
    @test Base.IndexStyle(typeof(A)) == Base.IndexStyle(typeof(ones(5,5)))
    @test AngularPowerSpectra.parenttype(typeof(A)) == typeof(ones(5,5))
    @test AngularPowerSpectra.parenttype(A) == typeof(ones(5,5))
end

##
@testset "SpectralArray constructor" begin
    A = spectralones(35:39,5:9)
    @test all(A.parent.offsets .== (34, 4))
    A = SpectralArray(ones(5,5), 35:39,5:9)
    @test all(A.parent.offsets .== (34, 4))
    A = SpectralArray(ones(5,5), 34, 4)
    @test all(A.parent.offsets .== (34, 4))
    A = SpectralArray(ones(5,5), (34, 4))
    @test all(A.parent.offsets .== (34, 4))

    @test first(axes(SpectralVector(ones(5),1),1)) == 2
end

##
@testset "SpectralArray linear algebra" begin
    A = spectralzeros(5:10,15:20)
    parent(A) .= 2Matrix(LinearAlgebra.I, 6, 6)
    Ainv = inv(A)
    @test all(parent(A) .== 4parent(Ainv))
    @test all(A.parent.offsets .== Ainv.parent.offsets)
end
