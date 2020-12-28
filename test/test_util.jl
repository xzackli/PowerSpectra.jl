

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

@testset "Base.show Field" begin
    Base.show(io::IO, ::MIME"text/plain", x::Field{T})
end
