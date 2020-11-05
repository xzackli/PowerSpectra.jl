using AngularPowerSpectra
using Healpix
using CSV
using Test
using DelimitedFiles

@testset "Mode Coupling Matrix EE" begin
    # Write your tests here.
    
    mask = readMapFromFITS("test/example_mask.fits", 1, Float64)
    flat_beam = SpectralVector(ones(3*nside))
    flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
    m_143_hm1 = PolarizedField("143_hm1", mask, mask, flat_mask, flat_mask, flat_beam, flat_beam)
    m_143_hm2 = PolarizedField("143_hm2", mask, mask, flat_mask, flat_mask, flat_beam, flat_beam)
    workspace = PolarizedSpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
    @time mcm = compute_mcm_EE(workspace, "143_hm1", "143_hm2")
    @time factorized_mcm = cholesky(Hermitian(mcm.parent))

    reference = readdlm("test/mcm_EE_diag.txt")
    @test all(reference .≈ diag(mcm.parent)[3:767])
end
