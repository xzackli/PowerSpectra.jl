using AngularPowerSpectra
using Healpix
using PyCall, PyPlot
using CSV, DataFrames, LinearAlgebra
using BenchmarkTools
hp = pyimport("healpy")
nmt = pyimport("pymaster")

##
nside = 256
mask1 = readMapFromFITS("notebooks/data/mask1.fits", 1, Float64)
mask2 = readMapFromFITS("notebooks/data/mask2.fits", 1, Float64)

Bl1 = CSV.read("notebooks/data/beam1.csv";).Bl
Bl2 = CSV.read("notebooks/data/beam2.csv";).Bl
beam1 = SpectralVector(Bl1 .* hp.pixwin(nside))
beam2 = SpectralVector(Bl2 .* hp.pixwin(nside))

theory = CSV.read("notebooks/data/theory.csv";)
noise = CSV.read("notebooks/data/noise.csv";)

##
flat_mask = Map{Float64, RingOrder}(ones(nside2npix(nside)) )
m_143_hm1 = Field("143_hm1", mask1, flat_mask, beam1)
m_143_hm2 = Field("143_hm2", mask2, flat_mask, beam2)
workspace = SpectralWorkspace(m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2)
@time mcm = compute_mcm(workspace, "143_hm1", "143_hm2")
@time factorized_mcm = cholesky(Hermitian(mcm.parent));

##
function get_sim()
    pixwin = true
    signal = hp.synfast(theory.cltt, 
        nside, verbose=false, pixwin=pixwin, new=true)
    n1 = hp.synfast(noise.nltt, 
        nside, verbose=false, pixwin=pixwin, new=true)
    n2 = hp.synfast(noise.nltt, 
        nside, verbose=false, pixwin=pixwin, new=true)
    map1 = Map{Float64, RingOrder}(nside)
    map1.pixels .= hp.sphtfunc.smoothing(signal .+ n1, beam_window=Bl1, verbose=false)
    map2 = Map{Float64, RingOrder}(nside)
    map2.pixels .= hp.sphtfunc.smoothing(signal .+ n2, beam_window=Bl2, verbose=false)
    Clhat = compute_spectra(map1 * mask1, map2 * mask2, factorized_mcm, beam1, beam2)
    return Clhat
end

function generate_sim_array(nsims)

    result = Array{Float64, 2}(undef, (3 * nside, nsims))
    for i in 1:nsims
        Clhat = get_sim()
        result[:,i] .= Clhat
    end

    return result
end

sims = generate_sim_array(1000)

using JLD2
@save "sims.jld2" sims

##


##
import AngularPowerSpectra: TT

cltt = SpectralVector(convert(Vector, theory.cltt))
nltt = SpectralVector(convert(Vector, noise.nltt))

spectra = Dict{AngularPowerSpectra.VIndex, SpectralVector{Float64, Vector{Float64}}}(
    (TT, "143_hm1", "143_hm1") => cltt .+ nltt,
    (TT, "143_hm1", "143_hm2") => cltt,
    (TT, "143_hm2", "143_hm1") => cltt,
    (TT, "143_hm2", "143_hm2") => cltt .+ nltt)
@time C = compute_covmat(workspace, spectra, factorized_mcm, factorized_mcm,
                         m_143_hm1, m_143_hm2, m_143_hm1, m_143_hm2);


##


using Statistics
clf()
plot( (Statistics.var(sims, dims=2) ./ diag(C.parent))[2:2nside] )
# plot( 1 ./ wl.parent .^ 4)
ylabel(raw"$\mathrm{Var}^{\mathrm{sim}}(C_{\ell}) / \mathrm{Var}^{\mathrm{analytic}}(C_{\ell})$")
xlabel(raw"Multipole moment, $\ell$")
# yscale("log")
gcf()


##

##

using Statistics
clf()
plot( (Statistics.mean(sims, dims=2) ./ theory.cltt[1:768])[2:2nside] )
# plot( 1 ./ wl.parent .^ 4)
ylabel(raw"$\mathrm{Var}^{\mathrm{sim}}(C_{\ell}) / \mathrm{Var}^{\mathrm{analytic}}(C_{\ell})$")
xlabel(raw"Multipole moment, $\ell$")
# yscale("log")
gcf()

